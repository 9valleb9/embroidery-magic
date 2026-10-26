"""Steps 4 and 7 — label mask -> clean shapely polygons, and per-shape stitch settings.

Where this sits in the flow:

    upload -> raster.py (rasterize) -> palette.py (colors, label map)
    -> vectorize.py (shapes, stitch settings) -> pipeline.py (sessions, color
    decisions, generation) -> svg_out.py (Ink/Stitch SVG) -> export.py (machine file)

pipeline.py builds one boolean *mask* per thread color (``labels == i`` on the
palette.py label map) and hands it to this module, which turns it into
geometry Ink/Stitch can sew, plus the Ink/Stitch parameters for each piece.
Nothing here knows about sessions, fabric names or SVG; it only takes masks
and numbers and returns shapes and ``{param: value}`` dicts (the keys are
Ink/Stitch's own attribute names, written by svg_out.py as ``inkstitch:<key>``).

Units and coordinates:
    * Masks are HxW arrays at ``raster.PX_PER_MM`` (20 px/mm). Pixel (y, x)
      maps to the mm point ((x + 0.5) / PX_PER_MM, (y + 0.5) / PX_PER_MM),
      i.e. the pixel's center.
    * All geometry returned (shapely polygons and lines, ``Satin`` rails) is
      in mm, with y pointing down as in SVG.
    * Skeleton chains (``_trace``) are lists of (y, x) pixel tuples.

The three stitch modes:

1. **Fills** (always). ``mask_to_polygons`` traces the mask into polygons
   with holes; ``stitch_params`` picks per polygon either
   ``fill_method="contour_fill"`` (thin shapes and rings, see
   ``is_thin_or_ring``) or ``fill_method="tatami_fill"`` with an ``angle``
   along the shape's long axis (``long_axis_angle``, 45 degrees for boxy
   shapes, see ``mostly_axis_aligned``), plus the fabric's density, underlay
   and pull compensation.

2. **Thin lines as running stitch** ("Stitch thin lines as outlines").
   ``split_thin`` removes features narrower than ``THIN_LINE_MM`` (1 mm) from
   the mask and returns their centerlines (from the mask's skeleton) as
   LineStrings, joined into long runs by ``_join_chains``. They are sewn with
   ``LINE_PARAMS``: ``stroke_method="running_stitch"``, 2 mm stitches,
   ``bean_stitch_repeats="1"`` (a triple stitch).

3. **Satin columns** ("Stitch narrow strokes as satin"). ``split_satin`` walks
   the skeleton of the (already thin-line-free) mask: ``_trace`` cuts it into
   chains between junctions, ``_join_smooth`` joins chains that continue each
   other with less than ``SATIN_MAX_TURN_DEG`` of turn, and every stretch of
   steady ``SATIN_MIN_MM``..``SATIN_MAX_MM`` width becomes a ``Satin`` via
   ``_column``. What the columns don't cover stays in the mask and is filled,
   except pieces too thin on average to fill (``split_strokes``), which join
   the hairlines below.
   In satin mode pipeline.py also passes each ``split_thin`` centerline to
   ``line_satin``, which turns hairlines into ``LINE_SATIN_MM`` (1 mm) satin
   columns so they match the lettering; lines it rejects stay running
   stitch. ``satin_params`` gives every column ``satin_column="true"``, the
   zig-zag spacing and underlay. svg_out.py writes a column as one path with
   two subpaths, ``Satin.rail1`` and ``Satin.rail2``, which have equal node
   counts so Ink/Stitch pairs them node by node without rungs.

Typical order per color (pipeline.py): ``split_thin`` -> ``split_satin`` ->
``line_satin`` on the lines -> ``mask_to_polygons`` on what is left ->
``stitch_params`` / ``satin_params`` / ``LINE_PARAMS`` at generation time.
``band`` makes the 1 mm outline and border layers.

Gotchas:
    * ``mask_to_polygons`` grows each mask by 1 px so neighboring colors
      overlap instead of leaving seams; the size limits and stitch_params
      see the grown shape (split_thin / split_satin work on the mask as given).
    * Small parts (< ``MIN_PART_MM2``) are dropped and small holes
      (< ``MIN_HOLE_MM2``) are filled over; palette.py's clean-up absorbs
      specks first, so dropping them leaves no gap.
    * Skeletons stop about half a stroke width short of a stroke's end, so
      ``_column`` extends each column past its skeleton ends.
    * Angles follow Ink/Stitch: degrees counter-clockwise from horizontal as
      seen on screen, so the y axis is flipped when computing them.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .raster import PX_PER_MM

MIN_PART_MM2 = 1.0        # polygons smaller than this (mm²) are dropped
MIN_HOLE_MM2 = 0.8        # holes smaller than this (mm²) are filled over
SIMPLIFY_MM = 0.04        # Douglas-Peucker tolerance for traced outlines
THIN_RATIO_MM = 1.2       # area / perimeter below this -> contour fill
RING_HOLE_SHARE = 0.5     # one hole covering more than this share of the outline -> ring
AXIS_TOL_DEG = 3.0        # an edge within this of horizontal/vertical counts as axis-aligned
AXIS_SHARE = 0.6          # share of outline length that must be axis-aligned for mostly_axis_aligned


def polygons(geom: BaseGeometry) -> list[Polygon]:
    """Flatten any shapely geometry into its list of Polygons.

    Polygons and MultiPolygons are unpacked; collections are walked
    recursively; lines, points and empty geometries contribute nothing.
    """
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in polygons(g)]
    return []


def _ring_mm(cnt: np.ndarray) -> list[tuple[float, float]]:
    """OpenCV contour (Nx1x2 pixel x, y) -> list of mm points at the pixel centers."""
    pts = cnt.reshape(-1, 2).astype(np.float64)
    return [((x + 0.5) / PX_PER_MM, (y + 0.5) / PX_PER_MM) for x, y in pts]


def mask_to_polygons(mask: np.ndarray) -> list[Polygon]:
    """Trace a boolean mask (keeping holes) into simplified, valid polygons in mm.

    The mask is dilated by 1 px, traced with ``cv2.findContours(RETR_CCOMP)``
    (outer boundaries plus their holes), made valid with ``buffer(0)`` and
    simplified at ``SIMPLIFY_MM``. Holes under ``MIN_HOLE_MM2`` and parts
    under ``MIN_PART_MM2`` are dropped.

    Args:
        mask: HxW bool mask of one color at ``PX_PER_MM``.

    Returns:
        Polygons in mm (possibly empty).
    """
    # Grow 1 px so neighboring colors overlap slightly instead of leaving hairline seams.
    m = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    out: list[Polygon] = []
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1 or len(cnt) < 3:
            continue
        holes = []
        child = hier[i][2]
        while child != -1:
            if len(contours[child]) >= 3:
                ring = _ring_mm(contours[child])
                if Polygon(ring).area >= MIN_HOLE_MM2:
                    holes.append(ring)
            child = hier[child][0]
        geom = Polygon(_ring_mm(cnt), holes).buffer(0)
        geom = geom.simplify(SIMPLIFY_MM, preserve_topology=True).buffer(0)
        for p in polygons(geom):
            if p.area < MIN_PART_MM2:
                continue
            kept = [r for r in p.interiors if Polygon(r).area >= MIN_HOLE_MM2]
            out.append(Polygon(p.exterior, kept) if len(kept) != len(p.interiors) else p)
    return out


def _edges(poly: Polygon):
    """Yield every segment ``((x0, y0), (x1, y1))`` of the exterior and all holes."""
    for ring in [poly.exterior, *poly.interiors]:
        c = list(ring.coords)
        yield from zip(c, c[1:])


def mostly_axis_aligned(poly: Polygon) -> bool:
    """True when most of the outline (by length) runs horizontally or vertically.

    The outline is first simplified at 0.2 mm so pixel jaggies don't count.
    An edge is aligned when it is within ``AXIS_TOL_DEG`` of 0 or 90 degrees;
    the result is True when aligned edges make up at least ``AXIS_SHARE`` of
    the total length. stitch_params uses it to give boxy shapes 45 degree rows.
    """
    p = poly.simplify(0.2, preserve_topology=True)
    total = aligned = 0.0
    for (x0, y0), (x1, y1) in _edges(p):
        ln = math.hypot(x1 - x0, y1 - y0)
        if ln == 0:
            continue
        a = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 90
        total += ln
        if a < AXIS_TOL_DEG or a > 90 - AXIS_TOL_DEG:
            aligned += ln
    return total > 0 and aligned / total >= AXIS_SHARE


def long_axis_angle(poly: Polygon) -> int:
    """Angle of the minimum rotated rectangle's long side, in Ink/Stitch's convention
    (degrees, counter-clockwise from horizontal as seen on screen), rounded to 5°.

    Returns:
        An int in 0..175. 0 when the rectangle is degenerate (not a Polygon).
    """
    rect = poly.minimum_rotated_rectangle
    if not isinstance(rect, Polygon):
        return 0
    c = list(rect.exterior.coords)
    e1 = (c[1][0] - c[0][0], c[1][1] - c[0][1])
    e2 = (c[2][0] - c[1][0], c[2][1] - c[1][1])
    dx, dy = e1 if math.hypot(*e1) >= math.hypot(*e2) else e2
    deg = math.degrees(math.atan2(-dy, dx)) % 180   # SVG y points down
    return int(5 * round(deg / 5)) % 180


def is_thin_or_ring(poly: Polygon) -> bool:
    """True when a shape should get contour fill instead of tatami.

    Thin: area / perimeter below ``THIN_RATIO_MM`` (perimeter includes holes).
    Ring: exactly one hole, covering more than ``RING_HOLE_SHARE`` of the
    area inside the outer edge.
    """
    if poly.length and poly.area / poly.length < THIN_RATIO_MM:
        return True
    if len(poly.interiors) == 1:
        outer = Polygon(poly.exterior).area
        return outer > 0 and Polygon(poly.interiors[0]).area / outer > RING_HOLE_SHARE
    return False


def stitch_params(poly: Polygon, light_on_dark: bool = False, pull_comp_mm: float = 0.0,
                  row_spacing_mm: float | None = None, underlay: bool = False) -> dict[str, str]:
    """row_spacing_mm / underlay come from the fabric type (None = Ink/Stitch default).

    Thin shapes and rings (``is_thin_or_ring``) get ``contour_fill`` with only
    a row spacing. Everything else gets ``tatami_fill`` with an ``angle``
    from ``long_axis_angle`` (45 instead of 0/90 for mostly axis-aligned
    shapes), pull compensation, underlay and row spacing. Light thread on
    dark fabric forces underlay (tatami only) and caps rows at 0.25 mm.

    Args:
        poly: The fill shape in mm.
        light_on_dark: Light thread on dark fabric (decided in pipeline.py).
        pull_comp_mm: Pull compensation; written only when > 0, tatami only.
        row_spacing_mm: Fill density from the fabric type, or None.
        underlay: Fabric type's ``fill_underlay`` setting (tatami only).

    Returns:
        Ink/Stitch params as strings, e.g. ``{"fill_method": "tatami_fill",
        "angle": "45", "row_spacing_mm": "0.25"}``. Keys left out mean
        Ink/Stitch's default.
    """
    if is_thin_or_ring(poly):   # contour fill takes the fabric's density, but has no underlay
        params = {"fill_method": "contour_fill"}
        rs = min(row_spacing_mm or 0.25, 0.25) if light_on_dark else row_spacing_mm
        if rs:
            params["row_spacing_mm"] = f"{rs:g}"
        return params
    angle = long_axis_angle(poly)
    if angle in (0, 90) and mostly_axis_aligned(poly):
        angle = 45
    params = {"fill_method": "tatami_fill", "angle": str(angle)}
    if pull_comp_mm > 0:   # rows are stretched this far past each edge (Ink/Stitch: tatami only)
        params["pull_compensation_mm"] = f"{pull_comp_mm:g}"
    if light_on_dark:      # never lighter than 0.25 mm rows, so the dark fabric doesn't show through
        underlay = True
        row_spacing_mm = min(row_spacing_mm or 0.25, 0.25)
    if underlay:
        params["fill_underlay"] = "true"
    if row_spacing_mm:
        params["row_spacing_mm"] = f"{row_spacing_mm:g}"
    return params


def band(polys: list[Polygon], width_mm: float, outer_only: bool = False) -> list[Polygon]:
    """A band of the given width just inside each polygon's edge (for outlines/borders).

    Args:
        polys: Shapes in mm.
        width_mm: Band width, measured inward (mitred corners).
        outer_only: Ignore holes, so only the outer edge gets a band (design border).

    Returns:
        Band polygons; pieces under 0.2 mm² are dropped.
    """
    out = []
    for p in polys:
        base = Polygon(p.exterior) if outer_only else p
        ring = base.difference(base.buffer(-width_mm, join_style="mitre"))
        out += [q for q in polygons(ring.buffer(0)) if q.area >= 0.2]
    return out


# ---------------------------------------------------------------- outline mode
# Features narrower than THIN_LINE_MM can't hold a fill. Their centerline is
# stitched instead as a triple (bean) running stitch in the same color.

THIN_LINE_MM = 1.0
MIN_LINE_MM = 2.0          # shorter thin bits are corner slivers — they stay in the fill
LINE_SIMPLIFY_MM = 0.08
LINE_PARAMS = {"stroke_method": "running_stitch", "running_stitch_length_mm": "2",
               "bean_stitch_repeats": "1"}


def split_thin(mask: np.ndarray) -> tuple[np.ndarray, list]:
    """Split a color mask into (fill mask, centerlines of its thin line features).

    A morphological open with a ``THIN_LINE_MM`` disk keeps what can hold a
    fill; the rest is "thin". A thin connected piece counts as a line when the
    mask's skeleton runs at least ``MIN_LINE_MM`` through it; shorter ones
    (corner slivers) stay in the fill. The skeleton over the line pieces (plus
    2 px past them, so lines reach the fills they touch) is traced, joined
    into long runs and simplified at ``LINE_SIMPLIFY_MM``; runs under 0.6 mm
    are dropped.

    Args:
        mask: HxW bool mask of one color at ``PX_PER_MM``.

    Returns:
        ``(fill_mask, lines)``: the mask without the line pieces, and a list
        of shapely LineStrings in mm. When there are no lines the input mask
        is returned as is.
    """
    from shapely.geometry import LineString
    from skimage.morphology import skeletonize

    m = mask.astype(np.uint8)
    d = max(3, round(THIN_LINE_MM * PX_PER_MM) | 1)
    core = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    thin = mask & ~core.astype(bool)
    if not thin.any():
        return mask, []
    n, cc = cv2.connectedComponents(thin.astype(np.uint8), connectivity=8)
    skel = skeletonize(mask)
    length_px = np.bincount(cc[skel & thin], minlength=n)
    lines = np.flatnonzero(length_px >= MIN_LINE_MM * PX_PER_MM)
    lines = lines[lines != 0]
    if not len(lines):
        return mask, []
    line_mask = np.isin(cc, lines)
    # a couple of px of skeleton past the thin part, so lines meet the fills they touch
    reach = cv2.dilate(line_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    out = []
    for chain in _join_chains([c for c in _trace(skel & reach) if len(c) >= 2]):
        ls = LineString([((x + 0.5) / PX_PER_MM, (y + 0.5) / PX_PER_MM) for y, x in chain])
        ls = ls.simplify(LINE_SIMPLIFY_MM)
        if ls.length >= 0.6:
            out.append(ls)
    return mask & ~line_mask, out


def split_strokes(mask: np.ndarray) -> tuple[np.ndarray, list]:
    """Split off whole pieces that are strokes too thin to fill, judged by average width.

    ``split_thin`` judges pixel by pixel, so a thin stroke with a blob on it (a
    0.6 mm crossbar whose serif corners are 1 mm across) breaks into bits too short
    to be lines and stays a fill, which scribbles. Here a connected piece whose area
    over skeleton length is under ``SATIN_MIN_MM`` and whose skeleton is at least
    ``MIN_LINE_MM`` long becomes centerlines. Satin mode runs it on what
    ``split_satin`` leaves, and the lines become ``line_satin`` columns.

    Returns:
        ``(fill_mask, lines)`` like ``split_thin``.
    """
    from shapely.geometry import LineString
    from skimage.morphology import skeletonize

    n, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        return mask, []
    skel = skeletonize(mask)
    area = np.bincount(cc.ravel(), minlength=n)
    length = np.bincount(cc[skel], minlength=n)
    ids = [i for i in range(1, n) if length[i] >= MIN_LINE_MM * PX_PER_MM
           and area[i] < SATIN_MIN_MM * PX_PER_MM * length[i]]
    if not ids:
        return mask, []
    strokes = np.isin(cc, ids)
    out = []
    for chain in _join_chains([c for c in _trace(skel & strokes) if len(c) >= 2]):
        ls = LineString([((x + 0.5) / PX_PER_MM, (y + 0.5) / PX_PER_MM) for y, x in chain]).simplify(LINE_SIMPLIFY_MM)
        if ls.length >= 0.6:
            out.append(ls)
    return mask & ~strokes, out


def _join_chains(chains: list[list]) -> list[list]:
    """Link chains that meet at a junction into long continuous runs, so a line is
    stitched in one go instead of jumping between its pieces.

    Greedy: starting from the longest unused chain, keep appending any unused
    chain that shares the run's current end (flipped if needed), then do the
    same from the other end. Unlike ``_join_smooth`` the turn angle is ignored.

    Args:
        chains: Chains of (y, x) pixels from ``_trace``.

    Returns:
        Runs of (y, x) pixels; every chain is used exactly once.
    """
    from collections import defaultdict

    ends = defaultdict(set)
    for i, c in enumerate(chains):
        ends[c[0]].add(i)
        ends[c[-1]].add(i)
    used = [False] * len(chains)
    runs = []
    for i in sorted(range(len(chains)), key=lambda i: -len(chains[i])):   # longest first
        if used[i]:
            continue
        used[i] = True
        run = list(chains[i])
        for _ in range(2):                      # extend forward, then from the other end
            while True:
                nxt = next((j for j in ends[run[-1]] if not used[j]), None)
                if nxt is None:
                    break
                used[nxt] = True
                c = chains[nxt] if chains[nxt][0] == run[-1] else chains[nxt][::-1]
                run.extend(c[1:])
            run.reverse()
        runs.append(run)
    return runs


def _trace(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Skeleton pixels -> chains between endpoints/junctions (closed loops included).

    Pixels are 8-connected, but a diagonal step only counts when no
    orthogonal pair of steps links the same two pixels, so a staircase isn't
    read as a junction. Nodes are pixels with other than two neighbors (ends
    and junctions). Every chain runs node to node; skeleton loops with no node
    become one closed chain that starts and ends on the same pixel. Each
    pixel-to-pixel step appears in exactly one chain.

    Args:
        skel: HxW bool skeleton.

    Returns:
        Chains as lists of (y, x) tuples, each at least two pixels long.
        Isolated single pixels produce no chain.
    """
    pts = set(zip(*np.nonzero(skel)))

    def nbrs(p):
        """Neighbors of pixel ``p`` in the skeleton (see the diagonal rule above)."""
        y, x = p
        out = [q for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)) if q in pts]
        for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            q = (y + dy, x + dx)
            # diagonal only when no orthogonal step already connects the two pixels
            if q in pts and (y + dy, x) not in pts and (y, x + dx) not in pts:
                out.append(q)
        return out

    nb = {p: nbrs(p) for p in pts}
    nodes = {p for p in pts if len(nb[p]) != 2}
    seen: set = set()

    def walk(a, b):
        """Follow the skeleton from ``a`` through ``b`` until a node or back at ``a``,
        marking each step as seen; returns the chain of pixels."""
        chain = [a, b]
        seen.add((a, b)); seen.add((b, a))
        prev, cur = a, b
        while cur not in nodes and cur != a:
            nxt = [q for q in nb[cur] if q != prev and (cur, q) not in seen]
            if not nxt:
                break
            q = nxt[0]
            seen.add((cur, q)); seen.add((q, cur))
            chain.append(q)
            prev, cur = cur, q
        return chain

    chains = []
    for start in list(nodes) + list(pts):      # junction/end chains first, then pure loops
        for q in nb[start]:
            if (start, q) not in seen:
                chains.append(walk(start, q))
    return chains


# ---------------------------------------------------------------- satin mode
# Strokes of a steady width (script letters, swashes, bars) are stitched as satin
# columns: two rails along the stroke's edges, one zig-zag across. Each column is
# built from a stretch of the color's skeleton; what the satins don't cover (wide
# areas, junction blobs) stays a fill. Rails get equal point counts, so Ink/Stitch
# pairs them node by node (no rungs needed).

SATIN_MIN_MM = 1.0
SATIN_MAX_MM = 7.0         # wider zig-zags snag and get loose on home machines
SATIN_MIN_LEN = 1.5        # a column must be at least this many widths long (else it's a blob)
SATIN_STEP_MM = 0.5        # rail node spacing
SATIN_MAX_TURN_DEG = 50    # skeleton pieces meeting straighter than this are one column
SATIN_MAX_SPILL = 0.08     # share of a column allowed outside the shape
SATIN_SPACING_MM = 0.4     # Ink/Stitch's default zig-zag spacing
LINE_SATIN_MM = 1.0        # in satin mode, hairlines are stitched as satin this wide
SATIN_NARROW_MM = 0.7      # strokes this wide up to SATIN_MIN_MM are widened to it
SATIN_BRIDGE_MM = 3.0      # a column carries on through a thinner dip up to this long, at SATIN_MIN_MM
SATIN_SLIVER_MM = 0.5      # leftovers narrower than this along a column are its edge, not a fill


class Satin:
    """A satin column in mm: rails as equal-length point lists, its outline, median width.

    Attributes:
        rail1: List of (x, y) mm points along one edge of the stroke.
        rail2: The other edge, same length and direction, node ``i`` paired
            with ``rail1[i]``. svg_out.py writes both as subpaths of one path.
        width_mm: Median stroke width (used by ``satin_params`` for underlay).
        dot: Made by ``split_dots``; sewn denser (``DOT_SPACING_MM``).
        poly: The column's outline (rail1 then rail2 reversed), the largest
            piece after ``buffer(0)``. Used for previews, coverage, outline
            bands and the design silhouette.
    """

    def __init__(self, rail1: list, rail2: list, width_mm: float, dot: bool = False):
        """Store the rails and build ``poly``.

        Raises:
            ValueError: If the outline is degenerate (no polygon area).
        """
        self.rail1, self.rail2, self.width_mm, self.dot = rail1, rail2, width_mm, dot
        self.poly = max(polygons(Polygon(rail1 + rail2[::-1]).buffer(0)), key=lambda p: p.area)

    def translated(self, dx: float, dy: float) -> "Satin":
        """A copy moved by (dx, dy) mm (pipeline.py uses it to center the design in the hoop)."""
        return Satin([(x + dx, y + dy) for x, y in self.rail1], [(x + dx, y + dy) for x, y in self.rail2],
                     self.width_mm, self.dot)


def satin_params(width_mm: float, pull_comp_mm: float = 0.0, row_spacing_mm: float | None = None,
                 underlay: bool = False, light_on_dark: bool = False) -> dict[str, str]:
    """Zig-zag spacing follows the fabric's fill density (0.25 mm rows = the 0.4 mm default).

    Center-walk underlay is always on; contour underlay from 2 mm wide;
    zig-zag underlay from 3 mm wide when the fabric (or light-on-dark) asks
    for underlay. Light thread on dark fabric caps the spacing at 0.4 mm and
    turns underlay on.

    Args:
        width_mm: The column's ``Satin.width_mm``.
        pull_comp_mm: Pull compensation; written only when > 0.
        row_spacing_mm: The fabric's fill row spacing, or None for standard.
        underlay: The fabric type's underlay setting.
        light_on_dark: Light thread on dark fabric.

    Returns:
        Ink/Stitch params as strings, starting with ``satin_column="true"``.
    """
    spacing = SATIN_SPACING_MM * (row_spacing_mm or 0.25) / 0.25
    if light_on_dark:
        spacing, underlay = min(spacing, SATIN_SPACING_MM), True
    params = {"satin_column": "true", "zigzag_spacing_mm": f"{spacing:.2f}".rstrip("0").rstrip("."),
              "center_walk_underlay": "true", "contour_underlay": "true" if width_mm >= 2 else "false",
              "zigzag_underlay": "true" if underlay and width_mm >= 3 else "false"}
    if pull_comp_mm > 0:
        params["pull_compensation_mm"] = f"{pull_comp_mm:g}"
    return params


def split_satin(mask: np.ndarray) -> tuple[np.ndarray, list[Satin]]:
    """Split a color mask into (what's left to fill, satin columns).

    Steps:
      1. Skeletonize the mask, trace it into chains and drop spurs (short
         twigs from a free tip, shorter than 1.5x their largest half-width).
      2. Join chains into smooth runs (``_join_smooth``). Stroke width along a
         run is twice the distance transform, median-smoothed over 9 pixels.
      3. Every stretch whose width stays within ``SATIN_MIN_MM``..``SATIN_MAX_MM``
         and is at least ``SATIN_MIN_LEN`` widths long becomes a ``_column``.
         A column is rejected if more than ``SATIN_MAX_SPILL`` of it falls
         outside the mask (grown by 1 px).
      3b. Where columns overlap, the smaller one is cut back to the bigger
         one's edge (``_clip_crossings``).
      4. The fill remainder is what the columns don't cover, minus strips
         narrower than ``SATIN_SLIVER_MM`` (a column's edge), keeping pieces of
         at least 0.2 mm², grown 0.3 mm so they tuck under the columns (see the
         comment below).

    Args:
        mask: HxW bool mask of one color at ``PX_PER_MM``. In pipeline.py
            this is already the fill mask returned by ``split_thin``.

    Returns:
        ``(fill_mask, satins)``. With no columns the input mask is returned
        unchanged with an empty list.
    """
    from collections import Counter
    from skimage.morphology import skeletonize

    m8 = mask.astype(np.uint8)
    if not m8.any():
        return mask, []
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    chains = [c for c in _trace(skeletonize(mask)) if len(c) >= 2]
    ends = Counter(p for c in chains for p in (c[0], c[-1]))

    def spur(c):   # a short twig off a corner of a wide area, left by the skeleton
        """True for a non-loop chain with a free tip that is shorter than 1.5x its max half-width (px)."""
        tip = ends[c[0]] == 1 or ends[c[-1]] == 1
        return tip and c[0] != c[-1] and len(c) < 1.5 * float(max(dist[y, x] for y, x in c))
    chains = [c for c in chains if not spur(c)]

    lo, hi = SATIN_MIN_MM * PX_PER_MM, SATIN_MAX_MM * PX_PER_MM
    # 3 px of slack: a stroke widened to SATIN_MIN_MM spills up to 0.15 mm per side
    allowed = cv2.dilate(m8, np.ones((7, 7), np.uint8)).astype(bool)
    covered = np.zeros_like(m8)
    out = []
    for run in _join_smooth(chains):
        w = 2 * np.array([dist[y, x] for y, x in run])   # distance to the edge is a half-width
        w = np.array([np.median(w[max(0, i - 4):i + 5]) for i in range(len(w))])
        # a stroke just under the minimum (a 0.9 mm serif bar) is sewn at the minimum,
        # as hairlines are, rather than left to a fill too thin to hold
        narrow = (w >= SATIN_NARROW_MM * PX_PER_MM) & (w < lo)
        w = np.where(narrow, lo, w)
        ok = (w >= lo) & (w <= hi)
        # a short thin dip between two good stretches (an arch thinning at its top) is
        # bridged at the minimum width; breaking there would leave a fill in the dip
        i = 0
        while i < len(run):
            if ok[i] or w[i] > hi:
                i += 1
                continue
            j = i
            while j < len(run) and not ok[j] and w[j] <= hi:
                j += 1
            if 0 < i and j < len(run) and ok[i - 1] and ok[j] and j - i <= SATIN_BRIDGE_MM * PX_PER_MM:
                ok[i:j], w[i:j] = True, np.maximum(w[i:j], lo)
            i = j
        i = 0
        while i < len(run):                    # each in-range stretch is a candidate column
            if not ok[i]:
                i += 1
                continue
            j = i
            while j < len(run) and ok[j]:
                j += 1
            seg, ws, start = run[i:j], w[i:j], i
            i = j
            if len(seg) < SATIN_MIN_LEN * float(np.median(ws)):
                continue
            # an end is a free tip only if this stretch reaches the run's end and nothing else meets there
            tips = (start == 0 and ends[run[0]] == 1, j == len(run) and ends[run[-1]] == 1)
            sat = _column(seg, ws, tips, mask)
            if sat is None:
                continue
            pm = np.zeros_like(m8)
            _fill_mm(pm, sat.poly)
            area = int(pm.sum())
            if not area or (pm.astype(bool) & ~allowed).sum() > SATIN_MAX_SPILL * area:
                continue
            out.append(sat)
    if not out:
        return mask, []
    out = _clip_crossings(out)
    for sat in out:
        _fill_mm(covered, sat.poly)
    # What's left is filled, judged by width rather than by closeness to a column:
    # - Strips between a column's rails and the shape's edge are hairline-thin (under
    #   SATIN_SLIVER_MM). Stitching them would sew a second layer over the satin's edge,
    #   so an opening drops them.
    # - Anything wider is a real part of the shape (serifs, arches, junction centers, the
    #   spot where a stroke narrows into a hairline). It stays, even right next to a
    #   column, and grows 0.3 mm so it tucks under its neighbors instead of leaving a gap.
    disk = lambda mm: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, round(mm * PX_PER_MM) | 1),) * 2)
    left = (mask & ~covered.astype(bool)).astype(np.uint8)
    solid = cv2.morphologyEx(left, cv2.MORPH_OPEN, disk(SATIN_SLIVER_MM))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 0.2 * PX_PER_MM ** 2) + 1
    solid = np.isin(cc, keep).astype(np.uint8)
    grown = cv2.dilate(solid, disk(2 * 0.3)).astype(bool)
    return mask & grown, out


DOT_SPACING_MM = 0.25      # dots get only a few zig-zags; tighter spacing keeps them solid
DOT_MAX_MM = 3.0           # an isolated blob up to this long is sewn as one satin (an i's dot)


def split_dots(mask: np.ndarray, satins: list[Satin]) -> tuple[np.ndarray, list[Satin]]:
    """Turn small isolated blobs (the dot of an i) into one satin each.

    A contour fill on a 1 to 3 mm dot spirals into a web of short stitches; a single
    satin across the dot is how dots are usually sewn. A blob qualifies when its
    longest side is at most ``DOT_MAX_MM``, it is at least ``SATIN_MIN_MM`` across,
    and it touches no satin column (junction leftovers stay fills). The column runs
    along the blob's long axis with rungs across it, every half ``SATIN_STEP_MM``.

    Returns:
        ``(fill_mask, dot_satins)``.
    """
    from shapely.geometry import LineString

    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out, drop = [], []
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        if max(w, h) > DOT_MAX_MM * PX_PER_MM or min(w, h) < SATIN_MIN_MM * PX_PER_MM:
            continue
        cnts, _ = cv2.findContours((cc[y:y + h, x:x + w] == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea) + np.array([x, y])
        if len(cnt) < 5:
            continue
        poly = Polygon(_ring_mm(cnt)).buffer(0)
        if poly.is_empty or any(poly.intersects(sat.poly) for sat in satins):
            continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(cnt)
        a = math.radians(ang if rw >= rh else ang + 90)
        u = np.array([math.cos(a), math.sin(a)])               # long axis
        v = np.array([-u[1], u[0]])
        c = (np.array([cx, cy]) + 0.5) / PX_PER_MM
        half = max(rw, rh) / PX_PER_MM / 2
        r1, r2 = [], []
        for t in np.arange(-half + SATIN_STEP_MM / 4, half, SATIN_STEP_MM / 2):
            p = c + u * t
            cut = poly.intersection(LineString([tuple(p - v * 5), tuple(p + v * 5)]))
            if cut.is_empty or cut.geom_type != "LineString":
                continue
            (ax, ay), (bx, by) = cut.coords[0], cut.coords[-1]
            if np.dot(np.array([bx - ax, by - ay]), v) < 0:
                (ax, ay), (bx, by) = (bx, by), (ax, ay)
            r1.append((float(ax), float(ay)))
            r2.append((float(bx), float(by)))
        if len(r1) < 3:
            continue
        try:
            out.append(Satin(r1, r2, min(rw, rh) / PX_PER_MM, dot=True))
        except ValueError:
            continue
        drop.append(i)
    if not out:
        return mask, []
    return mask & ~np.isin(cc, drop), out


def _clip_crossings(sats: list[Satin]) -> list[Satin]:
    """Where columns overlap, the bigger one runs through and the smaller one stops at its edge.

    Without this both columns sew across the overlap (a crossing, or a stroke running
    into another at a junction), stacking thread in two directions. Columns are taken
    biggest first, except that a column ending inside another (a stem under a serif
    bar) comes after the ones it runs into; a later column loses the rungs whose middle falls inside an earlier
    one, keeping one rung past each cut so it tucks under the edge (at most
    ``SATIN_STEP_MM``) instead of leaving a seam. A column crossed in its middle splits
    into two. Pieces under 3 rungs or 1 mm long are dropped.
    """
    from shapely.geometry import Point
    from shapely.prepared import prep

    kept: list[Satin] = []
    def ends_inside(s):
        """How many of the column's two end rungs sit inside another column."""
        ends = [Point((s.rail1[i][0] + s.rail2[i][0]) / 2, (s.rail1[i][1] + s.rail2[i][1]) / 2) for i in (0, -1)]
        return sum(any(o is not s and o.poly.contains(e) for o in sats) for e in ends)

    for sat in sorted(sats, key=lambda s: (ends_inside(s), -s.poly.area)):
        bigger = [prep(k.poly) for k in kept if k.poly.intersects(sat.poly)]
        if not bigger:
            kept.append(sat)
            continue
        mids = [Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(sat.rail1, sat.rail2)]
        inside = [any(k.contains(m) for k in bigger) for m in mids]
        if not any(inside):
            kept.append(sat)
            continue
        n, i = len(mids), 0
        while i < n:                           # each stretch outside the bigger columns, plus one rung each side
            if inside[i]:
                i += 1
                continue
            j = i
            while j < n and not inside[j]:
                j += 1
            a, b = max(0, i - 1), min(n, j + 1)
            i = j
            r1, r2 = sat.rail1[a:b], sat.rail2[a:b]
            if len(r1) < 3 or math.dist(r1[0], r1[-1]) < 1.0:
                continue
            try:
                kept.append(Satin(r1, r2, sat.width_mm))
            except ValueError:
                continue
    return kept


def line_satin(line, width_mm: float = LINE_SATIN_MM) -> Satin | None:
    """A thin feature's centerline -> a satin column of a steady width, so hairlines
    match satin lettering instead of reading as a loose running-stitch thread.

    The line is resampled every ``SATIN_STEP_MM`` and offset by half the width
    on each side along the local normal. Unlike ``_column`` the ends are not
    extended or tapered.

    Args:
        line: A shapely LineString in mm (from ``split_thin``).
        width_mm: Column width.

    Returns:
        A ``Satin``, or None when the line is shorter than 1 mm, too few node
        pairs survive the fold-back filter, or the outline is degenerate. The
        caller keeps rejected lines as running stitch.
    """
    if line.length < SATIN_STEP_MM * 2:
        return None
    n = max(3, int(round(line.length / SATIN_STEP_MM)) + 1)
    c = np.array([line.interpolate(t).coords[0] for t in np.linspace(0, line.length, n)])
    tan = np.gradient(c, axis=0)
    tan /= np.maximum(np.hypot(*tan.T), 1e-9)[:, None]
    # run 0.4 mm past both ends so the hairline overlaps the stroke it grows out of
    # instead of meeting it edge to edge (which leaves a visible seam)
    c = np.vstack([c[0] - tan[0] * 0.4, c, c[-1] + tan[-1] * 0.4])
    tan = np.vstack([tan[0], tan, tan[-1]])
    n = len(c)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1) * width_mm / 2
    left, right = c + nrm, c - nrm
    keep = [0]
    for i in range(1, n):                     # drop node pairs that fold back on a tight curve
        a = keep[-1]
        if np.dot(left[i] - left[a], tan[i]) > 0 and np.dot(right[i] - right[a], tan[i]) > 0:
            keep.append(i)
    if len(keep) < 3:
        return None
    try:
        return Satin([tuple(map(float, p)) for p in left[keep]], [tuple(map(float, p)) for p in right[keep]], width_mm)
    except ValueError:
        return None


def _fill_mm(mask: np.ndarray, poly: Polygon) -> None:
    """Rasterize a mm polygon (holes left empty) into ``mask`` at ``PX_PER_MM``, in place, as 1s.

    The -0.5 undoes the pixel-center offset used when tracing; coordinates
    are passed with 3 fractional bits (``shift=3``) for sub-pixel accuracy.
    """
    rings = [poly.exterior, *poly.interiors]
    pts = [np.round((np.asarray(r.coords) * PX_PER_MM - 0.5) * 8).astype(np.int32) for r in rings]
    cv2.fillPoly(mask, pts, 1, lineType=cv2.LINE_8, shift=3)


def _column(seg: list, widths: np.ndarray, tips: tuple[bool, bool] = (False, False),
            mask: np.ndarray | None = None) -> Satin | None:
    """Skeleton pixels + stroke widths (px) -> a satin column with paired rail nodes.
    tips: which ends are a stroke's free end (tapered) rather than a junction (run in full).

    The pixel centerline is smoothed (7-point moving average), resampled every
    ``SATIN_STEP_MM`` and offset by the local half-width (+ 0.5 px of overlap)
    to make the rails. The rung direction is averaged over about one stroke
    width so a crossing stroke doesn't skew it. Each end is extended past the skeleton: by 0.8 of the
    half-width and narrowed to 0.6 of it at a free tip, by a full half-width
    at a junction. Node pairs that would fold back on a tight curve are dropped.

    Args:
        seg: Skeleton (y, x) pixels in order.
        widths: Stroke width in px at each pixel of ``seg``.
        tips: (start is a free tip, end is a free tip).

    Returns:
        A ``Satin`` whose ``width_mm`` is the median width, or None when the
        centerline is under 1 mm, fewer than 3 node pairs survive, or the
        outline is degenerate.
    """
    xy = np.array([((x + 0.5) / PX_PER_MM, (y + 0.5) / PX_PER_MM) for y, x in seg])
    k = 7                                      # smooth the pixel staircase before taking tangents
    if len(xy) > k:
        pad = np.pad(xy, ((k // 2, k // 2), (0, 0)), mode="edge")
        xy = np.stack([np.convolve(pad[:, a], np.ones(k) / k, mode="valid") for a in (0, 1)], axis=1)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    if s[-1] < 1.0:
        return None
    n = max(3, int(round(s[-1] / SATIN_STEP_MM)) + 1)
    t = np.linspace(0, s[-1], n)
    c = np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], axis=1)
    r = np.interp(t, s, widths) / PX_PER_MM / 2 + 0.5 / PX_PER_MM   # a hair of overlap with the edge
    tan = np.gradient(c, axis=0)
    tan /= np.maximum(np.hypot(*tan.T), 1e-9)[:, None]
    # average the stitch angle over about one stroke width: where another stroke crosses,
    # the skeleton jogs sideways and would skew the rungs into an X across the crossing
    m = max(1, int(round(np.median(widths) / PX_PER_MM / SATIN_STEP_MM))) | 1
    if m > 1 and len(tan) > m:
        pad = np.pad(tan, ((m // 2, m // 2), (0, 0)), mode="edge")
        tan = np.stack([np.convolve(pad[:, a], np.ones(m) / m, mode="valid") for a in (0, 1)], axis=1)
        tan /= np.maximum(np.hypot(*tan.T), 1e-9)[:, None]
    # reach past the skeleton's ends, which stop about a half-width short of the stroke's
    # end (taper there) or of the junction's middle (run into it at full width)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    rl, rr = (r, r) if mask is None else _edges_along(mask, c, nrm, r)
    short = np.maximum(0, SATIN_MIN_MM / 2 - (rl + rr) / 2)   # never narrower than the minimum
    rl, rr = rl + short, rr + short
    reach = [0.8 if t else 1.0 for t in tips]
    c = np.vstack([c[0] - tan[0] * r[0] * reach[0], c, c[-1] + tan[-1] * r[-1] * reach[1]])
    tan = np.vstack([tan[0], tan, tan[-1]])
    nrm = np.vstack([nrm[0], nrm, nrm[-1]])
    end = lambda v: np.concatenate([[v[0] * (0.6 if tips[0] else 1.0)], v, [v[-1] * (0.6 if tips[1] else 1.0)]])
    rl, rr = end(rl), end(rr)
    left, right = c + nrm * rl[:, None], c - nrm * rr[:, None]
    keep = [0]
    for i in range(1, len(c)):                # drop node pairs that fold back on a tight inner curve
        a = keep[-1]
        if np.dot(left[i] - left[a], tan[i]) > 0 and np.dot(right[i] - right[a], tan[i]) > 0:
            keep.append(i)
    if len(keep) < 3:
        return None
    rail1 = [tuple(map(float, p)) for p in left[keep]]
    rail2 = [tuple(map(float, p)) for p in right[keep]]
    try:
        return Satin(rail1, rail2, float(np.median(widths)) / PX_PER_MM)
    except ValueError:                         # degenerate outline
        return None


def _edges_along(mask: np.ndarray, c: np.ndarray, nrm: np.ndarray, r: np.ndarray):
    """Distance (mm) from each centerline node to the stroke's edge on each side, along the rung.

    Rails offset symmetrically from the skeleton zig-zag unevenly wherever the skeleton
    sits off center or the width estimate is noisy; measuring each side puts every rung
    end on the real edge. A side that runs on (into a junction or another stroke) or
    reads under half its expected reach keeps the symmetric estimate ``r``. The result
    is median-smoothed over 5 nodes, plus 0.5 px of overlap like ``r``.
    """
    step = 0.5 / PX_PER_MM
    reach = np.arange(0, SATIN_MAX_MM, step)
    h, w = mask.shape
    out = []
    for sign in (1, -1):
        pts = c[:, None, :] + sign * nrm[:, None, :] * reach[None, :, None]   # node x sample x (x, y)
        px = np.clip(np.round(pts * PX_PER_MM - 0.5).astype(int), 0, [w - 1, h - 1])
        inside = mask[px[..., 1], px[..., 0]]
        first_out = np.where(inside.all(axis=1), len(reach), np.argmin(inside, axis=1))
        d = reach[np.minimum(first_out, len(reach) - 1)] - step / 2 + 0.5 / PX_PER_MM
        bad = (d > 1.5 * r) | (d < 0.5 * r)
        d = np.where(bad, r, d)
        d = np.array([np.median(d[max(0, i - 2):i + 3]) for i in range(len(d))])
        out.append(d)
    return out[0], out[1]


def _join_smooth(chains: list[list]) -> list[list]:
    """Join skeleton pieces that continue each other straight through a junction (turn
    under SATIN_MAX_TURN_DEG), so a stroke is one column; sharp corners start a new one.

    Greedy: starting from the longest unused chain, at the run's end pick the
    unused chain meeting there whose direction (taken over about 10 pixels)
    turns least, if under the limit; repeat, then extend from the other end.
    A run that closes into a loop stops growing.

    Args:
        chains: Chains of (y, x) pixels from ``_trace``.

    Returns:
        Runs of (y, x) pixels; every chain is used exactly once.
    """
    from collections import defaultdict

    def heading(c, at_start):                 # direction leaving the chain's end, into the junction
        """Unit (dy, dx) vector from about 30 pixels inside the chain toward its start (or end).

        The first few pixels are skipped: the skeleton curves into a junction over
        about a half-width, which made a stem meeting a serif bar look like it carried
        on into one half of the bar.
        """
        pts = c if at_start else c[::-1]
        a, b = np.array(pts[min(len(pts) - 1, 30)]), np.array(pts[min(len(pts) - 1, 6) if len(pts) > 12 else 0])
        v = (b - a).astype(float)
        return v / max(np.hypot(*v), 1e-9)

    ends = defaultdict(list)
    for i, c in enumerate(chains):
        ends[c[0]].append(i)
        ends[c[-1]].append(i)
    used = [False] * len(chains)
    runs = []
    cos_max = math.cos(math.radians(SATIN_MAX_TURN_DEG))
    for i in sorted(range(len(chains)), key=lambda i: -len(chains[i])):
        if used[i]:
            continue
        used[i] = True
        run = list(chains[i])
        for _ in range(2):
            while run[0] != run[-1]:
                into = heading(run, False)    # arriving at run[-1]
                best, best_cos = None, cos_max
                for j in ends[run[-1]]:
                    if used[j]:
                        continue
                    c = chains[j] if chains[j][0] == run[-1] else chains[j][::-1]
                    out = -heading(c, True)   # leaving run[-1] along c
                    cs = float(np.dot(into, out))
                    if cs > best_cos:
                        best, best_cos = (j, c), cs
                if best is None:
                    break
                used[best[0]] = True
                run.extend(best[1][1:])
            run.reverse()
        runs.append(run)
    return runs
