"""Steps 4 and 7 — label mask -> clean shapely polygons, and per-shape stitch settings."""
from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .raster import PX_PER_MM

MIN_PART_MM2 = 1.0
MIN_HOLE_MM2 = 0.8
SIMPLIFY_MM = 0.04
THIN_RATIO_MM = 1.2       # area / perimeter below this -> contour fill
RING_HOLE_SHARE = 0.5     # one hole covering more than this share of the outline -> ring
AXIS_TOL_DEG = 3.0
AXIS_SHARE = 0.6


def polygons(geom: BaseGeometry) -> list[Polygon]:
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
    pts = cnt.reshape(-1, 2).astype(np.float64)
    return [((x + 0.5) / PX_PER_MM, (y + 0.5) / PX_PER_MM) for x, y in pts]


def mask_to_polygons(mask: np.ndarray) -> list[Polygon]:
    """Trace a boolean mask (keeping holes) into simplified, valid polygons in mm."""
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
    for ring in [poly.exterior, *poly.interiors]:
        c = list(ring.coords)
        yield from zip(c, c[1:])


def mostly_axis_aligned(poly: Polygon) -> bool:
    """True when most of the outline (by length) runs horizontally or vertically."""
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
    (degrees, counter-clockwise from horizontal as seen on screen), rounded to 5°."""
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
    if poly.length and poly.area / poly.length < THIN_RATIO_MM:
        return True
    if len(poly.interiors) == 1:
        outer = Polygon(poly.exterior).area
        return outer > 0 and Polygon(poly.interiors[0]).area / outer > RING_HOLE_SHARE
    return False


def stitch_params(poly: Polygon, light_on_dark: bool = False, pull_comp_mm: float = 0.0,
                  row_spacing_mm: float | None = None, underlay: bool = False) -> dict[str, str]:
    """row_spacing_mm / underlay come from the fabric type (None = Ink/Stitch default)."""
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
    """A band of the given width just inside each polygon's edge (for outlines/borders)."""
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
    """Split a color mask into (fill mask, centerlines of its thin line features)."""
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


def _join_chains(chains: list[list]) -> list[list]:
    """Link chains that meet at a junction into long continuous runs, so a line is
    stitched in one go instead of jumping between its pieces."""
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
    """Skeleton pixels -> chains between endpoints/junctions (closed loops included)."""
    pts = set(zip(*np.nonzero(skel)))

    def nbrs(p):
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


class Satin:
    """A satin column in mm: rails as equal-length point lists, its outline, median width."""

    def __init__(self, rail1: list, rail2: list, width_mm: float):
        self.rail1, self.rail2, self.width_mm = rail1, rail2, width_mm
        self.poly = max(polygons(Polygon(rail1 + rail2[::-1]).buffer(0)), key=lambda p: p.area)

    def translated(self, dx: float, dy: float) -> "Satin":
        return Satin([(x + dx, y + dy) for x, y in self.rail1], [(x + dx, y + dy) for x, y in self.rail2],
                     self.width_mm)


def satin_params(width_mm: float, pull_comp_mm: float = 0.0, row_spacing_mm: float | None = None,
                 underlay: bool = False, light_on_dark: bool = False) -> dict[str, str]:
    """Zig-zag spacing follows the fabric's fill density (0.25 mm rows = the 0.4 mm default)."""
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
    """Split a color mask into (what's left to fill, satin columns)."""
    from collections import Counter
    from skimage.morphology import skeletonize

    m8 = mask.astype(np.uint8)
    if not m8.any():
        return mask, []
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    chains = [c for c in _trace(skeletonize(mask)) if len(c) >= 2]
    ends = Counter(p for c in chains for p in (c[0], c[-1]))

    def spur(c):   # a short twig off a corner of a wide area, left by the skeleton
        tip = ends[c[0]] == 1 or ends[c[-1]] == 1
        return tip and c[0] != c[-1] and len(c) < 1.5 * float(max(dist[y, x] for y, x in c))
    chains = [c for c in chains if not spur(c)]

    lo, hi = SATIN_MIN_MM * PX_PER_MM, SATIN_MAX_MM * PX_PER_MM
    allowed = cv2.dilate(m8, np.ones((3, 3), np.uint8)).astype(bool)
    covered = np.zeros_like(m8)
    out = []
    for run in _join_smooth(chains):
        w = 2 * np.array([dist[y, x] for y, x in run])
        w = np.array([np.median(w[max(0, i - 4):i + 5]) for i in range(len(w))])
        ok = (w >= lo) & (w <= hi)
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
            tips = (start == 0 and ends[run[0]] == 1, j == len(run) and ends[run[-1]] == 1)
            sat = _column(seg, ws, tips)
            if sat is None:
                continue
            pm = np.zeros_like(m8)
            _fill_mm(pm, sat.poly)
            area = int(pm.sum())
            if not area or (pm.astype(bool) & ~allowed).sum() > SATIN_MAX_SPILL * area:
                continue
            covered |= pm
            out.append(sat)
    if not out:
        return mask, []
    # What's left is filled. It reaches 0.3 mm under the columns so junctions don't leave
    # fabric showing, then slivers left along the columns' edges are dropped.
    disk = lambda mm: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, round(mm * PX_PER_MM) | 1),) * 2)
    rest = mask & ~cv2.erode(covered, disk(0.3)).astype(bool)
    # A leftover piece lying wholly along a column, and too small or too thin to fill, is a
    # sliver of its edge: drop it. Anything reaching further (serifs, hairlines) stays whole.
    opened = cv2.morphologyEx(rest.astype(np.uint8), cv2.MORPH_OPEN, disk(0.6)).astype(bool)
    near = cv2.dilate(covered, disk(1.0)).astype(bool)
    n, cc = cv2.connectedComponents(rest.astype(np.uint8), connectivity=8)
    idx = cc.ravel()
    area = np.bincount(idx, minlength=n)
    away = np.bincount(idx, weights=(rest & ~near).ravel(), minlength=n)
    solid = np.bincount(idx, weights=opened.ravel(), minlength=n)
    drop = (away == 0) & ((area < PX_PER_MM ** 2) | (solid == 0))
    drop[0] = True
    return rest & ~drop[cc], out


def _fill_mm(mask: np.ndarray, poly: Polygon) -> None:
    rings = [poly.exterior, *poly.interiors]
    pts = [np.round((np.asarray(r.coords) * PX_PER_MM - 0.5) * 8).astype(np.int32) for r in rings]
    cv2.fillPoly(mask, pts, 1, lineType=cv2.LINE_8, shift=3)


def _column(seg: list, widths: np.ndarray, tips: tuple[bool, bool] = (False, False)) -> Satin | None:
    """Skeleton pixels + stroke widths (px) -> a satin column with paired rail nodes.
    tips: which ends are a stroke's free end (tapered) rather than a junction (run in full)."""
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
    # reach past the skeleton's ends, which stop about a half-width short of the stroke's
    # end (taper there) or of the junction's middle (run into it at full width)
    reach = [0.8 if t else 1.0 for t in tips]
    c = np.vstack([c[0] - tan[0] * r[0] * reach[0], c, c[-1] + tan[-1] * r[-1] * reach[1]])
    tan = np.vstack([tan[0], tan, tan[-1]])
    r = np.concatenate([[r[0] * (0.6 if tips[0] else 1.0)], r, [r[-1] * (0.6 if tips[1] else 1.0)]])
    nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    left, right = c + nrm * r[:, None], c - nrm * r[:, None]
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


def _join_smooth(chains: list[list]) -> list[list]:
    """Join skeleton pieces that continue each other straight through a junction (turn
    under SATIN_MAX_TURN_DEG), so a stroke is one column; sharp corners start a new one."""
    from collections import defaultdict

    def heading(c, at_start):                 # direction leaving the chain's end, into the junction
        pts = c if at_start else c[::-1]
        a, b = np.array(pts[min(len(pts) - 1, 10)]), np.array(pts[0])
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
