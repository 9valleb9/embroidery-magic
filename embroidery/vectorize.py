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


def stitch_params(poly: Polygon, light_on_dark: bool = False, pull_comp_mm: float = 0.0) -> dict[str, str]:
    if is_thin_or_ring(poly):
        return {"fill_method": "contour_fill"}
    angle = long_axis_angle(poly)
    if angle in (0, 90) and mostly_axis_aligned(poly):
        angle = 45
    params = {"fill_method": "tatami_fill", "angle": str(angle)}
    if pull_comp_mm > 0:   # rows are stretched this far past each edge (Ink/Stitch: tatami only)
        params["pull_compensation_mm"] = f"{pull_comp_mm:g}"
    if light_on_dark:
        params["fill_underlay"] = "true"
        params["row_spacing_mm"] = "0.25"
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
