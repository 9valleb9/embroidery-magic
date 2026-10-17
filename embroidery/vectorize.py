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


def stitch_params(poly: Polygon, light_on_dark: bool = False) -> dict[str, str]:
    if is_thin_or_ring(poly):
        return {"fill_method": "contour_fill"}
    angle = long_axis_angle(poly)
    if angle in (0, 90) and mostly_axis_aligned(poly):
        angle = 45
    params = {"fill_method": "auto_fill", "angle": str(angle)}
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
