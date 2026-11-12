"""Stitch routing inside one thread: the order and direction of its shapes, plus hidden
travel stitches, so the machine runs from shape to shape instead of jumping.

Where it sits: ``pipeline.generate`` calls :func:`route_layers` on the final layers
(after scaling, before they're placed on the hoop). ``svg_out`` writes the travel
lines as ordinary running-stitch paths; Ink/Stitch sews everything in document order.

Why: home machines like the SE400 don't cut jump threads. Every jump is a thread to
trim by hand, and lettering has dozens of pieces (letter strokes, serifs, dots).
Pieces that touch can be linked with a running stitch laid where a later piece will
sew over it, the way a digitizer hides "travel runs" under satin.

How: greedy, from where the last shape ended. For every shape not yet sewn (and both
directions of each satin and line) the walk to its start is measured on a 0.1 mm
grid, staying ``HIDE_MM`` inside shapes that are still to be sewn (so the travel ends
up covered). The nearest reachable start wins; a shape that can't be reached costs a
jump. When the walk is longer than ``DIRECT_MM`` a travel line is added before the
shape.

Where shapes start and end: a satin starts midway between its rails' first nodes and
ends at their last (Ink/Stitch's underlays return to the start before the top stitch
runs forward). A fill starts near the last stitch and ends anywhere inside itself, so
its whole area counts as "here" for the next walk. A running line runs end to end.
"""
from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, unary_union
from shapely.strtree import STRtree
from skimage.graph import MCP_Geometric

RES = 10                # grid px per mm
HIDE_MM = 0.3           # travel stays this far inside shapes still to be sewn
SNAP_MM = 0.8           # a travel may start this far from where the last shape ended
DIRECT_MM = 0.5         # shorter walks need no travel line
VISIBLE_COST = 4.0      # a mm of travel over stitches already sewn costs this many hidden mm
TOUCH_MM = 0.05         # shapes this close count as touching (one piece, no jump between them)
TRAP_COST = 20.0        # a mm of jump thread lying under stitches still to come costs this many mm
TRAVEL_PARAMS = {"stroke_method": "running_stitch", "running_stitch_length_mm": "1.5",
                 "bean_stitch_repeats": "0", "_travel": True}


class _Grid:
    """Rasterizes mm geometry onto a grid covering the layer (with a margin)."""

    def __init__(self, bounds):
        x0, y0, x1, y1 = bounds
        self.ox, self.oy = x0 - 2, y0 - 2
        self.shape = (int((y1 - y0 + 4) * RES) + 1, int((x1 - x0 + 4) * RES) + 1)

    def px(self, x, y):
        return int(round((y - self.oy) * RES)), int(round((x - self.ox) * RES))

    def mm(self, r, c):
        return c / RES + self.ox, r / RES + self.oy

    def poly(self, p) -> np.ndarray:
        m = np.zeros(self.shape, np.uint8)
        for g in getattr(p, "geoms", [p]):
            if g.is_empty:
                continue
            ext = np.array([[(x - self.ox) * RES, (y - self.oy) * RES] for x, y in g.exterior.coords])
            cv2.fillPoly(m, [np.round(ext).astype(np.int32)], 1)
            for ring in g.interiors:
                hole = np.array([[(x - self.ox) * RES, (y - self.oy) * RES] for x, y in ring.coords])
                cv2.fillPoly(m, [np.round(hole).astype(np.int32)], 0)
        return m.astype(bool)

    def disk(self, pt, mm) -> np.ndarray:
        m = np.zeros(self.shape, np.uint8)
        r, c = self.px(*pt)
        cv2.circle(m, (c, r), max(1, int(round(mm * RES))), 1, -1)
        return m.astype(bool)


def _ends(geom, prm):
    """Ways to sew a shape: [(variant geometry, variant params, start point, end point)].
    The end point is None for a fill (it ends somewhere inside itself)."""
    if "_satin" in prm:
        s = prm["_satin"]
        r = s.reversed()
        return [(geom, prm, s.start, s.end), (geom, {**prm, "_satin": r}, r.start, r.end)]
    if geom.geom_type == "LineString":
        rev = LineString(geom.coords[::-1])
        return [(geom, prm, geom.coords[0], geom.coords[-1]), (rev, prm, rev.coords[0], rev.coords[-1])]
    return [(geom, prm, None, None)]


def route_layer(shapes: list, start) -> tuple[list, int]:
    """Order one layer's shapes, choosing directions and adding hidden travel lines.

    Args:
        shapes: ``(geometry, params)`` pairs as built by ``pipeline.generate``.
        start: Where the needle is before this layer (a shapely Point).

    Returns:
        (the routed pairs, travel lines included; the number of jumps left, counting
        the one into the layer only when it has no shapes before it).
    """
    if not shapes:
        return [], 0
    polys = [prm["_satin"].poly if "_satin" in prm else g for g, prm in shapes]
    grid = _Grid(unary_union([p.buffer(1) for p in polys]).bounds)
    hide = int(round(HIDE_MM * RES))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hide + 1,) * 2)
    cover, area, tiny = [], [], {}
    for i, ((g, prm), p) in enumerate(zip(shapes, polys)):
        if g.geom_type == "LineString" and "_satin" not in prm:
            cover.append(np.zeros(grid.shape, bool))     # a running line hides nothing
            area.append(grid.disk(g.coords[0], 0.1))
            continue
        full = grid.poly(p)
        if not cv2.erode(full.astype(np.uint8), ker).any():   # too small to erode: its middle still hides
            tiny[i] = full & grid.disk(p.representative_point().coords[0], 0.2)
        cover.append(full)
        area.append(full)
    count = np.sum(cover, axis=0, dtype=np.int32)
    inside = cv2.erode((count > 0).astype(np.uint8), ker).astype(bool)
    variants = [_ends(g, prm) for g, prm in shapes]
    targets = [[np.flatnonzero(grid.disk(s, SNAP_MM) if s is not None else area[i])
                for _, _, s, _ in variants[i]] for i in range(len(shapes))]
    snap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(SNAP_MM * RES) + 1,) * 2)

    # shapes that touch, and the pieces they make (a letter, a dot)
    tree = STRtree(polys)
    adj = [[int(j) for j in tree.query(p.buffer(TOUCH_MM)) if j != i and p.buffer(TOUCH_MM).intersects(polys[j])]
           for i, p in enumerate(polys)]
    piece, n_pieces = [-1] * len(shapes), 0
    for i in range(len(shapes)):
        if piece[i] < 0:
            stack, piece[i] = [i], n_pieces
            while stack:
                for j in adj[stack.pop()]:
                    if piece[j] < 0:
                        piece[j] = n_pieces
                        stack.append(j)
            n_pieces += 1
    members = [[i for i in range(len(shapes)) if piece[i] == k] for k in range(n_pieces)]
    outline = [unary_union([polys[i] for i in m]) for m in members]

    route, jumps = [], 0
    here_pt, here_geom, here_area = (start.x, start.y), Point(start.x, start.y), None
    todo = set(range(n_pieces))
    while todo:
        # the next piece: a short jump that doesn't lie under pieces still to come
        best = None
        for k in todo:
            land = nearest_points(here_geom, outline[k])
            seg = LineString([land[0].coords[0], land[1].coords[0]])
            rest = [outline[q] for q in todo if q != k]
            trap = seg.intersection(unary_union(rest)).length if rest else 0.0
            c = seg.length + TRAP_COST * trap
            if best is None or c < best[0]:
                best = (c, k, land[1])
        _, k, land = best
        todo.discard(k)
        if route:
            jumps += 1
            # end the last piece with a trim: without one Ink/Stitch sews a jump under
            # 3 mm as a stitch, a loose thread across the fabric the trim map won't show
            last = max(n for n, (_, q) in enumerate(route) if "_travel" not in q)
            route[last] = (route[last][0], {**route[last][1], "trim_after": "true"})
        # walk the piece's shapes as a tree from the one we land on, children first:
        # each shape is reached by travel under shapes not yet sewn and sewn back toward
        # its parent, so the next travel starts under the parent
        root = min(members[k], key=lambda i: polys[i].distance(land))
        parent, seen, bfs = {root: None}, {root}, [root]
        for i in bfs:
            for j in sorted(adj[i], key=lambda j: polys[j].distance(polys[i].centroid)):
                if j not in seen and piece[j] == k:
                    seen.add(j)
                    parent[j] = i
                    bfs.append(j)
        kids = {i: [j for j in bfs if parent[j] == i] for i in bfs}
        order, stack = [], [(root, False)]
        while stack:
            i, done = stack.pop()
            if done:
                order.append(i)
            else:
                stack.append((i, True))
                stack.extend((j, False) for j in reversed(kids[i]))
        others = unary_union([outline[q] for q in todo]) if todo else None
        here_pt, here_geom, here_area = land.coords[0], land, None
        # directions, parents first: each shape ends at its parent, nearest the parent's
        # start on a tie (the root ends toward the pieces still to come)
        pick = {}
        for i in bfs:
            vs, up = variants[i], parent[i]
            if len(vs) == 1 or (up is None and others is None):
                pick[i] = 0
                continue
            towards = others if up is None else polys[up]
            ps = None if up is None else variants[up][pick[up]][2]
            pick[i] = min(range(len(vs)), key=lambda v: (round(Point(vs[v][3]).distance(towards), 1),
                                                         Point(vs[v][3]).distance(Point(ps)) if ps else 0))
        for i in order:
            vi = pick[i]
            g, prm, s, e = variants[i][vi]
            # travel: under shapes still to be sewn, or over this piece's sewn stitches at a penalty
            hidden = cv2.erode((count > 0).astype(np.uint8), ker).astype(bool)
            for t in tiny:
                if count[tiny[t]].all():
                    hidden |= tiny[t]
            costs = np.where(hidden, 1.0, np.where(inside, VISIBLE_COST, np.inf))
            seed = (grid.disk(here_pt, SNAP_MM) if here_area is None
                    else cv2.dilate(here_area.astype(np.uint8), snap).astype(bool)) & np.isfinite(costs)
            tg = targets[i][vi]
            if seed.any() and tg.size:
                mcp = MCP_Geometric(costs)
                cum = mcp.find_costs(list(zip(*np.nonzero(seed))))[0].ravel()
                tc = cum[tg]
                if np.isfinite(tc).any():
                    j = int(np.argmin(tc))
                    if tc[j] / RES > DIRECT_MM:
                        r0, c0 = np.unravel_index(tg[j], grid.shape)
                        line = LineString([grid.mm(r, c) for r, c in mcp.traceback((r0, c0))]).simplify(0.15)
                        if line.length > DIRECT_MM:
                            route.append((line, dict(TRAVEL_PARAMS)))
            route.append((g, prm))
            count -= cover[i]
            if e is not None:
                here_pt, here_geom, here_area = e, Point(e), None
            else:
                here_pt, here_geom, here_area = polys[i].representative_point().coords[0], polys[i], area[i]
    return route, jumps


def route_layers(layers: list[dict], start) -> None:
    """Route every layer in place (see :func:`route_layer`), each starting where the
    previous one ended. Sets ``layer["jumps"]`` to the jumps left inside it."""
    cursor = start
    for layer in layers:
        layer["shapes"], layer["jumps"] = route_layer(layer["shapes"], cursor)
        if layer["shapes"]:
            g, prm = layer["shapes"][-1]
            if "_satin" in prm:
                cursor = Point(prm["_satin"].end)
            elif g.geom_type == "LineString":
                cursor = Point(g.coords[-1])
            else:
                cursor = g.representative_point()
