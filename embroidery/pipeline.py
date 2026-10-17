"""Orchestration: analysis sessions, fabric-aware color decisions, and SVG generation.

Geometry depends only on (image, size, palette size, merges). Fabric color, hex
edits, knockouts and stitch order are "color decisions" and never re-trace shapes.
"""
from __future__ import annotations

import base64
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np
from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from . import colors as C
from .palette import TRANSPARENT, analyze, edge_connected, frame_label
from .export import EST_STITCHES_PER_MM2, HOOP_MM, SPEED_SPM, STITCH_WARN
from .raster import PX_PER_MM, rasterize
from .svg_out import write_svg
from .vectorize import LINE_PARAMS, band, mask_to_polygons, polygons, split_thin, stitch_params

OUTLINE_MM = 1.0
BORDER_MM = 1.0
NARROW_STITCH_MM = 1.0
NARROW_GAP_MM = 0.6
PREVIEW_MAX_PX = 1000
LINE_DRAW_MM = 0.7         # how wide a triple running stitch reads, for preview + gap checks


@dataclass
class Geometry:
    labels: np.ndarray                         # merged label map
    shapes: dict[int, list[Polygon]]
    lines: dict[int, list[LineString]]         # outline mode: thin features as centerlines
    area_mm2: dict[int, float]
    adjacency: dict[int, dict[int, int]]       # id -> neighbor id -> shared edge length (px)
    frame_label: int | None                    # label covering most of the image frame (>= 50%)


@dataclass
class Session:
    id: str
    filename: str
    labels: np.ndarray
    hexes: list[str]
    width_mm: float
    height_mm: float
    suggested_k: int
    warnings: list[dict]
    background: dict | None = None             # {hex, share, removed} when an edge background was found
    _geom: dict = field(default_factory=dict)

    def geometry(self, mapping: dict[int, int], thin_lines: bool = True) -> Geometry:
        key = (tuple(sorted(mapping.items())), thin_lines)
        if key not in self._geom:
            self._geom[key] = _build_geometry(self.labels, mapping, thin_lines)
        return self._geom[key]


_SESSIONS: "OrderedDict[str, Session]" = OrderedDict()
_MAX_SESSIONS = 6


def get_session(sid: str) -> Session:
    if sid not in _SESSIONS:
        raise KeyError("Session expired — drop the image again.")
    _SESSIONS.move_to_end(sid)
    return _SESSIONS[sid]


# ---------------------------------------------------------------- analysis

def new_session(data: bytes, filename: str, width_mm: float, height_mm: float,
                n_colors: int | None, remove_bg: str = "auto") -> Session:
    """remove_bg: "auto"/"on" remove a detected edge background, "off" keeps it."""
    rgb, alpha, w_mm, h_mm = rasterize(data, filename, width_mm, height_mm)
    res = analyze(rgb, alpha, n_colors)

    background = None
    bg_id, share = frame_label(res.labels)
    if bg_id is not None and bg_id != TRANSPARENT:
        background = {"hex": res.hexes[bg_id], "share": round(share, 2), "removed": remove_bg != "off"}
    if background and background["removed"]:
        labels = res.labels
        keep = (labels != TRANSPARENT) & ~edge_connected(labels == bg_id)
        ys, xs = np.nonzero(keep)
        if not len(xs):
            raise ValueError("Removing the background left nothing to stitch.")
        H, W = labels.shape
        m = 2   # px of slack so anti-aliased edges aren't clipped
        x0, y0 = max(0, xs.min() - m), max(0, ys.min() - m)
        x1, y1 = min(W, xs.max() + 1 + m), min(H, ys.max() + 1 + m)
        if (x1 - x0) < 0.98 * W or (y1 - y0) < 0.98 * H:
            # Crop to the subject so it fills the requested size, then re-analyze.
            rgb, alpha, w_mm, h_mm = rasterize(data, filename, width_mm, height_mm,
                                               crop=(x0 / W, y0 / H, x1 / W, y1 / H))
            res = analyze(rgb, alpha, n_colors)
        _remove_background(res, background["hex"])

    s = Session(id=uuid.uuid4().hex, filename=filename, labels=res.labels, hexes=res.hexes,
                width_mm=w_mm, height_mm=h_mm, suggested_k=res.suggested_k, warnings=res.warnings,
                background=background)
    _SESSIONS[s.id] = s
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    return s


def _remove_background(res, bg_hex: str) -> None:
    """Edge-connected regions in the background color become empty (never stitched).
    Same-colored areas enclosed by the design stay a normal thread color."""
    labels = res.labels
    for i, hx in enumerate(res.hexes):
        if C.delta_e(hx, bg_hex) < C.KNOCKOUT_DE:
            labels[edge_connected(labels == i)] = TRANSPARENT
    counts = np.bincount(labels.ravel(), minlength=256)
    keep = [i for i in range(len(res.hexes)) if counts[i]]
    if not keep:
        raise ValueError("Removing the background left nothing to stitch.")
    lut = np.full(256, TRANSPARENT, np.uint8)
    for new, old in enumerate(keep):
        lut[old] = new
    res.labels = lut[labels]
    res.hexes = [res.hexes[i] for i in keep]


def session_payload(s: Session) -> dict:
    counts = np.bincount(s.labels.ravel(), minlength=256)
    px_mm2 = PX_PER_MM ** 2
    total = float(counts[: len(s.hexes)].sum()) or 1.0
    return {
        "session_id": s.id,
        "width_mm": s.width_mm,
        "height_mm": s.height_mm,
        "suggested_k": s.suggested_k,
        "k": len(s.hexes),
        "palette": [{"id": i, "hex": hx, "area_mm2": round(counts[i] / px_mm2, 1),
                     "share": round(counts[i] / total, 4)} for i, hx in enumerate(s.hexes)],
        "auto_merge": _auto_merge(s.hexes, counts),
        "background": s.background,
        "label_png": _label_png(s.labels),
        "warnings": s.warnings,
    }


def _auto_merge(hexes: list[str], counts: np.ndarray) -> dict[int, int]:
    """Near-identical colors (ΔE2000 < 5) merge into the larger one by default:
    on a single-needle machine every extra color is a manual thread change."""
    merged: dict[int, int] = {}
    keep: list[int] = []
    for i in sorted(range(len(hexes)), key=lambda i: -counts[i]):
        target = next((k for k in keep if C.delta_e(hexes[i], hexes[k]) < C.MERGE_DE), None)
        if target is None:
            keep.append(i)
        else:
            merged[i] = target
    return merged


def _png_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _label_png(labels: np.ndarray) -> str:
    """Label indices as a grayscale PNG so the browser can recolor it live."""
    h, w = labels.shape
    scale = min(1.0, PREVIEW_MAX_PX / max(h, w))
    small = cv2.resize(labels, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_NEAREST) if scale < 1 else labels
    return _png_data_url(small)


# ---------------------------------------------------------------- geometry

def _build_geometry(base_labels: np.ndarray, mapping: dict[int, int], thin_lines: bool) -> Geometry:
    lut = np.arange(256, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[src] = dst
    labels = lut[base_labels]

    ids = [int(i) for i in np.unique(labels) if i != TRANSPARENT]
    shapes, lines = {}, {}
    for i in ids:
        mask = labels == i
        if thin_lines:
            mask, lines[i] = split_thin(mask)
        else:
            lines[i] = []
        shapes[i] = mask_to_polygons(mask)
    counts = np.bincount(labels.ravel(), minlength=256)
    area = {i: float(counts[i]) / PX_PER_MM ** 2 for i in ids}

    adjacency: dict[int, dict[int, int]] = {i: {} for i in ids + [TRANSPARENT]}
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        diff = a != b
        codes = a[diff].astype(np.int32) * 256 + b[diff].astype(np.int32)
        for code, n in zip(*np.unique(codes, return_counts=True)):
            x, y = divmod(int(code), 256)
            for p, q in ((x, y), (y, x)):
                adjacency.setdefault(p, {})
                adjacency[p][q] = adjacency[p].get(q, 0) + int(n)

    frame = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    vals, cnts = np.unique(frame, return_counts=True)
    top = int(cnts.argmax())
    frame_label = int(vals[top]) if cnts[top] >= 0.5 * len(frame) else None
    return Geometry(labels, shapes, lines, area, adjacency, frame_label)


def _resolve_merges(colors: list[dict]) -> dict[int, int]:
    target = {c["id"]: c.get("merged_into") for c in colors}
    mapping = {}
    for cid in target:
        seen, cur = {cid}, cid
        while target.get(cur) is not None and target[cur] not in seen:
            cur = target[cur]
            seen.add(cur)
        if cur != cid:
            mapping[cid] = cur
    return mapping


# ---------------------------------------------------------------- color decisions

def build_plan(s: Session, req: dict) -> dict:
    """Fabric-aware color decisions. req keys: fabric_hex, prev_fabric_hex, colors
    [{id, hex, fabric (None|bool override), merged_into, was_fabric, restored_from}],
    order (list|None), outlines [ids], border (bool)."""
    fabric = req.get("fabric_hex") or "#ffffff"
    prev_fabric = req.get("prev_fabric_hex")
    by_id = {c["id"]: c for c in req["colors"]}
    mapping = _resolve_merges(req["colors"])
    geom = s.geometry(mapping, bool(req.get("thin_lines", True)))
    eff_ids = [c["id"] for c in req["colors"] if c["id"] not in mapping and c["id"] in geom.shapes]
    outlines = set(req.get("outlines") or [])
    warnings: list[dict] = []

    decided = {}
    for cid in eff_ids:
        c = by_id[cid]
        hx = c["hex"].lower()
        de = C.delta_e(hx, fabric)
        auto = de < C.KNOCKOUT_DE
        override = c.get("fabric")
        is_fabric = auto if override is None else bool(override)
        name = C.color_name(hx)
        restored = None if is_fabric else c.get("restored_from")
        if (not is_fabric and override is None and c.get("was_fabric")
                and prev_fabric and prev_fabric.lower() != fabric.lower()):
            restored = C.fabric_name(prev_fabric).lower()
        decided[cid] = {
            "id": cid, "hex": hx, "name": name, "fabric": is_fabric, "auto_fabric": auto,
            "override": override, "delta_e": round(de, 1), "restored_from": restored,
            "area_mm2": round(geom.area_mm2[cid], 1),
            "merged": [m for m, t in mapping.items() if t == cid],
            "light_on_dark": (not is_fabric) and C.is_light_on_dark(hx, fabric),
            "layer_name": f"{name} (fabric was {restored})" if restored else name,
        }
        if restored:
            warnings.append({"level": "info", "code": "restored", "color_id": cid,
                             "message": f"{name} was left as fabric on {restored} fabric. It doesn't match "
                                        f"{C.fabric_label(fabric)} fabric, so it is now stitched as "
                                        f"“{decided[cid]['layer_name']}”."})

    # Names must be unique per layer: "Green", "Green 2".
    seen: dict[str, int] = {}
    for cid in eff_ids:
        d = decided[cid]
        seen[d["layer_name"]] = seen.get(d["layer_name"], 0) + 1
        if seen[d["layer_name"]] > 1:
            d["layer_name"] = f'{d["layer_name"]} {seen[d["layer_name"]]}'

    fabric_ids = {cid for cid in eff_ids if decided[cid]["fabric"]} | {TRANSPARENT}
    stitched = [cid for cid in eff_ids if not decided[cid]["fabric"]]

    # Stitch order: keep the user's order if given, else suggest one.
    if req.get("order"):
        order = [i for i in req["order"] if i in decided]
        order += [i for i in eff_ids if i not in order]
        order_source = "user"
    else:
        order = suggest_order(stitched, decided, geom, fabric_ids)
        order += [i for i in eff_ids if i not in order]
        order_source = "auto"

    fabric_label = C.fabric_label(fabric)
    for cid in stitched:
        d = decided[cid]
        if d["delta_e"] < C.LOW_CONTRAST_DE and cid not in outlines:
            warnings.append({
                "level": "warn", "code": "low_contrast", "color_id": cid,
                "message": f"{d['name']} will barely show on {fabric_label} fabric (ΔE {d['delta_e']}).",
                "fixes": [
                    {"type": "swap", "color_id": cid, "hex": C.contrast_swap(d["hex"], fabric),
                     "label": "Swap thread color"},
                    {"type": "outline", "color_id": cid,
                     "label": f"Add 1 mm {C.contrasting_bw(fabric)[0].lower()} outline"},
                ],
            })
        if d["light_on_dark"]:
            warnings.append({"level": "info", "code": "light_on_dark", "color_id": cid,
                             "message": f"{d['name']} is a light thread on dark fabric: fill underlay on, "
                                        "0.25 mm row spacing so the fabric doesn't show through."})

    # Design edge: the color forming the outside of the shape.
    outer = _outermost(geom, fabric_ids)
    if outer is not None and outer in decided and not req.get("border"):
        d = decided[outer]
        if d["delta_e"] < C.LOW_CONTRAST_DE:
            warnings.append({
                "level": "warn", "code": "edge", "color_id": outer,
                "message": f"The outer color ({d['name']}) is close to the {fabric_label} fabric, "
                           "so the design's shape will disappear into it.",
                "fixes": [{"type": "border",
                           "label": f"Add thin {C.contrasting_bw(fabric)[0].lower()} border"}],
            })

    bobbin_name, bobbin_hex = C.bobbin_for(fabric)
    threads = []
    n = 0
    for cid in order:
        d = decided[cid]
        if d["fabric"]:
            threads.append({"order": None, "hex": d["hex"], "name": d["name"], "status": "not stitched — fabric"})
            continue
        n += 1
        threads.append({"order": n, "hex": d["hex"], "name": d["layer_name"], "status": "stitched"})
        if cid in outlines:
            n += 1
            bw = C.contrasting_bw(fabric)
            threads.append({"order": n, "hex": bw[1], "name": f"{bw[0]} outline ({d['name']})", "status": "stitched"})
    if req.get("border"):
        bw = C.contrasting_bw(fabric)
        threads.append({"order": n + 1, "hex": bw[1], "name": f"{bw[0]} border", "status": "stitched"})

    return {
        "colors": [decided[i] for i in order],
        "order": order,
        "order_source": order_source,
        "outermost": outer,
        "warnings": warnings,
        "summary": {"fabric_hex": fabric, "fabric_name": C.fabric_name(fabric),
                    "bobbin_name": bobbin_name, "bobbin_hex": bobbin_hex, "threads": threads},
        "_geom": geom,
    }


def suggest_order(stitched: list[int], decided: dict, geom: Geometry, fabric_ids: set[int]) -> list[int]:
    """Largest area first (the base), then colors touching the fabric and large areas,
    with lighter colors ahead of darker ones wherever the two touch."""
    if not stitched:
        return []
    area = geom.area_mm2
    adj = geom.adjacency
    touches = {i: any(adj.get(i, {}).get(f, 0) for f in fabric_ids) for i in stitched}
    first = max(stitched, key=lambda i: area[i])
    rest = [i for i in stitched if i != first]
    light = {i: C.lightness(decided[i]["hex"]) for i in rest}
    before = {i: set() for i in rest}   # i must come after everything in before[i]
    for a in rest:
        for b in rest:
            if a != b and adj.get(a, {}).get(b, 0) and light[a] > light[b] + 10:
                before[b].add(a)
    order = [first]
    remaining = set(rest)
    while remaining:
        ready = [i for i in remaining if not (before[i] & remaining)]
        pick = min(ready, key=lambda i: (not touches[i], -area[i]))
        order.append(pick)
        remaining.remove(pick)
    return order


def _outermost(geom: Geometry, fabric_ids: set[int]) -> int | None:
    bg = geom.frame_label
    if bg is not None and bg not in fabric_ids:
        return bg                                   # a stitched background is the edge itself
    ref = bg if bg is not None else TRANSPARENT
    neighbors = {k: v for k, v in geom.adjacency.get(ref, {}).items() if k != TRANSPARENT and k != ref}
    if not neighbors:
        return None
    return max(neighbors, key=neighbors.get)


# ---------------------------------------------------------------- generation

def generate(s: Session, req: dict) -> dict:
    t0 = time.time()
    plan = build_plan(s, req)
    geom: Geometry = plan.pop("_geom")
    fabric = plan["summary"]["fabric_hex"]
    outlines = set(req.get("outlines") or [])
    bw_name, bw_hex = C.contrasting_bw(fabric)
    decided = {d["id"]: d for d in plan["colors"]}

    layers = []
    for cid in plan["order"]:
        d = decided[cid]
        if d["fabric"]:
            continue
        shapes = [(p, stitch_params(p, d["light_on_dark"])) for p in geom.shapes[cid]]
        shapes += [(ln, dict(LINE_PARAMS)) for ln in geom.lines[cid]]
        layers.append({"label": d["layer_name"], "hex": d["hex"], "shapes": shapes, "color_id": cid, "kind": "fill"})
        if cid in outlines:
            ring = band(geom.shapes[cid], OUTLINE_MM)
            layers.append({"label": f"{bw_name} outline ({d['name']})", "hex": bw_hex, "kind": "outline",
                           "shapes": [(p, {"fill_method": "contour_fill"}) for p in ring]})
    if req.get("border"):
        fabric_ids = {i for i, d in decided.items() if d["fabric"]} | {TRANSPARENT}
        bg = geom.frame_label
        skip = {bg} if bg is not None and bg in fabric_ids else set()
        sil = unary_union([p for i, ps in geom.shapes.items() if i not in skip for p in ps]
                          + [ln.buffer(0.4) for i, ls in geom.lines.items() if i not in skip for ln in ls])
        ring = band(polygons(sil), BORDER_MM, outer_only=True)
        layers.append({"label": f"{bw_name} border", "hex": bw_hex, "kind": "border",
                       "shapes": [(p, {"fill_method": "contour_fill"}) for p in ring]})
    for n, layer in enumerate(layers, 1):
        layer["label"] = f"{n} {layer['label']}"
    _order_shapes(layers, Point(s.width_mm / 2, s.height_mm / 2))
    machine = _machine_summary(layers)

    # The SVG canvas is the SE400 hoop; the design sits centered on it at (50, 50).
    dx, dy = (HOOP_MM - s.width_mm) / 2, (HOOP_MM - s.height_mm) / 2
    placed = [{**l, "shapes": [(translate(p, dx, dy), prm) for p, prm in l["shapes"]]} for l in layers]
    meta = {**plan["summary"]}
    svg = write_svg(HOOP_MM, HOOP_MM, placed, meta)
    preview, preview_marked, geo_warnings = _render_and_check(s, layers, fabric)

    plan["warnings"] = s.warnings + plan["warnings"] + machine.pop("warnings") + geo_warnings
    plan["machine"] = machine
    plan["svg"] = svg
    plan["preview_png"] = preview
    plan["preview_marked_png"] = preview_marked
    plan["layers"] = [{"label": l["label"], "hex": l["hex"], "shapes": len(l["shapes"]),
                       "contour": sum(p.get("fill_method") == "contour_fill" for _, p in l["shapes"]),
                       "lines": sum("stroke_method" in p for _, p in l["shapes"])}
                      for l in layers]
    plan["width_mm"], plan["height_mm"] = s.width_mm, s.height_mm
    plan["elapsed_s"] = round(time.time() - t0, 2)
    return plan


def _order_shapes(layers: list[dict], start: Point) -> None:
    """Within each color, visit fills and outline lines in nearest-neighbor order,
    starting next to where the previous color finished, so stitching flows from a
    fill into the outline that touches it. The SE400 doesn't trim, so every
    remaining jump is a thread to cut by hand. Pure-fill colors get a 2-opt pass."""
    cursor = start                        # the machine starts at hoop center
    for layer in layers:
        left = list(layer["shapes"])
        route = []
        here = cursor                     # geometry we're leaving from
        while left:
            nxt = min(left, key=lambda it: it[0].distance(here))
            left.remove(nxt)
            g, prm = nxt
            if g.geom_type == "LineString":   # run lines from whichever end is closer
                if Point(g.coords[-1]).distance(here) < Point(g.coords[0]).distance(here):
                    g = LineString(g.coords[::-1])
                here = Point(g.coords[-1])
            else:
                here = g
            route.append((g, prm))
        if 1 < len(route) <= 40 and all(g.geom_type != "LineString" for g, _ in route):
            route = _two_opt(route, cursor)
        layer["shapes"] = route
        if route:
            last = route[-1][0]
            cursor = Point(last.coords[-1]) if last.geom_type == "LineString" else last.representative_point()


def _two_opt(route: list, start) -> list:
    n = len(route)
    polys = [p for p, _ in route]
    d0 = [p.distance(start) for p in polys]
    d = [[a.distance(b) if i != j else 0.0 for j, b in enumerate(polys)] for i, a in enumerate(polys)]

    def cost(order):
        return d0[order[0]] + sum(d[a][b] for a, b in zip(order, order[1:]))

    best = list(range(n))
    best_cost, improved = cost(best), True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                c = cost(cand)
                if c < best_cost - 1e-6:
                    best, best_cost, improved = cand, c, True
    return [route[i] for i in best]


def _machine_summary(layers: list[dict]) -> dict:
    """Brother SE400 view: thread changes, estimated stitches and run time."""
    blocks: list[str] = []
    for layer in layers:
        if layer["shapes"] and (not blocks or blocks[-1] != layer["hex"]):
            blocks.append(layer["hex"])
    warnings = []
    seen = set()
    for i, hx in enumerate(blocks):
        if hx in seen:
            warnings.append({"level": "warn", "code": "color_revisit",
                             "message": f"Thread {C.color_name(hx)} ({hx}) is used again later in the order — "
                                        "that's an extra manual thread change. Reorder or change the outline color to avoid it."})
        seen.add(hx)
    area = sum(g.area for l in layers for g, _ in l["shapes"])
    run = sum(g.length for l in layers for g, _ in l["shapes"] if g.geom_type == "LineString")
    est = int(round(area * EST_STITCHES_PER_MM2 + run * 3 / 2.0, -2))   # triple run, 2 mm stitches
    jumps = sum(max(0, len(l["shapes"]) - 1) for l in layers)
    if est > STITCH_WARN:
        warnings.append({"level": "warn", "code": "stitch_count",
                         "message": f"About {est:,} stitches — over {STITCH_WARN:,}, roughly "
                                    f"{est / SPEED_SPM:.0f} min of stitching on the SE400. Consider a smaller size "
                                    "or leaving a large area as fabric."})
    return {"thread_blocks": len(blocks), "thread_changes": max(0, len(blocks) - 1),
            "estimated_stitches": est, "estimated_minutes": round(est / SPEED_SPM, 1),
            "estimated_jumps": jumps, "hoop_mm": HOOP_MM, "warnings": warnings}


def _fill(mask: np.ndarray, poly: Polygon) -> None:
    rings = [poly.exterior, *poly.interiors]
    pts = [np.round((np.asarray(r.coords) * PX_PER_MM - 0.5) * 8).astype(np.int32) for r in rings]
    cv2.fillPoly(mask, pts, 1, lineType=cv2.LINE_8, shift=3)


def _components(mask: np.ndarray, min_px: float) -> tuple[int, np.ndarray]:
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_px) + 1
    return len(keep), np.isin(cc, keep)


def _disk(mm: float) -> np.ndarray:
    d = max(3, round(mm * PX_PER_MM) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _render_and_check(s: Session, layers: list[dict], fabric_hex: str):
    h, w = s.labels.shape
    img = np.empty((h, w, 3), np.uint8)
    img[:] = C.hex_to_rgb(fabric_hex)[::-1]          # BGR for OpenCV
    covered = np.zeros((h, w), np.uint8)
    thin_all = np.zeros((h, w), bool)
    warnings = []
    px2 = PX_PER_MM ** 2

    line_px = max(1, round(LINE_DRAW_MM * PX_PER_MM))
    for layer in layers:
        m = np.zeros((h, w), np.uint8)
        for g, _ in layer["shapes"]:
            if g.geom_type != "LineString":
                _fill(m, g)
        stroke = np.zeros((h, w), np.uint8)
        for g, _ in layer["shapes"]:
            if g.geom_type == "LineString":
                pts = np.round((np.asarray(g.coords) * PX_PER_MM - 0.5) * 8).astype(np.int32)
                cv2.polylines(stroke, [pts], False, 1, line_px, cv2.LINE_8, shift=3)
        bgr = C.hex_to_rgb(layer["hex"])[::-1]
        img[m.astype(bool)] = bgr
        img[stroke.astype(bool)] = bgr
        covered |= m | stroke
        if layer["kind"] == "fill":
            narrow = m.astype(bool) & ~cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(NARROW_STITCH_MM)).astype(bool)
            n, marked = _components(narrow, 0.3 * px2)
            if n:
                thin_all |= marked
                warnings.append({"level": "warn", "code": "narrow_stitch",
                                 "message": f"{layer['label']}: {n} area(s) narrower than {NARROW_STITCH_MM:g} mm "
                                            "— fill stitches may not hold there (orange on preview)."})

    fabric_mask = (covered == 0).astype(np.uint8)
    gaps = fabric_mask.astype(bool) & ~cv2.morphologyEx(fabric_mask, cv2.MORPH_OPEN, _disk(NARROW_GAP_MM)).astype(bool)
    n_gaps, gap_mask = _components(gaps, 0.1 * px2)
    if n_gaps:
        warnings.append({"level": "warn", "code": "narrow_gap",
                         "message": f"{n_gaps} fabric gap(s) narrower than {NARROW_GAP_MM:g} mm will likely close up "
                                    "when stitched (red on preview)."})

    scale = min(1.0, PREVIEW_MAX_PX / max(h, w))
    grow = np.ones((2 * int(1 / scale) + 1,) * 2, np.uint8)
    marked = img.copy()
    if thin_all.any():
        t = cv2.dilate(thin_all.astype(np.uint8), grow).astype(bool)
        marked[t] = (0.45 * marked[t] + 0.55 * np.array([0, 150, 255])).astype(np.uint8)
    if gap_mask.any():
        marked[cv2.dilate(gap_mask.astype(np.uint8), grow).astype(bool)] = (40, 30, 230)

    def small(a):
        return cv2.resize(a, (max(1, round(w * scale)), max(1, round(h * scale))),
                          interpolation=cv2.INTER_AREA) if scale < 1 else a
    return _png_data_url(small(img)), _png_data_url(small(marked)), warnings
