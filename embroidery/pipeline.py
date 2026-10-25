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
from . import fabrics as F
from .palette import MORPH_DISK_MM, TRANSPARENT, analyze, edge_connected, frame_label
from . import machines as M
from .export import EST_STITCHES_PER_MM2, STITCH_WARN
from .raster import PX_PER_MM, rasterize
from .svg_out import write_svg
from .vectorize import (LINE_PARAMS, Satin, band, line_satin, mask_to_polygons, polygons, satin_params,
                        split_satin, split_thin, stitch_params)

OUTLINE_MM = 1.0
BORDER_MM = 1.0
NARROW_STITCH_MM = 1.0
NARROW_GAP_MM = 0.6
PREVIEW_MAX_PX = 1000
LINE_DRAW_MM = 0.7         # how wide a triple running stitch reads, for preview + gap checks
DEFAULT_OVERLAP_MM = 0.3   # each color reaches this far under the colors it touches
DEFAULT_PULL_COMP_MM = 0.2 # tatami rows stretched this far past each edge


@dataclass
class Geometry:
    labels: np.ndarray                         # merged label map
    shapes: dict[int, list[Polygon]]
    lines: dict[int, list[LineString]]         # outline mode: thin features as centerlines
    satins: dict[int, list[Satin]]             # satin mode: steady 1–7 mm strokes as satin columns
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

    def geometry(self, mapping: dict[int, int], thin_lines: bool = True, satin: bool = False) -> Geometry:
        key = (tuple(sorted(mapping.items())), thin_lines, satin)
        if key not in self._geom:
            self._geom[key] = _build_geometry(self.labels, mapping, thin_lines, satin)
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
                n_colors: int | None, remove_bg: str = "auto", sliver_mm: float = MORPH_DISK_MM) -> Session:
    """remove_bg: "auto"/"on" remove a detected edge background, "off" keeps it.
    sliver_mm: color bands narrower than this are absorbed into their neighbors."""
    rgb, alpha, w_mm, h_mm = rasterize(data, filename, width_mm, height_mm)
    res = analyze(rgb, alpha, n_colors, sliver_mm)

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
            res = analyze(rgb, alpha, n_colors, sliver_mm)
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

def _build_geometry(base_labels: np.ndarray, mapping: dict[int, int], thin_lines: bool,
                    satin: bool = False) -> Geometry:
    lut = np.arange(256, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[src] = dst
    labels = lut[base_labels]

    ids = [int(i) for i in np.unique(labels) if i != TRANSPARENT]
    shapes, lines, satins = {}, {}, {}
    for i in ids:
        mask = labels == i
        if thin_lines:
            mask, lines[i] = split_thin(mask)
        else:
            lines[i] = []
        if satin:
            mask, satins[i] = split_satin(mask)
            # hairlines become 1 mm satin too, so they don't read as loose threads next to satin
            hair = [line_satin(ln) for ln in lines[i]]
            satins[i] += [h for h in hair if h is not None]
            lines[i] = [ln for ln, h in zip(lines[i], hair) if h is None]
        else:
            satins[i] = []
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
    return Geometry(labels, shapes, lines, satins, area, adjacency, frame_label)


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
    geom = s.geometry(mapping, bool(req.get("thin_lines", True)), bool(req.get("satin", False)))
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

    fab = F.get(req.get("fabric_type"))
    pull = float(req.get("pull_comp_mm", DEFAULT_PULL_COMP_MM) or 0)
    overlap = float(req.get("overlap_mm", DEFAULT_OVERLAP_MM) or 0)
    stitched = [cid for cid in plan["order"] if not decided[cid]["fabric"]]
    grow = _overlap(geom, stitched, overlap)
    satin_rows = fab["row_spacing_mm"]

    layers = []
    for cid in plan["order"]:
        d = decided[cid]
        if d["fabric"]:
            continue
        shapes = [(g, prm) for p in geom.shapes[cid]
                  for prm in [stitch_params(p, d["light_on_dark"], pull, fab["row_spacing_mm"], fab["underlay"])]
                  for g in grow(p)]
        shapes += [(sat.poly, {**satin_params(sat.width_mm, pull, satin_rows, fab["underlay"], d["light_on_dark"]),
                               "_satin": sat}) for sat in geom.satins[cid]]
        shapes += [(ln, dict(LINE_PARAMS)) for ln in geom.lines[cid]]
        layers.append({"label": d["layer_name"], "hex": d["hex"], "shapes": shapes, "color_id": cid, "kind": "fill"})
        if cid in outlines:
            ring = band(geom.shapes[cid] + [sat.poly for sat in geom.satins[cid]], OUTLINE_MM)
            layers.append({"label": f"{bw_name} outline ({d['name']})", "hex": bw_hex, "kind": "outline",
                           "shapes": [(p, {"fill_method": "contour_fill"}) for p in ring]})
    if req.get("border"):
        fabric_ids = {i for i, d in decided.items() if d["fabric"]} | {TRANSPARENT}
        bg = geom.frame_label
        skip = {bg} if bg is not None and bg in fabric_ids else set()
        sil = unary_union([p for i, ps in geom.shapes.items() if i not in skip for p in ps]
                          + [st.poly for i, ss in geom.satins.items() if i not in skip for st in ss]
                          + [ln.buffer(0.4) for i, ls in geom.lines.items() if i not in skip for ln in ls])
        ring = band(polygons(sil), BORDER_MM, outer_only=True)
        layers.append({"label": f"{bw_name} border", "hex": bw_hex, "kind": "border",
                       "shapes": [(p, {"fill_method": "contour_fill"}) for p in ring]})
    for n, layer in enumerate(layers, 1):
        layer["label"] = f"{n} {layer['label']}"
    _order_shapes(layers, Point(s.width_mm / 2, s.height_mm / 2))
    hoop_w, hoop_h = M.oriented(s.width_mm, s.height_mm, *_hoop(req))
    speed = req.get("speed_spm") or M.DEFAULT_SPEED_SPM
    density = F.STANDARD_ROW_SPACING_MM / (fab["row_spacing_mm"] or F.STANDARD_ROW_SPACING_MM)
    machine = _machine_summary(layers, speed, (hoop_w, hoop_h), density)

    # The SVG canvas is the hoop's stitch field; the design sits centered on it.
    dx, dy = (hoop_w - s.width_mm) / 2, (hoop_h - s.height_mm) / 2
    placed = [{**l, "shapes": [(translate(p, dx, dy),
                                {**prm, "_satin": prm["_satin"].translated(dx, dy)} if "_satin" in prm else prm)
                               for p, prm in l["shapes"]]} for l in layers]
    meta = {**plan["summary"], "fabric_type": fab["label"], "stabilizer": fab["stabilizer"],
            "needle": fab["needle"]}
    svg = write_svg(hoop_w, hoop_h, placed, meta)
    preview, preview_marked, geo_warnings = _render_and_check(s, layers, fabric)
    if max(s.width_mm - hoop_w, s.height_mm - hoop_h) + 2 * pull > 0:
        geo_warnings.append({"level": "warn", "code": "hoop_edge",
                             "message": f"With {pull:g} mm pull compensation the stitches reach past the "
                                        f"{hoop_w:g} × {hoop_h:g} mm hoop — make the design a little smaller."})

    plan["fabric"] = _fabric_advice(s, fab, pull, layers, machine, geo_warnings, (hoop_w, hoop_h))
    plan["warnings"] = s.warnings + plan["warnings"] + machine.pop("warnings") + geo_warnings
    plan["machine"] = machine
    plan["svg"] = svg
    plan["preview_png"] = preview
    plan["preview_marked_png"] = preview_marked
    plan["layers"] = [{"label": l["label"], "hex": l["hex"], "shapes": len(l["shapes"]),
                       "contour": sum(p.get("fill_method") == "contour_fill" for _, p in l["shapes"]),
                       "lines": sum("stroke_method" in p for _, p in l["shapes"]),
                       "satins": sum("_satin" in p for _, p in l["shapes"])}
                      for l in layers]
    plan["width_mm"], plan["height_mm"] = s.width_mm, s.height_mm
    plan["elapsed_s"] = round(time.time() - t0, 2)
    return plan


def _overlap(geom: Geometry, stitched: list[int], mm: float):
    """Each fill grows by `mm` into the stitched colors it touches, and into fabric gaps
    narrower than 2×mm (they'd close up anyway), so neighbors overlap instead of leaving
    slivers of fabric between them. It never grows outward past the design's edge."""
    if mm <= 0:
        return lambda p: [p]
    union = unary_union([p for cid in stitched for p in geom.shapes[cid]]
                        + [sat.poly for cid in stitched for sat in geom.satins[cid]])
    allowed = union.buffer(mm, join_style="round").buffer(-mm, join_style="round")   # closing

    def grow(p: Polygon) -> list[Polygon]:
        g = p.buffer(mm, join_style="round").intersection(allowed).buffer(0)
        return [q for q in polygons(g) if q.area >= 0.2] or [p]
    return grow


def _hoop(req: dict) -> tuple[float, float]:
    h = M.HOOPS[M.DEFAULT_HOOP]
    return float(req.get("hoop_w") or h["w"]), float(req.get("hoop_h") or h["h"])


def _order_shapes(layers: list[dict], start: Point) -> None:
    """Within each color, visit fills and outline lines in nearest-neighbor order,
    starting next to where the previous color finished, so stitching flows from a
    fill into the outline that touches it. Most home machines don't trim jumps, so
    every remaining jump is a thread to cut by hand. Pure-fill colors get a 2-opt pass."""
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


def _fabric_advice(s: Session, fab: dict, pull: float, layers: list[dict], machine: dict,
                   geo_warnings: list[dict], hoop: tuple[float, float]) -> dict:
    """The "Before you stitch" box: what goes under and on top of the fabric, the stitch
    settings the fabric type applied, and notes about this design on this fabric."""
    notes = list(fab["extras"])
    n_lines = sum("stroke_method" in p for l in layers for _, p in l["shapes"])
    narrow = any(w["code"] == "narrow_stitch" for w in geo_warnings)
    if n_lines or narrow:
        mw, mh = M.max_design(*hoop)
        scale = min(mw / s.width_mm, mh / s.height_mm)
        what = f"{n_lines} thin feature{'s' if n_lines != 1 else ''} will stitch as outlines" if n_lines \
            else "Some areas are under 1 mm wide"
        bigger = (f" A bigger design holds more of them as fills: this hoop allows up to "
                  f"{s.width_mm * scale:.0f} × {s.height_mm * scale:.0f} mm.") if scale > 1.05 else ""
        notes.append(f"{what}.{bigger}")
    n_satins = sum("_satin" in p for l in layers for _, p in l["shapes"])
    if n_satins:
        notes.append(f"{n_satins} stroke{'s' if n_satins != 1 else ''} stitch as satin. Satin can snag on "
                     "loopy or textured fabric, so use topping there.")
    if fab.get("max_height_mm") and min(s.width_mm, s.height_mm) > fab["max_height_mm"]:
        geo_warnings.append({"level": "warn", "code": "fabric_height",
                             "message": f"The design is {min(s.width_mm, s.height_mm):g} mm tall. A folded "
                                        f"beanie cuff gives about {fab['max_height_mm']} mm, so make it smaller."})
    if fab.get("stitch_warn") and machine["estimated_stitches"] > fab["stitch_warn"]:
        geo_warnings.append({"level": "warn", "code": "fabric_stitches",
                             "message": f"About {machine['estimated_stitches']:,} stitches is heavy for "
                                        f"{fab['label'].lower()} (aim for under {fab['stitch_warn']:,}). "
                                        "Make the design smaller or leave large areas as fabric."})
    return {"key": fab["key"], "label": fab["label"], "stabilizer": fab["stabilizer"], "needle": fab["needle"],
            "pull_comp_mm": pull, "density": F.density_label(fab["row_spacing_mm"]),
            "underlay": fab["underlay"], "notes": notes}


def _machine_summary(layers: list[dict], speed: float, hoop: tuple[float, float], density: float = 1.0) -> dict:
    """Single-needle view: thread changes, estimated stitches and run time.
    density: fill stitches relative to standard 0.25 mm rows (the fabric type's row spacing)."""
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
    est = int(round(area * EST_STITCHES_PER_MM2 * density + run * 3 / 2.0, -2))   # triple run, 2 mm stitches
    jumps = sum(max(0, len(l["shapes"]) - 1) for l in layers)
    if est > STITCH_WARN:
        warnings.append({"level": "warn", "code": "stitch_count",
                         "message": f"About {est:,} stitches — over {STITCH_WARN:,}, roughly "
                                    f"{est / speed:.0f} min of stitching at {speed:g} spm. Consider a smaller size "
                                    "or leaving a large area as fabric."})
    return {"thread_blocks": len(blocks), "thread_changes": max(0, len(blocks) - 1), "thread_hexes": blocks,
            "estimated_stitches": est, "estimated_minutes": round(est / speed, 1),
            "estimated_jumps": jumps, "hoop_mm": list(hoop), "speed_spm": speed, "warnings": warnings}


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
