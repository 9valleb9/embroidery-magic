"""Orchestration: analysis sessions, fabric-aware color decisions, and SVG generation.

Geometry depends only on (image, size, palette size, merges). Fabric color, hex
edits, knockouts and stitch order are "color decisions" and never re-trace shapes.

Where this sits in the flow::

    upload -> raster.py rasterize -> palette.py colors + label map
           -> vectorize.py shapes + stitch settings
           -> pipeline.py (this module: sessions, color decisions, generation)
           -> svg_out.py SVG -> export.py Ink/Stitch machine file

``app.py`` is the FastAPI layer that calls into this module; ``static/index.html`` is
the UI.

Main entry points:
    * ``new_session``: rasterize + analyze an upload once and cache the result.
    * ``get_session``: fetch a cached session by id (``KeyError`` once expired).
    * ``session_payload``: the JSON the UI gets back after analysis.
    * ``build_plan``: fabric-aware color decisions (knockouts, merges, stitch order,
      contrast warnings, thread list). Fast, no SVG.
    * ``generate``: the full result: layers, overlap, stitch ordering, SVG, previews,
      machine summary, fabric advice and warnings.

Key concepts:
    * Label map: a ``uint8`` array (H x W) at ``raster.PX_PER_MM`` (20 px/mm) where each
      pixel holds a palette id (index into ``Session.hexes``). ``TRANSPARENT`` (255)
      marks pixels with no design. Ids are small integers, so 256-entry lookup tables
      (``lut[labels]``) are used to apply merges and re-numbering.
    * Units: all geometry (shapely polygons, lines, satins, areas) is in mm, with the
      origin at the design's top-left. Pixel quantities are converted with
      ``PX_PER_MM``. The SVG canvas is the hoop, with the design centered on it.
    * Color decisions vs geometry: merges, thin-line mode and satin mode change the
      traced shapes, so they form the geometry cache key
      ``(sorted merge mapping, thin_lines, satin)`` in ``Session._geom``. Everything
      else (fabric color, hex edits, fabric overrides, order, outlines, border,
      overlap, pull compensation, fabric type, hoop) is applied on top without
      re-tracing.
    * Fabric ids: colors decided to be "fabric" are not stitched; they stay as holes
      in neighboring colors. ``TRANSPARENT`` is always treated as fabric too.

Gotchas:
    * Sessions are kept in memory only (``_SESSIONS``, LRU of ``_MAX_SESSIONS``). They
      are lost on restart and are not shared between worker processes.
    * The per-session geometry cache is never evicted; it lives as long as the session.
    * ``build_plan`` returns an internal ``"_geom"`` key (a ``Geometry`` object).
      Callers that serialize the plan must pop it first.
    * Background removal crops and re-analyzes, so palette ids after ``new_session``
      are not the ids of a plain analysis of the same image.
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
                        split_satin, split_strokes, split_thin, stitch_params)

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
    """Traced shapes for one geometry cache key (merge mapping, thin_lines, satin).

    Attributes:
        labels: Label map with merges applied (same shape as ``Session.labels``).
        shapes: Color id -> fill polygons in mm (after thin lines / satins are split off).
        lines: Color id -> centerlines in mm stitched as triple running stitch
            (thin-line mode; empty lists otherwise).
        satins: Color id -> satin columns (satin mode; empty lists otherwise).
        area_mm2: Color id -> pixel area of that color in the merged label map, in mm².
        adjacency: Color id -> neighbor id -> shared edge length in px, counted on
            the merged label map. Includes ``TRANSPARENT`` as a key.
        frame_label: Label covering at least half the image border, else ``None``.
    """
    labels: np.ndarray                         # merged label map
    shapes: dict[int, list[Polygon]]
    lines: dict[int, list[LineString]]         # outline mode: thin features as centerlines
    satins: dict[int, list[Satin]]             # satin mode: steady 1–7 mm strokes as satin columns
    area_mm2: dict[int, float]
    adjacency: dict[int, dict[int, int]]       # id -> neighbor id -> shared edge length (px)
    frame_label: int | None                    # label covering most of the image frame (>= 50%)


@dataclass
class Session:
    """One analyzed upload, cached in memory and addressed by ``id``.

    Attributes:
        id: Random hex id handed to the UI.
        filename: Original upload name (app.py uses its stem for the export file name).
        labels: Label map at ``PX_PER_MM`` (palette ids, ``TRANSPARENT`` for empty).
        hexes: Palette colors as ``#rrggbb``; index = label id.
        width_mm: Actual design width in mm (aspect kept inside the requested box).
        height_mm: Actual design height in mm.
        suggested_k: Palette size the analysis would choose automatically.
        warnings: Analysis warnings (photo input, many colors, ...), shown with every plan.
        background: ``{hex, share, removed}`` when an edge background was found.
        _geom: Geometry cache, keyed by ``(sorted mapping items, thin_lines, satin)``.
    """
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
        """Return the traced geometry for these settings, building and caching it on first use.

        Args:
            mapping: Merge mapping, color id -> id it is merged into (already resolved
                to final targets, see ``_resolve_merges``).
            thin_lines: Split features under 1 mm into running-stitch centerlines.
            satin: Split steady 1 to 7 mm strokes into satin columns.

        Returns:
            The cached ``Geometry``. Tracing is the slow step, so repeated plan calls
            with the same merges are cheap.
        """
        key = (tuple(sorted(mapping.items())), thin_lines, satin)
        if key not in self._geom:
            self._geom[key] = _build_geometry(self.labels, mapping, thin_lines, satin)
        return self._geom[key]


_SESSIONS: "OrderedDict[str, Session]" = OrderedDict()
_MAX_SESSIONS = 6


def get_session(sid: str) -> Session:
    """Return a cached session and mark it most recently used.

    Raises:
        KeyError: If the session was evicted or never existed (for example after a restart).
    """
    if sid not in _SESSIONS:
        raise KeyError("Session expired — drop the image again.")
    _SESSIONS.move_to_end(sid)
    return _SESSIONS[sid]


# ---------------------------------------------------------------- analysis

def new_session(data: bytes, filename: str, width_mm: float, height_mm: float,
                n_colors: int | None, remove_bg: str = "auto", sliver_mm: float = MORPH_DISK_MM) -> Session:
    """Rasterize and analyze an upload, then cache it as a new session.

    remove_bg: "auto"/"on" remove a detected edge background, "off" keeps it.
    sliver_mm: color bands narrower than this are absorbed into their neighbors.

    Args:
        data: Raw file bytes (raster image or SVG).
        filename: Upload name; its extension helps detect SVG input.
        width_mm: Width of the box the design is fitted into, in mm.
        height_mm: Height of the box the design is fitted into, in mm.
        n_colors: Palette size, or ``None`` to let ``palette.analyze`` choose.
        remove_bg: See above.
        sliver_mm: See above.

    Returns:
        The new ``Session``. When a background is removed and the remaining subject is
        noticeably smaller than the frame, the image is cropped to the subject and
        re-rendered so the subject fills the requested size.

    Raises:
        ValueError: If removing the background leaves nothing to stitch (and whatever
            ``rasterize`` / ``analyze`` raise for unreadable input).

    Side effects:
        Adds the session to ``_SESSIONS`` and evicts the oldest ones beyond
        ``_MAX_SESSIONS``.
    """
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
        # Only re-render when the crop is worth it (subject clearly smaller than the frame).
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
    Same-colored areas enclosed by the design stay a normal thread color.

    Every palette color within ``C.KNOCKOUT_DE`` of ``bg_hex`` is cleared where it
    touches the border. Colors left with no pixels are dropped and the remaining ids
    are renumbered to stay contiguous.

    Args:
        res: The ``palette.analyze`` result; ``res.labels`` and ``res.hexes`` are
            replaced in place.
        bg_hex: Detected background color.

    Raises:
        ValueError: If no color is left.
    """
    labels = res.labels
    for i, hx in enumerate(res.hexes):
        if C.delta_e(hx, bg_hex) < C.KNOCKOUT_DE:
            labels[edge_connected(labels == i)] = TRANSPARENT
    counts = np.bincount(labels.ravel(), minlength=256)
    keep = [i for i in range(len(res.hexes)) if counts[i]]
    if not keep:
        raise ValueError("Removing the background left nothing to stitch.")
    # Old id -> new contiguous id; anything not kept (incl. TRANSPARENT) maps to TRANSPARENT.
    lut = np.full(256, TRANSPARENT, np.uint8)
    for new, old in enumerate(keep):
        lut[old] = new
    res.labels = lut[labels]
    res.hexes = [res.hexes[i] for i in keep]


def session_payload(s: Session) -> dict:
    """Build the ``/api/analyze`` response for a session.

    Returns:
        A JSON-ready dict: session id, size in mm, suggested and actual palette size,
        the palette (id, hex, area in mm², share of the design), default merges
        (``auto_merge``), detected background, ``label_png`` (the label map as a
        grayscale PNG data URL for live recoloring in the browser), analysis warnings
        and stitch-mode recommendations.
    """
    counts = np.bincount(s.labels.ravel(), minlength=256)
    px_mm2 = PX_PER_MM ** 2
    total = float(counts[: len(s.hexes)].sum()) or 1.0   # design pixels only, TRANSPARENT excluded
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
        "recommend": recommend(s, _auto_merge(s.hexes, counts)),
    }


SATIN_RECOMMEND_SHARE = 0.4   # recommend satin when this much of the design is stroke-like


def recommend(s: Session, mapping: dict[int, int]) -> dict:
    """Stitch options that suit this image: satin when most of it is strokes under 7 mm wide
    (script, swashes, line art, borders), fills when it's mostly solid areas. Estimated at
    5 px/mm with a width test (what a 7 mm disk can't fit in is stroke), not the full satin
    layout, so it stays fast. Thin-line outlines stay on: satin mode needs them for hairlines.

    Args:
        s: The session.
        mapping: Merge mapping to apply before measuring (usually the auto merges).

    Returns:
        ``{thin_lines, satin, satin_share, why}``: recommended toggles, the measured
        stroke share (0 to 1) and a sentence explaining the choice for the UI.
    """
    lut = np.arange(256, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[src] = dst
    labels = lut[s.labels]
    f = 5 / PX_PER_MM
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))   # 7 mm at 5 px/mm
    stroke = total = 0
    for i in np.unique(labels):
        if i == TRANSPARENT:
            continue
        m = cv2.resize((labels == i).astype(np.uint8) * 255, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) > 127
        # Opening keeps only what a 7 mm disk fits inside; the rest is stroke-like.
        wide = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk).astype(bool)
        stroke += int((m & ~wide).sum())
        total += int(m.sum())
    share = stroke / total if total else 0.0
    use = share >= SATIN_RECOMMEND_SHARE
    why = (f"About {share:.0%} of this design is strokes under 7 mm wide (lettering, swashes, borders), "
           + ("so satin columns are recommended." if use else "so fills suit it better than satin."))
    return {"thin_lines": True, "satin": use, "satin_share": round(share, 2), "why": why}


def _auto_merge(hexes: list[str], counts: np.ndarray) -> dict[int, int]:
    """Near-identical colors (ΔE2000 < 5) merge into the larger one by default:
    on a single-needle machine every extra color is a manual thread change.

    Colors are visited from largest to smallest pixel count, so a color always merges
    into a bigger one that was kept.

    Returns:
        Color id -> id it merges into (only for merged colors).
    """
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
    """Encode an image array (grayscale or BGR) as a ``data:image/png;base64`` URL."""
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _label_png(labels: np.ndarray) -> str:
    """Label indices as a grayscale PNG so the browser can recolor it live.

    Each pixel value is a label id (``TRANSPARENT`` = 255). The map is scaled down to at
    most ``PREVIEW_MAX_PX`` on its long side with nearest-neighbor sampling, which never
    invents ids that aren't in the map.
    """
    h, w = labels.shape
    scale = min(1.0, PREVIEW_MAX_PX / max(h, w))
    small = cv2.resize(labels, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_NEAREST) if scale < 1 else labels
    return _png_data_url(small)


# ---------------------------------------------------------------- geometry

def _build_geometry(base_labels: np.ndarray, mapping: dict[int, int], thin_lines: bool,
                    satin: bool = False) -> Geometry:
    """Trace the label map into mm geometry for one cache key (slow; use ``Session.geometry``).

    Applies the merge mapping, then for each color: splits off thin features as
    centerlines (``thin_lines``), splits off satin columns (``satin``) and traces what
    remains into fill polygons. Also measures per-color area, pixel adjacency and the
    frame label.

    Args:
        base_labels: The session's label map (not modified).
        mapping: Resolved merge mapping, color id -> target id.
        thin_lines: Enable running-stitch centerlines for features under 1 mm.
        satin: Enable satin columns for steady 1 to 7 mm strokes.

    Returns:
        A new ``Geometry``.
    """
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
            # what the columns leave can still hold too-thin strokes (a 0.6 mm crossbar
            # between its serifs): they'd scribble as fills, so they join the hairlines
            mask, more = split_strokes(mask)
            lines[i] = lines[i] + more
            # hairlines become 1 mm satin too, so they don't read as loose threads next to satin
            hair = [line_satin(ln) for ln in lines[i]]
            satins[i] += [h for h in hair if h is not None]
            lines[i] = [ln for ln, h in zip(lines[i], hair) if h is None]
        else:
            satins[i] = []
        shapes[i] = mask_to_polygons(mask)
    counts = np.bincount(labels.ravel(), minlength=256)
    area = {i: float(counts[i]) / PX_PER_MM ** 2 for i in ids}

    # Count label changes between horizontal and vertical pixel neighbors. Each pair is
    # packed into one int (a * 256 + b) so np.unique can count all pairs at once.
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
    """Flatten ``merged_into`` chains (A -> B -> C becomes A -> C, B -> C).

    Cycles are cut where they would revisit a color, so a bad request can't loop forever.

    Returns:
        Color id -> final target id, only for colors that are merged away.
    """
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
    order (list|None), outlines [ids], border (bool).

    Also reads ``thin_lines`` and ``satin`` (they select the cached geometry).

    For each color that survives merging: decides fabric vs stitched (ΔE2000 under
    ``C.KNOCKOUT_DE`` is fabric unless overridden), names its layer, flags restored
    knockouts, light-on-dark thread and low contrast; picks the stitch order (user order
    kept, missing ids appended); checks the design edge; and builds the thread list
    including outline and border threads.

    Args:
        s: The session.
        req: The plan request as a dict (``PlanIn.model_dump()``).

    Returns:
        A dict with ``colors`` (per-color decisions in stitch order), ``order``,
        ``order_source`` ("user" or "auto"), ``outermost``, ``warnings``, ``summary``
        (fabric, bobbin, threads) and ``_geom``, the internal ``Geometry``. Pop ``_geom``
        before returning the plan as JSON.
    """
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
        # Was auto-knocked-out on the previous fabric but stitches on this one.
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
    with lighter colors ahead of darker ones wherever the two touch.

    "Lighter" means more than 10 L* lighter; that rule is a strict order, so the
    dependency graph has no cycles and a ready color always exists.

    Args:
        stitched: Ids of colors that are stitched (not fabric).
        decided: Per-color decisions from ``build_plan`` (used for the hex).
        geom: Geometry providing areas and adjacency.
        fabric_ids: Ids treated as fabric, including ``TRANSPARENT``.

    Returns:
        The stitched ids in suggested order (fabric ids are not included).
    """
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
    """The color that forms the design's outside edge, or ``None`` if there is none.

    A stitched frame label is the edge itself. Otherwise it is the color sharing the
    longest pixel border with the frame label (or with ``TRANSPARENT`` when no label
    dominates the frame).
    """
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
    """Build the full stitch-ready result for a plan request.

    Steps: ``build_plan``; per stitched color, grow fills by the overlap and attach
    stitch params (tatami/contour, satin, running-stitch lines); add outline and border
    layers; number layers; order shapes within each layer to cut jumps; orient the hoop;
    center the design on the hoop and write the SVG; render previews and run the
    narrow-area / narrow-gap checks; add hoop-edge and fabric warnings.

    Args:
        s: The session.
        req: The plan request dict. Besides the ``build_plan`` keys it reads
            ``fabric_type``, ``pull_comp_mm``, ``overlap_mm``, ``hoop_w``, ``hoop_h``
            and ``speed_spm``.

    Returns:
        The plan dict (without ``_geom``) plus ``fabric`` (before-you-stitch advice),
        ``machine`` (thread changes, stitch/time/jump estimates, hoop), ``svg`` (text),
        ``preview_png`` and ``preview_marked_png`` (data URLs), ``layers`` (per-layer
        counts), ``width_mm`` / ``height_mm``, ``elapsed_s`` and the merged ``warnings``
        (analysis + plan + machine + geometry).
    """
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
    # Number layers in stitch order ("1 Dark green") so Ink/Stitch and the user see the order.
    for n, layer in enumerate(layers, 1):
        layer["label"] = f"{n} {layer['label']}"
    _order_shapes(layers, Point(s.width_mm / 2, s.height_mm / 2))
    hoop_w, hoop_h = M.oriented(s.width_mm, s.height_mm, *_hoop(req))
    speed = req.get("speed_spm") or M.DEFAULT_SPEED_SPM
    # Tighter rows than standard mean proportionally more stitches per mm².
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
    # Checks run on the unplaced layers, i.e. in design coordinates matching s.labels.
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
    slivers of fabric between them. It never grows outward past the design's edge.

    Args:
        geom: Geometry for the plan.
        stitched: Ids of stitched colors (fabric colors don't count as design).
        mm: Overlap distance in mm; 0 or less disables growth.

    Returns:
        A function mapping one fill polygon to its grown pieces (pieces under 0.2 mm²
        are dropped; if nothing is left the original polygon is returned).
    """
    if mm <= 0:
        return lambda p: [p]
    union = unary_union([p for cid in stitched for p in geom.shapes[cid]]
                        + [sat.poly for cid in stitched for sat in geom.satins[cid]])
    # Closing = the design plus gaps narrower than 2*mm, but no further out than its edge.
    allowed = union.buffer(mm, join_style="round").buffer(-mm, join_style="round")   # closing

    def grow(p: Polygon) -> list[Polygon]:
        """Grow one polygon by ``mm``, clipped to the closed design outline."""
        g = p.buffer(mm, join_style="round").intersection(allowed).buffer(0)
        return [q for q in polygons(g) if q.area >= 0.2] or [p]
    return grow


def _hoop(req: dict) -> tuple[float, float]:
    """Hoop (width, height) in mm from the request, falling back to the default hoop."""
    h = M.HOOPS[M.DEFAULT_HOOP]
    return float(req.get("hoop_w") or h["w"]), float(req.get("hoop_h") or h["h"])


def _order_shapes(layers: list[dict], start: Point) -> None:
    """Within each color, visit fills and outline lines in nearest-neighbor order,
    starting next to where the previous color finished, so stitching flows from a
    fill into the outline that touches it. Most home machines don't trim jumps, so
    every remaining jump is a thread to cut by hand. Pure-fill colors get a 2-opt pass.

    Distances are shapely ``distance`` (closest points), measured from the previous
    shape itself for fills and from the end point for lines. Lines may be reversed so
    they start at the nearer end. The 2-opt pass is limited to 40 shapes per layer.

    Args:
        layers: Layers in stitch order; each layer's ``shapes`` list is replaced in place.
        start: Where the machine starts (hoop center in design coordinates).
    """
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
    """Improve a fill visiting order with 2-opt (reverse sub-runs while it gets shorter).

    The cost is an open path: distance from ``start`` to the first shape plus the gaps
    between consecutive shapes. Distances are precomputed, so this is O(n²) memory and
    roughly O(n³) per improving pass; callers keep n small.

    Args:
        route: ``(geometry, params)`` pairs.
        start: Starting point (shapely geometry).

    Returns:
        The same pairs in the improved order.
    """
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
    settings the fabric type applied, and notes about this design on this fabric.

    Args:
        s: The session (for design size).
        fab: Fabric type entry from ``fabrics``.
        pull: Pull compensation in mm that was applied.
        layers: Final layers (counted for lines and satins).
        machine: ``_machine_summary`` result (for the stitch estimate).
        geo_warnings: Geometry warnings so far; fabric limit warnings (beanie height,
            stitch count) are appended to it in place.
        hoop: Oriented hoop (w, h) in mm, used to suggest a bigger size.

    Returns:
        The ``fabric`` block for the response: key, label, stabilizer, needle, pull comp,
        density label, underlay and notes.
    """
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
    # The short side is the height that matters: the design can be turned to fit.
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
    density: fill stitches relative to standard 0.25 mm rows (the fabric type's row spacing).

    Consecutive layers with the same thread hex count as one block. A thread that comes
    back later gets a ``color_revisit`` warning. Stitches are estimated from fill area
    (``EST_STITCHES_PER_MM2`` x density) plus running-stitch length (triple run at 2 mm
    per stitch), rounded to hundreds. Jumps are counted as shapes per layer minus one.

    Args:
        layers: Final, ordered layers.
        speed: Machine speed in stitches per minute.
        hoop: Oriented hoop (w, h) in mm, echoed back.
        density: See above.

    Returns:
        Thread blocks/changes/hexes, estimated stitches, minutes and jumps, hoop, speed,
        and ``warnings`` (the caller pops these into the main warning list).
    """
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
    """Rasterize a mm polygon (with holes) into ``mask`` at ``PX_PER_MM``, in place.

    Coordinates are shifted by half a pixel (mm edges -> pixel centers) and passed with
    3 fractional bits (``shift=3``) for sub-pixel accuracy. Holes are cut because all
    rings go to one ``fillPoly`` call.
    """
    rings = [poly.exterior, *poly.interiors]
    pts = [np.round((np.asarray(r.coords) * PX_PER_MM - 0.5) * 8).astype(np.int32) for r in rings]
    cv2.fillPoly(mask, pts, 1, lineType=cv2.LINE_8, shift=3)


def _components(mask: np.ndarray, min_px: float) -> tuple[int, np.ndarray]:
    """Keep 8-connected components of at least ``min_px`` pixels.

    Returns:
        (number of kept components, boolean mask of the kept components).
    """
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_px) + 1
    return len(keep), np.isin(cc, keep)


def _disk(mm: float) -> np.ndarray:
    """Elliptical structuring element ``mm`` wide at ``PX_PER_MM`` (odd size, at least 3 px)."""
    d = max(3, round(mm * PX_PER_MM) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _render_and_check(s: Session, layers: list[dict], fabric_hex: str):
    """Paint the stitched preview and flag areas that won't stitch well.

    Layers are painted in stitch order over the fabric color at the label map's
    resolution (fills filled, lines drawn ``LINE_DRAW_MM`` wide). Two checks run on the
    painted result:

    * narrow stitch: parts of a fill layer a ``NARROW_STITCH_MM`` disk can't fit in
      (components of at least 0.3 mm²), marked orange;
    * narrow gap: uncovered fabric a ``NARROW_GAP_MM`` disk can't fit in (components
      of at least 0.1 mm²), marked red.

    Args:
        s: The session (for the canvas size).
        layers: Final layers in design coordinates (not yet centered on the hoop).
        fabric_hex: Fabric color for the background.

    Returns:
        (preview PNG data URL, marked preview PNG data URL, list of warnings). Previews
        are scaled to at most ``PREVIEW_MAX_PX`` on the long side.
    """
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
    # Dilate marks by about one preview pixel so they stay visible after downscaling.
    grow = np.ones((2 * int(1 / scale) + 1,) * 2, np.uint8)
    marked = img.copy()
    if thin_all.any():
        t = cv2.dilate(thin_all.astype(np.uint8), grow).astype(bool)
        marked[t] = (0.45 * marked[t] + 0.55 * np.array([0, 150, 255])).astype(np.uint8)
    if gap_mask.any():
        marked[cv2.dilate(gap_mask.astype(np.uint8), grow).astype(bool)] = (40, 30, 230)

    def small(a):
        """Downscale an image to preview size (no-op when it's already small enough)."""
        return cv2.resize(a, (max(1, round(w * scale)), max(1, round(h * scale))),
                          interpolation=cv2.INTER_AREA) if scale < 1 else a
    return _png_data_url(small(img)), _png_data_url(small(marked)), warnings
