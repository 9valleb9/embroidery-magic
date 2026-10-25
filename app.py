"""Embroidery Magic — logo image -> Ink/Stitch-ready embroidery SVG.

Run:  python app.py      (opens http://127.0.0.1:8765)

This module is the HTTP layer: a small FastAPI app that serves the single-page UI
(``static/index.html``, no build step) and a JSON API around ``embroidery.pipeline``.
It holds no image state of its own. All heavy lifting happens in the package:

    upload -> raster.py (rasterize at raster.PX_PER_MM = 20 px/mm)
           -> palette.py (k-means colors, label map)
           -> vectorize.py (shapes, stitch settings)
           -> pipeline.py (sessions, color decisions, generation)
           -> svg_out.py (Ink/Stitch SVG)
           -> export.py (Ink/Stitch machine file, optional)

API flow used by the UI:

1. ``POST /api/analyze`` (multipart upload + size options) rasterizes and analyzes the
   image once, stores a ``pipeline.Session`` in memory and returns its palette, a label
   map PNG and stitch recommendations.
2. ``POST /api/plan`` (``PlanIn``) applies color decisions (fabric color, knockouts,
   merges, order) and returns the color plan without writing an SVG. The UI calls it on
   every edit, so it is meant to be quick.
3. ``POST /api/generate`` (``PlanIn``) builds the full result: SVG, previews, warnings,
   machine summary.
4. ``POST /api/export`` starts a background Ink/Stitch export; poll
   ``GET /api/export/{jid}`` and download from ``GET /api/export/{jid}/file``.

``GET /api/machine`` returns the static option lists (formats, hoops, fabric types) the
UI needs at startup.

Units: every size in the API is in millimeters. Color ids in ``PlanIn.colors`` are the
palette ids returned by ``/api/analyze``.

Gotchas:
    * Sessions live only in this process (``pipeline._SESSIONS``, a small LRU of 6).
      A restart, or too many newer uploads, expires a session; the API answers
      410 Gone and the UI asks the user to drop the image again. Run a single worker.
    * ``/api/plan`` does not run ``_check_req``; only generate and export validate the
      hoop, format and stitch settings.
    * Machine-file export needs a local Ink/Stitch install (``export.find_inkstitch``);
      without it ``/api/export`` returns 501 and the user exports from Inkscape.

Environment: ``HOST`` and ``PORT`` set the bind address, ``NO_BROWSER`` (any value)
stops ``main`` from opening a browser tab.
"""
from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from embroidery import export, fabrics, machines, pipeline

HERE = Path(__file__).parent
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8765"))
MAX_UPLOAD = 25 * 1024 * 1024

app = FastAPI(title="Embroidery Magic")


class ColorIn(BaseModel):
    """One palette color as the UI currently sees it, sent back on every plan request.

    Attributes:
        id: Palette id from ``/api/analyze`` (index into the session's colors).
        hex: Thread color as ``#rrggbb``. May differ from the analyzed color after a user edit.
        fabric: ``True``/``False`` forces "leave as fabric" / "stitch"; ``None`` lets the
            ΔE2000 distance to the fabric color decide.
        merged_into: Id of another color this one is merged into, or ``None``.
        was_fabric: Whether this color was effectively fabric under ``prev_fabric_hex``;
            used to detect knockouts restored by a fabric change.
        restored_from: Fabric name a previous plan restored this color from, echoed back
            so the restored layer name stays stable.
    """
    id: int
    hex: str
    fabric: Optional[bool] = None          # None = automatic (ΔE to fabric decides)
    merged_into: Optional[int] = None
    was_fabric: Optional[bool] = None       # effective state under prev_fabric_hex
    restored_from: Optional[str] = None


class PlanIn(BaseModel):
    """Request body for ``/api/plan``, ``/api/generate`` and ``/api/export``.

    It is passed to the pipeline as a plain dict (``model_dump()``); see
    ``pipeline.build_plan`` and ``pipeline.generate`` for how each key is used. Sizes are
    in mm, speed in stitches per minute, and ``outlines`` / ``order`` hold palette ids.
    """
    session_id: str
    fabric_hex: str = "#ffffff"
    prev_fabric_hex: Optional[str] = None
    colors: list[ColorIn]
    order: Optional[list[int]] = None       # None = suggested order
    outlines: list[int] = []
    border: bool = False
    thin_lines: bool = True                 # stitch < 1 mm features as triple running-stitch outlines
    satin: bool = False                     # stitch steady 1–7 mm strokes as satin columns
    hoop_w: float = machines.HOOPS[machines.DEFAULT_HOOP]["w"]
    hoop_h: float = machines.HOOPS[machines.DEFAULT_HOOP]["h"]
    format: str = machines.DEFAULT_FORMAT
    speed_spm: float = machines.DEFAULT_SPEED_SPM
    pull_comp_mm: float = pipeline.DEFAULT_PULL_COMP_MM   # Ink/Stitch pull compensation (tatami fills)
    overlap_mm: float = pipeline.DEFAULT_OVERLAP_MM       # how far each color reaches under its neighbors
    fabric_type: str = fabrics.DEFAULT_FABRIC             # sets row spacing + underlay; the UI also loads its pull comp


def _session(sid: str) -> pipeline.Session:
    """Look up a session, turning an expired or unknown id into HTTP 410 Gone."""
    try:
        return pipeline.get_session(sid)
    except KeyError as e:
        raise HTTPException(410, str(e))


@app.get("/")
def index():
    """Serve the single-page UI."""
    return FileResponse(HERE / "static" / "index.html")


@app.get("/logo.svg")
def logo():
    """Serve the app logo (used by the UI header and browser tab)."""
    return FileResponse(HERE / "static" / "logo.svg", media_type="image/svg+xml")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), width_mm: float = Form(99), height_mm: float = Form(99),
                  hoop_w: float = Form(100), hoop_h: float = Form(100),
                  n_colors: str = Form("auto"), remove_bg: str = Form("auto"),
                  sliver_mm: float = Form(pipeline.MORPH_DISK_MM)):
    """Analyze an uploaded image and start a new session.

    Args:
        file: The logo (raster image or SVG), at most 25 MB.
        width_mm: Requested design width in mm (at least 5).
        height_mm: Requested design height in mm (at least 5). The design keeps its
            aspect ratio inside width_mm x height_mm (see ``raster.rasterize``).
        hoop_w: Hoop width in mm, used only to check that the size fits.
        hoop_h: Hoop height in mm, used only to check that the size fits.
        n_colors: Palette size as a string, or ``"auto"`` / ``""`` to let the palette
            step choose.
        remove_bg: ``"auto"`` / ``"on"`` remove a detected edge background, ``"off"``
            keeps it.
        sliver_mm: Color bands narrower than this (0.1 to 2 mm) are absorbed into
            their neighbors.

    Returns:
        ``pipeline.session_payload`` for the new session (session id, palette, label
        map PNG, warnings, recommendations).

    Raises:
        HTTPException: 400 for bad sizes, 413 for oversize uploads, 422 when the image
            can't be read or nothing is left to stitch.

    Side effects:
        Stores the session in the in-memory session cache, possibly evicting the
        oldest one.
    """
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File is larger than 25 MB.")
    _check_hoop(hoop_w, hoop_h)
    if not machines.fits(width_mm, height_mm, *machines.max_design(hoop_w, hoop_h)):
        mw, mh = machines.max_design(hoop_w, hoop_h)
        raise HTTPException(400, f"{width_mm:g} × {height_mm:g} mm doesn't fit the "
                                 f"{machines.hoop_label(hoop_w, hoop_h)} hoop — the largest design is {mw:g} × {mh:g} mm "
                                 f"({machines.EDGE_MARGIN_MM:g} mm kept free at each edge for pull compensation).")
    if width_mm < 5 or height_mm < 5:
        raise HTTPException(400, "Size must be at least 5 mm.")
    if not (0.1 <= sliver_mm <= 2):
        raise HTTPException(400, "Sliver cleanup must be between 0.1 and 2 mm.")
    # Form fields arrive as strings; "auto" means let the palette step pick k.
    k = None if n_colors in ("", "auto") else int(n_colors)
    try:
        s = pipeline.new_session(data, file.filename or "upload", width_mm, height_mm, k, remove_bg, sliver_mm)
    except Exception as e:  # unreadable image, bad SVG, fully transparent, ...
        raise HTTPException(422, f"Couldn't read that image: {e}")
    return pipeline.session_payload(s)


def _check_hoop(w: float, h: float) -> None:
    """Reject hoop sizes outside 10 mm to ``machines.MAX_HOOP_MM`` with HTTP 400."""
    if not (10 <= w <= machines.MAX_HOOP_MM and 10 <= h <= machines.MAX_HOOP_MM):
        raise HTTPException(400, f"Hoop size must be between 10 and {machines.MAX_HOOP_MM:g} mm.")


def _check_req(req: PlanIn) -> None:
    """Validate the machine and stitch settings of a plan request.

    Checks hoop size, output format, machine speed, pull compensation, overlap and
    fabric type. Raises HTTP 400 on the first bad value.
    """
    _check_hoop(req.hoop_w, req.hoop_h)
    if req.format.lower() not in machines.FORMATS:
        raise HTTPException(400, f"Unsupported format: {req.format}")
    if not (50 <= req.speed_spm <= 2000):
        raise HTTPException(400, "Machine speed must be between 50 and 2000 stitches per minute.")
    if not (0 <= req.pull_comp_mm <= machines.MAX_PULL_COMP_MM):
        raise HTTPException(400, f"Pull compensation must be between 0 and {machines.MAX_PULL_COMP_MM:g} mm.")
    if not (0 <= req.overlap_mm <= 2):
        raise HTTPException(400, "Overlap must be between 0 and 2 mm.")
    if req.fabric_type.lower() not in fabrics.FABRICS:
        raise HTTPException(400, f"Unknown fabric type: {req.fabric_type}")


@app.post("/api/plan")
def plan(req: PlanIn):
    """Return the color plan (fabric decisions, order, warnings, thread list) without an SVG.

    Cheap after the first call for a given merge set, because the session caches the
    traced geometry. Raises 410 if the session has expired.
    """
    p = pipeline.build_plan(_session(req.session_id), req.model_dump())
    # _geom is the internal Geometry object (numpy + shapely), not JSON-serializable.
    p.pop("_geom", None)
    return p


@app.post("/api/generate")
def generate(req: PlanIn):
    """Validate the request and return the full result from ``pipeline.generate``.

    The response includes the Ink/Stitch SVG text, preview PNGs (data URLs), layers,
    machine summary, fabric advice and all warnings.
    """
    _check_req(req)
    return pipeline.generate(_session(req.session_id), req.model_dump())


@app.get("/api/machine")
def machine():
    """Static options for the UI: formats, hoops, fabric types, limits and defaults.

    Also reports whether Ink/Stitch was found (``inkstitch``), which decides whether the
    UI offers direct machine-file export or shows ``manual_steps`` instead.
    """
    exe = export.find_inkstitch()
    return {"formats": {k: {"label": v["label"], "machines": v["machines"], "palette": v["palette"]}
                        for k, v in machines.FORMATS.items()},
            "default_format": machines.DEFAULT_FORMAT, "hoops": machines.HOOPS,
            "default_hoop": machines.DEFAULT_HOOP, "default_speed_spm": machines.DEFAULT_SPEED_SPM,
            "stitch_warn": export.STITCH_WARN, "max_stitch_mm": export.MAX_STITCH_MM,
            "inkstitch": bool(exe), "manual_steps": export.MANUAL_STEPS,
            "fabrics": fabrics.FABRICS, "default_fabric": fabrics.DEFAULT_FABRIC}


@app.post("/api/export")
def start_export(req: PlanIn):
    """Generate the SVG and start a background machine-file export through Ink/Stitch.

    Returns:
        ``{"job_id": ...}`` to poll with ``/api/export/{jid}``.

    Raises:
        HTTPException: 400 for invalid settings, 410 for an expired session, 501 when
            Ink/Stitch isn't installed.

    Side effects:
        Starts an export job in ``export`` (runs in the background). The download is
        named ``<upload stem>-<hoop w>x<hoop h>mm.<format>``.
    """
    _check_req(req)
    if not export.find_inkstitch():
        raise HTTPException(501, "Ink/Stitch isn't installed — export the machine file from Inkscape instead.")
    s = _session(req.session_id)
    r = pipeline.generate(s, req.model_dump())
    fmt = req.format.lower()
    m = r["machine"]
    base = Path(s.filename).stem or "design"
    hoop_w, hoop_h = m["hoop_mm"]
    jid = export.start_export(r["svg"], fmt, m["thread_hexes"], req.fabric_hex, (hoop_w, hoop_h),
                              m["speed_spm"], f"{base}-{hoop_w:g}x{hoop_h:g}mm.{fmt}")
    return {"job_id": jid}


@app.get("/api/export/{jid}")
def export_status(jid: str):
    """Report an export job's state (status, filename, format, ...), 404 if unknown."""
    try:
        j = export.job(jid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {k: v for k, v in j.items() if k != "data"}   # file bytes come from /file, not the status poll


@app.get("/api/export/{jid}/file")
def export_file(jid: str):
    """Download a finished export as an attachment (404 unknown job, 409 not done yet)."""
    try:
        j = export.job(jid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    if j.get("status") != "done":
        raise HTTPException(409, "Export isn't finished.")
    return Response(j["data"], media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{j["filename"]}"'})


def main() -> None:
    """Run the server with uvicorn and, unless ``NO_BROWSER`` is set, open the UI.

    The browser opens on a 1 s timer so the server is listening by then.
    """
    import uvicorn

    url = f"http://{HOST}:{PORT}"
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Embroidery Magic running at {url}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
