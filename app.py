"""Embroidery Magic — logo image -> Ink/Stitch-ready embroidery SVG.

Run:  python app.py      (opens http://127.0.0.1:8765)
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
    id: int
    hex: str
    fabric: Optional[bool] = None          # None = automatic (ΔE to fabric decides)
    merged_into: Optional[int] = None
    was_fabric: Optional[bool] = None       # effective state under prev_fabric_hex
    restored_from: Optional[str] = None


class PlanIn(BaseModel):
    session_id: str
    fabric_hex: str = "#ffffff"
    prev_fabric_hex: Optional[str] = None
    colors: list[ColorIn]
    order: Optional[list[int]] = None       # None = suggested order
    outlines: list[int] = []
    border: bool = False
    thin_lines: bool = True                 # stitch < 1 mm features as triple running-stitch outlines
    hoop_w: float = machines.HOOPS[machines.DEFAULT_HOOP]["w"]
    hoop_h: float = machines.HOOPS[machines.DEFAULT_HOOP]["h"]
    format: str = machines.DEFAULT_FORMAT
    speed_spm: float = machines.DEFAULT_SPEED_SPM
    pull_comp_mm: float = pipeline.DEFAULT_PULL_COMP_MM   # Ink/Stitch pull compensation (tatami fills)
    overlap_mm: float = pipeline.DEFAULT_OVERLAP_MM       # how far each color reaches under its neighbors
    fabric_type: str = fabrics.DEFAULT_FABRIC             # sets row spacing + underlay; the UI also loads its pull comp


def _session(sid: str) -> pipeline.Session:
    try:
        return pipeline.get_session(sid)
    except KeyError as e:
        raise HTTPException(410, str(e))


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), width_mm: float = Form(99), height_mm: float = Form(99),
                  hoop_w: float = Form(100), hoop_h: float = Form(100),
                  n_colors: str = Form("auto"), remove_bg: str = Form("auto"),
                  sliver_mm: float = Form(pipeline.MORPH_DISK_MM)):
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
    k = None if n_colors in ("", "auto") else int(n_colors)
    try:
        s = pipeline.new_session(data, file.filename or "upload", width_mm, height_mm, k, remove_bg, sliver_mm)
    except Exception as e:  # unreadable image, bad SVG, fully transparent, ...
        raise HTTPException(422, f"Couldn't read that image: {e}")
    return pipeline.session_payload(s)


def _check_hoop(w: float, h: float) -> None:
    if not (10 <= w <= machines.MAX_HOOP_MM and 10 <= h <= machines.MAX_HOOP_MM):
        raise HTTPException(400, f"Hoop size must be between 10 and {machines.MAX_HOOP_MM:g} mm.")


def _check_req(req: PlanIn) -> None:
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
    p = pipeline.build_plan(_session(req.session_id), req.model_dump())
    p.pop("_geom", None)
    return p


@app.post("/api/generate")
def generate(req: PlanIn):
    _check_req(req)
    return pipeline.generate(_session(req.session_id), req.model_dump())


@app.get("/api/machine")
def machine():
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
    """Generate the SVG and start a background machine-file export through Ink/Stitch."""
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
    try:
        j = export.job(jid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {k: v for k, v in j.items() if k != "data"}


@app.get("/api/export/{jid}/file")
def export_file(jid: str):
    try:
        j = export.job(jid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    if j.get("status") != "done":
        raise HTTPException(409, "Export isn't finished.")
    return Response(j["data"], media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{j["filename"]}"'})


def main() -> None:
    import uvicorn

    url = f"http://{HOST}:{PORT}"
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Embroidery Magic running at {url}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
