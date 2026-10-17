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

from embroidery import export, pipeline
from embroidery.export import HOOP_MM

HERE = Path(__file__).parent
HOST = "127.0.0.1"
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


def _session(sid: str) -> pipeline.Session:
    try:
        return pipeline.get_session(sid)
    except KeyError as e:
        raise HTTPException(410, str(e))


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), width_mm: float = Form(HOOP_MM), height_mm: float = Form(HOOP_MM),
                  n_colors: str = Form("auto")):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File is larger than 25 MB.")
    if width_mm > HOOP_MM or height_mm > HOOP_MM:
        raise HTTPException(400, f"The Brother SE400 stitch field is {HOOP_MM:g}×{HOOP_MM:g} mm — "
                                 "choose a size up to 100 mm.")
    if width_mm < 5 or height_mm < 5:
        raise HTTPException(400, "Size must be at least 5 mm.")
    k = None if n_colors in ("", "auto") else int(n_colors)
    try:
        s = pipeline.new_session(data, file.filename or "upload", width_mm, height_mm, k)
    except Exception as e:  # unreadable image, bad SVG, fully transparent, ...
        raise HTTPException(422, f"Couldn't read that image: {e}")
    return pipeline.session_payload(s)


@app.post("/api/plan")
def plan(req: PlanIn):
    p = pipeline.build_plan(_session(req.session_id), req.model_dump())
    p.pop("_geom", None)
    return p


@app.post("/api/generate")
def generate(req: PlanIn):
    return pipeline.generate(_session(req.session_id), req.model_dump())


@app.get("/api/machine")
def machine():
    exe = export.find_inkstitch()
    return {"name": "Brother SE400", "hoop_mm": HOOP_MM, "speed_spm": export.SPEED_SPM,
            "stitch_warn": export.STITCH_WARN, "inkstitch": bool(exe), "manual_steps": export.MANUAL_STEPS}


@app.post("/api/export")
def start_export(req: PlanIn):
    """Generate the SVG and start a background .pes export through Ink/Stitch."""
    if not export.find_inkstitch():
        raise HTTPException(501, "Ink/Stitch isn't installed — export the .pes from Inkscape instead.")
    s = _session(req.session_id)
    r = pipeline.generate(s, req.model_dump())
    base = Path(s.filename).stem or "design"
    jid = export.start_export(r["svg"], r["machine"]["thread_blocks"], req.fabric_hex, f"{base}-se400.pes")
    return {"job_id": jid}


@app.get("/api/export/{jid}")
def export_status(jid: str):
    try:
        j = export.job(jid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {k: v for k, v in j.items() if k != "pes"}


@app.get("/api/export/{jid}/pes")
def export_file(jid: str):
    try:
        j = export.job(jid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    if j.get("status") != "done":
        raise HTTPException(409, "Export isn't finished.")
    return Response(j["pes"], media_type="application/octet-stream",
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
