"""Machine file export (.pes, .jef, .vp3, .exp, .xxx, .dst) through the Ink/Stitch CLI,
then verification of the written file with pyembroidery (bounds, colors, stitch
count, hand trims)."""
from __future__ import annotations

import base64
import glob
import math
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
import pyembroidery as pe

from . import colors as C
from . import machines as M

STITCH_WARN = 25_000
MAX_STITCH_MM = 12.1      # DST's per-stitch limit; Brother hardware allows 12.7 mm
EXPORT_TIMEOUT_S = 600
# Ink/Stitch defaults (0.25 mm rows, 3 mm stitches, underlay) come out near this
# density; used only for the estimate shown before the real export finishes.
EST_STITCHES_PER_MM2 = 1.9

_CANDIDATES = [
    "~/Library/Application Support/org.inkscape.Inkscape/config/inkscape/extensions/inkstitch/inkstitch.app/Contents/MacOS/inkstitch",
    "/Applications/Inkscape.app/Contents/Resources/share/inkscape/extensions/inkstitch/inkstitch.app/Contents/MacOS/inkstitch",
    "~/.config/inkscape/extensions/inkstitch/bin/inkstitch",
]


def find_inkstitch() -> str | None:
    env = os.environ.get("INKSTITCH_BIN")
    if env:
        return env if os.access(env, os.X_OK) else None
    for c in _CANDIDATES:
        for p in glob.glob(os.path.expanduser(c)):
            if os.access(p, os.X_OK):
                return p
    return shutil.which("inkstitch")


MANUAL_STEPS = [
    "Open the downloaded SVG in Inkscape (with Ink/Stitch installed).",
    "Extensions › Ink/Stitch › Visualize and Export › Simulator — check the stitch-out.",
    "File › Save a Copy… and choose your machine's Ink/Stitch format (e.g. “Brother Embroidery Format (.pes)”).",
    "Copy the file to the machine over USB (most show up as a removable disk).",
]


def export_file(svg_text: str, fmt: str = M.DEFAULT_FORMAT) -> bytes:
    exe = find_inkstitch()
    if not exe:
        raise RuntimeError("Ink/Stitch not found")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "design.svg"
        src.write_text(svg_text, encoding="utf-8")
        r = subprocess.run([exe, "--extension=output", f"--format={M.fmt(fmt)}", str(src)],
                           capture_output=True, timeout=EXPORT_TIMEOUT_S, cwd=tmp)
    if r.returncode != 0 or not r.stdout:
        msg = r.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
        raise RuntimeError("Ink/Stitch export failed: " + " / ".join(msg))
    return r.stdout


def _cmd(s) -> int:
    return s[2] & pe.COMMAND_MASK


def analyze_file(data: bytes, fmt: str = M.DEFAULT_FORMAT, expected_colors: int | None = None,
                 fabric_hex: str = "#ffffff", hoop: tuple[float, float] = (100.0, 100.0),
                 speed: float = M.DEFAULT_SPEED_SPM, thread_hexes: list[str] | None = None) -> dict:
    """Read a machine file back and check it against the hoop and machine limits.
    thread_hexes: your actual threads in stitch order, used to draw the stitch-out
    (the file itself may only hold the nearest palette colors, or none)."""
    fmt = M.fmt(fmt)
    hoop_w, hoop_h = hoop
    problems: list[str] = []
    sig = M.FORMATS[fmt]["signature"]
    if sig and not data.startswith(sig):
        problems.append(f"File doesn't start with the {sig.decode()} signature.")
    with tempfile.NamedTemporaryFile(suffix="." + fmt) as f:
        f.write(data)
        f.flush()
        pattern = pe.read(f.name)
    if pattern is None:
        return {"ok": False, "problems": ["pyembroidery could not read the file back."]}

    stitches = pattern.stitches
    pts = np.array([(x, y) for x, y, c in stitches if (c & pe.COMMAND_MASK) == pe.STITCH], float)
    if not len(pts):
        return {"ok": False, "problems": ["The file contains no stitches."]}
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w_mm, h_mm = (x1 - x0) / 10, (y1 - y0) / 10
    if not M.fits(w_mm, h_mm, hoop_w, hoop_h):
        problems.append(f"Stitches span {w_mm:.1f}×{h_mm:.1f} mm — larger than the "
                        f"{M.hoop_label(hoop_w, hoop_h)} hoop.")

    # Walk the stitch list: color blocks, longest stitch, and in-color jumps (hand trims).
    threads = pattern.threadlist
    trims, blocks = [], []
    color, block_stitches, started = 0, 0, False
    last = None
    longest = 0.0
    jump_from = None
    for x, y, c in stitches:
        cmd = c & pe.COMMAND_MASK
        if cmd == pe.STITCH:
            if jump_from is not None and started:
                d = math.hypot(x - jump_from[0], y - jump_from[1]) / 10
                if d > 0.5:
                    trims.append({"color": color + 1, "from": jump_from, "to": (x, y), "length_mm": round(d, 1)})
            elif last is not None and started:
                longest = max(longest, math.hypot(x - last[0], y - last[1]) / 10)
            jump_from = None
            started = True
            block_stitches += 1
            last = (x, y)
        elif cmd in (pe.JUMP, pe.TRIM):
            if jump_from is None and last is not None:
                jump_from = last
        elif cmd == pe.COLOR_CHANGE:
            blocks.append(block_stitches)
            color, block_stitches, started, jump_from, last = color + 1, 0, False, None, None
        elif cmd == pe.END:
            break
    blocks.append(block_stitches)
    blocks = [b for b in blocks if b]
    if longest > MAX_STITCH_MM:
        problems.append(f"Longest stitch is {longest:.1f} mm (limit {MAX_STITCH_MM} mm).")
    if expected_colors is not None and len(blocks) != expected_colors:
        problems.append(f"Expected {expected_colors} color blocks, found {len(blocks)}.")

    total = int(len(pts))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    thread_info = []
    for i, n in enumerate(blocks):
        t = threads[i] if i < len(threads) else None
        has_color = t is not None and M.FORMATS[fmt]["palette"] is not None
        thread_info.append({"order": i + 1, "stitches": n,
                            "machine_name": (t.description or "").strip() if has_color else "",
                            "hex": t.hex_color() if has_color else None})
    for n, t in enumerate(trims, 1):   # positions in mm from the hoop's top-left, like the SVG
        t["number"] = n
        t["from_mm"] = [round((t["from"][0] - cx) / 10 + hoop_w / 2, 1), round((t["from"][1] - cy) / 10 + hoop_h / 2, 1)]
        t["to_mm"] = [round((t["to"][0] - cx) / 10 + hoop_w / 2, 1), round((t["to"][1] - cy) / 10 + hoop_h / 2, 1)]
        t["color_name"] = thread_info[t["color"] - 1]["machine_name"] if t["color"] <= len(thread_info) else ""

    return {
        "ok": not problems,
        "problems": problems,
        "format": fmt,
        "palette": M.FORMATS[fmt]["palette"],
        "signature": data[:8].decode("latin-1") if sig else None,
        "hoop_mm": [hoop_w, hoop_h],
        "max_stitch_mm": MAX_STITCH_MM,
        "size_bytes": len(data),
        "stitches": total,
        "color_blocks": len(blocks),
        "thread_changes": max(0, len(blocks) - 1),
        "minutes": round(total / speed, 1),
        "speed_spm": speed,
        "extent_mm": [round(w_mm, 1), round(h_mm, 1)],
        "longest_stitch_mm": round(longest, 1),
        "threads": thread_info,
        "trims": [{k: v for k, v in t.items() if k not in ("from", "to")} for t in trims],
        "preview_png": _render(stitches, _draw_colors(threads, fmt, len(blocks), thread_hexes),
                               (x0, y0, x1, y1), fabric_hex, trims, (hoop_w, hoop_h)),
    }


def _draw_colors(threads, fmt: str, n_blocks: int, thread_hexes: list[str] | None) -> list[str]:
    if thread_hexes and len(thread_hexes) == n_blocks:
        return list(thread_hexes)
    if M.FORMATS[fmt]["palette"]:
        return [t.hex_color() for t in threads]
    return []


def _render(stitches, colors, bounds, fabric_hex, trims, hoop, px_per_mm: float = 8.0) -> str:
    """Stitch-out preview on the hoop: thread lines in their PES colors, and the
    in-color jumps you'll trim by hand drawn dashed red and numbered."""
    hoop_w, hoop_h = hoop
    px_per_mm = min(px_per_mm, 1000 / max(hoop_w, hoop_h))
    W, H = int(hoop_w * px_per_mm), int(hoop_h * px_per_mm)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = C.hex_to_rgb(fabric_hex)[::-1]
    x0, y0, x1, y1 = bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def to_px(x, y):
        return (int(round(((x - cx) / 10 + hoop_w / 2) * px_per_mm)),
                int(round(((y - cy) / 10 + hoop_h / 2) * px_per_mm)))

    grid = (150, 150, 150)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), grid, 1)
    mx, my = W // 2, H // 2
    cv2.line(img, (mx - 8, my), (mx + 8, my), grid, 1)
    cv2.line(img, (mx, my - 8), (mx, my + 8), grid, 1)
    # formats without colors (DST) cycle through a fixed set so blocks stay distinguishable
    cycle = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]

    color, prev = 0, None
    for x, y, c in stitches:
        cmd = c & pe.COMMAND_MASK
        if cmd == pe.STITCH:
            if prev is not None:
                hx = colors[color] if color < len(colors) else cycle[color % len(cycle)]
                bgr = C.hex_to_rgb(hx)[::-1]
                cv2.line(img, to_px(*prev), to_px(x, y), bgr, 1, cv2.LINE_AA)
            prev = (x, y)
        elif cmd in (pe.JUMP, pe.TRIM):
            prev = None
        elif cmd == pe.COLOR_CHANGE:
            color, prev = color + 1, None

    red = (40, 30, 230)
    for n, t in enumerate(trims, 1):
        p, q = to_px(*t["from"]), to_px(*t["to"])
        dashes = max(2, int(math.dist(p, q) / 6))
        for i in range(0, dashes, 2):
            a = (p[0] + (q[0] - p[0]) * i // dashes, p[1] + (q[1] - p[1]) * i // dashes)
            b = (p[0] + (q[0] - p[0]) * (i + 1) // dashes, p[1] + (q[1] - p[1]) * (i + 1) // dashes)
            cv2.line(img, a, b, red, 2, cv2.LINE_AA)
        mid = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
        cv2.circle(img, mid, 10, red, -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(img, str(n), (mid[0] - tw // 2, mid[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------- background jobs

_JOBS: dict[str, dict] = {}


def start_export(svg_text: str, fmt: str, thread_hexes: list[str], fabric_hex: str, hoop: tuple[float, float],
                 speed: float, filename: str) -> str:
    jid = uuid.uuid4().hex
    _JOBS[jid] = {"status": "running", "filename": filename, "format": fmt}

    def run():
        try:
            data = export_file(svg_text, fmt)
            report = analyze_file(data, fmt, len(thread_hexes), fabric_hex, hoop, speed, thread_hexes)
            _JOBS[jid].update(status="done", data=data, report=report)
        except Exception as e:  # surfaced to the UI
            _JOBS[jid].update(status="error", error=str(e))

    threading.Thread(target=run, daemon=True).start()
    while len(_JOBS) > 20:
        _JOBS.pop(next(iter(_JOBS)))
    return jid


def job(jid: str) -> dict:
    if jid not in _JOBS:
        raise KeyError("Unknown export job.")
    return _JOBS[jid]
