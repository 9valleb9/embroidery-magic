"""Machine file export (.pes, .jef, .vp3, .exp, .xxx, .dst) through the Ink/Stitch CLI,
then verification of the written file with pyembroidery (bounds, colors, stitch
count, hand trims).

Where it sits: the last step. ``pipeline.generate`` produces the SVG (``svg_out``);
``app.py`` passes it to :func:`start_export`, the UI polls :func:`job` until the export
is done, then downloads the file and shows the report from :func:`analyze_file`.
``pipeline`` also imports :data:`STITCH_WARN` and :data:`EST_STITCHES_PER_MM2` for its
pre-export stitch estimate.

Ink/Stitch invocation: :func:`export_file` runs
``inkstitch --extension=output --format=<fmt> design.svg``. Ink/Stitch writes the
machine file to stdout. The binary
is found by :func:`find_inkstitch` (``INKSTITCH_BIN``, the standard Inkscape extension
folders, then ``PATH``). On Linux it needs a display, so ``INKSTITCH_BIN`` usually points
at an ``xvfb-run`` wrapper. An export takes roughly 20 to 60 seconds.

Background job model: :func:`start_export` runs export plus analysis on a daemon
thread and returns a job id at once. Jobs live in the in-memory :data:`_JOBS` dict
(status ``"running"``, ``"done"`` or ``"error"``), so they are lost on restart and are
not shared between worker processes. Only the 20 most recent jobs are kept.

Units: pyembroidery coordinates are in 0.1 mm, so every distance read from the pattern
is divided by 10. Positions reported to the UI (trims) are in mm from the hoop's
top-left corner, like the SVG, assuming the design is centered on the hoop.

Main entry points: :func:`find_inkstitch`, :func:`export_file`, :func:`analyze_file`,
:func:`start_export`, :func:`job`.
"""
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

STITCH_WARN = 25_000      # above this many stitches, warn that the run is long
MAX_STITCH_MM = 12.1      # DST's per-stitch limit; Brother hardware allows 12.7 mm
EXPORT_TIMEOUT_S = 600    # hard limit on one Ink/Stitch run, in seconds
# Ink/Stitch defaults (0.25 mm rows, 3 mm stitches, underlay) come out near this
# density; used only for the estimate shown before the real export finishes.
EST_STITCHES_PER_MM2 = 1.9

# Where Ink/Stitch installs itself as an Inkscape extension (macOS user, macOS app
# bundle, Linux user). Globs and ~ are expanded.
_CANDIDATES = [
    "~/Library/Application Support/org.inkscape.Inkscape/config/inkscape/extensions/inkstitch/inkstitch.app/Contents/MacOS/inkstitch",
    "/Applications/Inkscape.app/Contents/Resources/share/inkscape/extensions/inkstitch/inkstitch.app/Contents/MacOS/inkstitch",
    "~/.config/inkscape/extensions/inkstitch/bin/inkstitch",
]


def find_inkstitch() -> str | None:
    """Locate the Ink/Stitch executable.

    ``INKSTITCH_BIN`` wins if set: it is used only if executable, with no fallback.
    Otherwise the first executable match in :data:`_CANDIDATES`, then ``inkstitch`` on
    ``PATH``.

    Returns:
        The path, or None if Ink/Stitch isn't installed (the UI then shows
        :data:`MANUAL_STEPS` instead of an export button).
    """
    env = os.environ.get("INKSTITCH_BIN")
    if env:
        return env if os.access(env, os.X_OK) else None
    for c in _CANDIDATES:
        for p in glob.glob(os.path.expanduser(c)):
            if os.access(p, os.X_OK):
                return p
    return shutil.which("inkstitch")


# Shown in the UI when Ink/Stitch isn't installed: how to export the SVG by hand.
MANUAL_STEPS = [
    "Open the downloaded SVG in Inkscape (with Ink/Stitch installed).",
    "Extensions › Ink/Stitch › Visualize and Export › Simulator — check the stitch-out.",
    "File › Save a Copy… and choose your machine's Ink/Stitch format (e.g. “Brother Embroidery Format (.pes)”).",
    "Copy the file to the machine over USB (most show up as a removable disk).",
]


def export_file(svg_text: str, fmt: str = M.DEFAULT_FORMAT) -> bytes:
    """Convert an Ink/Stitch SVG to a machine file by running the Ink/Stitch CLI.

    Blocks until Ink/Stitch finishes (up to :data:`EXPORT_TIMEOUT_S`). The SVG is written
    to a temporary directory, which is also the working directory, and removed after.

    Args:
        svg_text: The SVG from ``svg_out.write_svg``.
        fmt: Format code (see ``machines.FORMATS``).

    Returns:
        The machine file's bytes (Ink/Stitch's stdout).

    Raises:
        RuntimeError: If Ink/Stitch isn't found, or exits non-zero or with no output
            (the message carries the last 5 lines of its stderr).
        subprocess.TimeoutExpired: If the run exceeds the timeout.
        ValueError: If ``fmt`` is unsupported.
    """
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
    """The command (STITCH, JUMP, TRIM, COLOR_CHANGE, END...) of a pyembroidery ``[x, y, cmd]`` entry, flags masked off."""
    return s[2] & pe.COMMAND_MASK


def analyze_file(data: bytes, fmt: str = M.DEFAULT_FORMAT, expected_colors: int | None = None,
                 fabric_hex: str = "#ffffff", hoop: tuple[float, float] = (100.0, 100.0),
                 speed: float = M.DEFAULT_SPEED_SPM, thread_hexes: list[str] | None = None) -> dict:
    """Read a machine file back and check it against the hoop and machine limits.
    thread_hexes: your actual threads in stitch order, used to draw the stitch-out
    (the file itself may only hold the nearest palette colors, or none).

    Checks, each adding a message to ``problems``:

    - the file starts with the format's signature (.pes, .dst only);
    - the stitch extent fits the hoop (either way round);
    - the longest stitch is at most :data:`MAX_STITCH_MM`;
    - the number of color blocks equals ``expected_colors`` (each thread in one block).

    A "trim" is a jump (or trim command) inside one color block longer than 0.5 mm: home
    machines like the SE400 don't cut these, so the user snips them by hand.

    Args:
        data: The machine file bytes.
        fmt: Format code (see ``machines.FORMATS``).
        expected_colors: Number of stitched threads, or None to skip that check.
        fabric_hex: Background color of the preview.
        hoop: ``(w, h)`` of the (oriented) hoop in mm.
        speed: Machine speed in stitches per minute, for the time estimate.
        thread_hexes: See above. Ignored unless it has one entry per color block.

    Returns:
        A report dict. ``ok`` and ``problems`` are always present; if pyembroidery can't
        read the file or it has no stitches, those are the only keys. Otherwise it also
        has the stitch count, color blocks, per-thread info (``machine_name`` is what the
        machine's screen shows, from the brand palette), the numbered ``trims`` (with
        ``from_mm``/``to_mm`` hoop positions), the time estimate in ``minutes`` and
        ``preview_png`` (a PNG data URL from :func:`_render`).
    """
    fmt = M.fmt(fmt)
    hoop_w, hoop_h = hoop
    problems: list[str] = []
    sig = M.FORMATS[fmt]["signature"]
    if sig and not data.startswith(sig):
        problems.append(f"File doesn't start with the {sig.decode()} signature.")
    # pyembroidery picks its reader from the file extension, so it needs a real file.
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
    blocks = [b for b in blocks if b]   # color changes with no stitches (e.g. a leading one) aren't blocks
    if longest > MAX_STITCH_MM:
        problems.append(f"Longest stitch is {longest:.1f} mm (limit {MAX_STITCH_MM} mm).")
    if expected_colors is not None and len(blocks) != expected_colors:
        problems.append(f"Expected {expected_colors} color blocks, found {len(blocks)}.")

    total = int(len(pts))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2   # stitch-extent center, mapped to the hoop center
    thread_info = []
    for i, n in enumerate(blocks):
        t = threads[i] if i < len(threads) else None
        # Only palette formats (.pes, .jef) store a brand thread the machine's screen shows,
        # so other formats report no machine name or hex.
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
    """Pick the colors to draw each color block in.

    Prefers the user's real threads (when there is exactly one per block), else the
    file's palette colors for palette formats, else an empty list (so :func:`_render`
    falls back to its fixed color cycle).

    Args:
        threads: pyembroidery ``pattern.threadlist``.
        fmt: Normalized format code.
        n_blocks: Number of non-empty color blocks in the file.
        thread_hexes: The user's thread colors in stitch order, or None.

    Returns:
        Hex colors, indexed by block.
    """
    if thread_hexes and len(thread_hexes) == n_blocks:
        return list(thread_hexes)
    if M.FORMATS[fmt]["palette"]:
        return [t.hex_color() for t in threads]
    return []


def _render(stitches, colors, bounds, fabric_hex, trims, hoop, px_per_mm: float = 8.0) -> str:
    """Stitch-out preview on the hoop: thread lines in their PES colors, and the
    in-color jumps you'll trim by hand drawn dashed red and numbered.

    The stitch extent is centered on the hoop, matching how ``pipeline`` places the
    design on the SVG canvas. The image also shows the hoop outline and a center cross.

    Args:
        stitches: pyembroidery ``[x, y, cmd]`` entries in 0.1 mm.
        colors: Hex color per color block (see :func:`_draw_colors`); blocks past the
            end of the list use a fixed color cycle.
        bounds: ``(x0, y0, x1, y1)`` of the stitch points, in 0.1 mm.
        fabric_hex: Background color.
        trims: Trim dicts from :func:`analyze_file`, with raw ``from``/``to`` points.
        hoop: ``(w, h)`` in mm.
        px_per_mm: Requested scale, reduced so the longer side is at most 1000 px.

    Returns:
        A ``data:image/png;base64,...`` URL.
    """
    hoop_w, hoop_h = hoop
    px_per_mm = min(px_per_mm, 1000 / max(hoop_w, hoop_h))
    W, H = int(hoop_w * px_per_mm), int(hoop_h * px_per_mm)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = C.hex_to_rgb(fabric_hex)[::-1]   # OpenCV images are BGR
    x0, y0, x1, y1 = bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def to_px(x, y):
        """Map a pyembroidery point (0.1 mm) to image pixels, with the design centered on the hoop."""
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
        mid = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)   # the trim number sits on a red dot at the jump's midpoint
        cv2.circle(img, mid, 10, red, -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(img, str(n), (mid[0] - tw // 2, mid[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------- background jobs

# job id -> {status, filename, format, and data + report when done, or error}. In memory,
# per process; insertion order doubles as age for pruning.
_JOBS: dict[str, dict] = {}


def start_export(svg_text: str, fmt: str, thread_hexes: list[str], fabric_hex: str, hoop: tuple[float, float],
                 speed: float, filename: str) -> str:
    """Start a machine-file export plus read-back analysis on a background thread.

    Side effects: adds a job to :data:`_JOBS` and starts a daemon thread that runs
    :func:`export_file` then :func:`analyze_file`, storing ``data`` and ``report`` (or
    ``error``) on the job. Drops the oldest jobs beyond 20.

    Args:
        svg_text: The SVG from ``svg_out.write_svg``.
        fmt: Format code.
        thread_hexes: The user's stitched threads in order; its length is the expected
            number of color blocks.
        fabric_hex: Fabric color for the preview.
        hoop: ``(w, h)`` of the oriented hoop in mm.
        speed: Machine speed in stitches per minute.
        filename: Download name stored on the job.

    Returns:
        The job id (32 hex characters) to poll with :func:`job`.
    """
    jid = uuid.uuid4().hex
    _JOBS[jid] = {"status": "running", "filename": filename, "format": fmt}

    def run():
        """Thread body: export, analyze, and record the outcome on the job."""
        try:
            data = export_file(svg_text, fmt)
            report = analyze_file(data, fmt, len(thread_hexes), fabric_hex, hoop, speed, thread_hexes)
            _JOBS[jid].update(status="done", data=data, report=report)
        except Exception as e:  # surfaced to the UI
            _JOBS[jid].update(status="error", error=str(e))

    threading.Thread(target=run, daemon=True).start()
    while len(_JOBS) > 20:   # dicts keep insertion order, so this drops the oldest jobs (finished or not)
        _JOBS.pop(next(iter(_JOBS)))
    return jid


def job(jid: str) -> dict:
    """Return the live job dict for an id (``status`` is ``"running"``, ``"done"`` or ``"error"``).

    The dict is shared with the worker thread, not a copy. When done it holds the file
    bytes under ``data`` and the :func:`analyze_file` report under ``report``.

    Raises:
        KeyError: If the id is unknown or the job was pruned.
    """
    if jid not in _JOBS:
        raise KeyError("Unknown export job.")
    return _JOBS[jid]
