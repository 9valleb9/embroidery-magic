"""Step 1 — rasterize PNG/JPG/SVG input at 20 px per mm of final size."""
from __future__ import annotations

import io
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageOps

PX_PER_MM = 20

SVG_NS = "http://www.w3.org/2000/svg"
for _prefix, _uri in {
    "": SVG_NS,
    "xlink": "http://www.w3.org/1999/xlink",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
}.items():
    ET.register_namespace(_prefix, _uri)

_HIDDEN_STYLE = re.compile(
    r"(display\s*:\s*none|visibility\s*:\s*(hidden|collapse)|(?<![-\w])opacity\s*:\s*0*(\.0*)?\s*(;|$))"
)
_DROP_TAGS = {"image", "script", "foreignObject"}


def _ensure_cairo_on_path() -> None:
    """Homebrew's libcairo isn't on the default dyld path on Apple Silicon."""
    extra = [p for p in ("/opt/homebrew/lib", "/usr/local/lib") if os.path.isdir(p)]
    cur = [p for p in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":") if p]
    missing = [p for p in extra if p not in cur]
    if missing:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(cur + missing)


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _is_hidden(el: ET.Element) -> bool:
    if el.get("display") == "none" or el.get("visibility") in ("hidden", "collapse"):
        return True
    op = el.get("opacity")
    if op is not None:
        try:
            if float(op.strip().rstrip("%")) == 0:
                return True
        except ValueError:
            pass
    return bool(_HIDDEN_STYLE.search(el.get("style", "")))


def sanitize_svg(data: bytes) -> bytes:
    """Remove hidden elements and embedded/linked <image> elements before rendering."""
    root = ET.fromstring(data)

    def walk(parent: ET.Element) -> None:
        for child in list(parent):
            if _local(child.tag) in _DROP_TAGS or _is_hidden(child):
                parent.remove(child)
            else:
                walk(child)

    walk(root)
    return ET.tostring(root)


def is_svg(data: bytes, filename: str) -> bool:
    if filename.lower().endswith((".svg", ".svgz")):
        return True
    head = data[:512].lstrip().lower()
    return head.startswith(b"<?xml") and b"<svg" in head or head.startswith(b"<svg")


def rasterize(data: bytes, filename: str, box_w_mm: float, box_h_mm: float):
    """Render the input to fit inside box_w x box_h mm (aspect kept) at PX_PER_MM.

    Returns (rgb uint8 HxWx3, alpha uint8 HxW, width_mm, height_mm).
    Strokes in SVG input are rendered as pixels and so come out of vectorization
    as filled outlines.
    """
    if box_w_mm <= 0 or box_h_mm <= 0:
        raise ValueError("size must be positive")
    svg = is_svg(data, filename)
    if svg:
        _ensure_cairo_on_path()
        import cairosvg

        clean = sanitize_svg(data)
        probe = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=clean)))
        w0, h0 = probe.size
    else:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGBA")
        w0, h0 = img.size

    scale = min(box_w_mm / w0, box_h_mm / h0)
    W = max(1, round(w0 * scale * PX_PER_MM))
    H = max(1, round(h0 * scale * PX_PER_MM))

    if svg:
        png = cairosvg.svg2png(bytestring=clean, output_width=W, output_height=H)
        img = Image.open(io.BytesIO(png)).convert("RGBA")
        if img.size != (W, H):
            img = img.resize((W, H), Image.LANCZOS)
    else:
        img = img.resize((W, H), Image.LANCZOS)

    arr = np.asarray(img)
    return arr[..., :3].copy(), arr[..., 3].copy(), W / PX_PER_MM, H / PX_PER_MM
