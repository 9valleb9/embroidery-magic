"""Step 1 — rasterize PNG/JPG/SVG input at 20 px per mm of final size.

Where this sits in the flow:

    upload -> raster.py (rasterize) -> palette.py (colors, label map)
    -> vectorize.py (shapes, stitch settings) -> pipeline.py (sessions, color
    decisions, generation) -> svg_out.py (Ink/Stitch SVG) -> export.py (machine file)

This module turns whatever the user uploaded into a plain RGBA pixel grid at a
fixed resolution of ``PX_PER_MM`` pixels per millimetre of *final stitched
size*. Every later step works on that grid (palette.py's label map,
vectorize.py's masks), so a pixel always means 1 / PX_PER_MM mm (0.05 mm) and
mm <-> px conversions elsewhere just multiply or divide by ``PX_PER_MM``.

Main entry point:
    ``rasterize(data, filename, box_w_mm, box_h_mm, crop=None)``: render the
    upload to fit a box in mm, returning RGB, alpha and the real size in mm.

Helpers used on the way:
    ``is_svg`` sniffs the format, ``sanitize_svg`` strips content that must not
    be rendered (hidden elements, ``<image>``, ``<script>``, ``<foreignObject>``)
    and ``_ensure_cairo_on_path`` lets cairosvg find Homebrew's libcairo.

Gotchas:
    * SVG strokes are rasterized like any other paint, so a stroked outline
      reaches vectorization as a thin filled band, not as a path.
    * Raster input is EXIF-rotated before anything else, so phone photos come
      out upright.
    * cairosvg is imported lazily, so PNG/JPG work without the Cairo library.
    * A crop on SVG input renders the whole image larger and cuts the region
      out (capped at ``MAX_RENDER_PX``), so the cropped part keeps full detail.
"""
from __future__ import annotations

import io
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageOps

PX_PER_MM = 20            # resolution of every raster in the app: 20 px = 1 mm of stitched size

SVG_NS = "http://www.w3.org/2000/svg"
# Register the usual prefixes so the SVG that sanitize_svg re-serializes keeps
# them (default svg namespace, xlink:, inkscape:) instead of ns0:, ns1:, ...
for _prefix, _uri in {
    "": SVG_NS,
    "xlink": "http://www.w3.org/1999/xlink",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
}.items():
    ET.register_namespace(_prefix, _uri)

# Inline style that hides an element. The lookbehind keeps fill-opacity /
# stroke-opacity from matching; only a plain opacity of zero ("0", "0.0", ".0") counts.
_HIDDEN_STYLE = re.compile(
    r"(display\s*:\s*none|visibility\s*:\s*(hidden|collapse)|(?<![-\w])opacity\s*:\s*0*(\.0*)?\s*(;|$))"
)
_DROP_TAGS = {"image", "script", "foreignObject"}   # elements removed (with their children) before rendering


def _ensure_cairo_on_path() -> None:
    """Homebrew's libcairo isn't on the default dyld path on Apple Silicon.

    Appends ``/opt/homebrew/lib`` and ``/usr/local/lib`` (those that exist) to
    ``DYLD_FALLBACK_LIBRARY_PATH`` so the cairosvg import that follows can load
    the library.

    Side effects:
        Mutates ``os.environ`` for the current process. Paths already listed
        are not added twice.
    """
    extra = [p for p in ("/opt/homebrew/lib", "/usr/local/lib") if os.path.isdir(p)]
    cur = [p for p in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":") if p]
    missing = [p for p in extra if p not in cur]
    if missing:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(cur + missing)


def _local(tag) -> str:
    """Strip the ``{namespace}`` from an ElementTree tag (comments and PIs have no str tag: returns "")."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _is_hidden(el: ET.Element) -> bool:
    """True when the element itself is invisible.

    Checks the ``display="none"``, ``visibility="hidden|collapse"`` and
    ``opacity="0"`` (also ``0%``) attributes, then the same rules in its inline
    ``style`` via ``_HIDDEN_STYLE``. Inherited visibility and CSS stylesheets
    are not considered.
    """
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
    """Remove hidden elements and embedded/linked <image> elements before rendering.

    Also drops ``<script>`` and ``<foreignObject>`` (see ``_DROP_TAGS``). A
    removed element takes its whole subtree with it.

    Args:
        data: Raw SVG bytes.

    Returns:
        The cleaned SVG serialized back to bytes.

    Raises:
        xml.etree.ElementTree.ParseError: If ``data`` is not well-formed XML.
    """
    root = ET.fromstring(data)

    def walk(parent: ET.Element) -> None:
        """Depth-first removal of dropped or hidden children of ``parent`` (in place)."""
        for child in list(parent):
            if _local(child.tag) in _DROP_TAGS or _is_hidden(child):
                parent.remove(child)
            else:
                walk(child)

    walk(root)
    return ET.tostring(root)


def is_svg(data: bytes, filename: str) -> bool:
    """True if the upload is an SVG: by ``.svg``/``.svgz`` extension, or by an
    ``<?xml ... <svg`` or ``<svg`` start within the first 512 bytes."""
    if filename.lower().endswith((".svg", ".svgz")):
        return True
    head = data[:512].lstrip().lower()
    return head.startswith(b"<?xml") and b"<svg" in head or head.startswith(b"<svg")


MAX_RENDER_PX = 60_000_000   # cap for the full-image render behind a crop


def rasterize(data: bytes, filename: str, box_w_mm: float, box_h_mm: float, crop=None):
    """Render the input to fit inside box_w x box_h mm (aspect kept) at PX_PER_MM.

    crop: optional (x0, y0, x1, y1) as fractions of the image; that region is
    what gets fitted to the box (used after background removal).
    Returns (rgb uint8 HxWx3, alpha uint8 HxW, width_mm, height_mm).
    Strokes in SVG input are rendered as pixels and so come out of vectorization
    as filled outlines.

    Args:
        data: The uploaded file's bytes (PNG, JPG or anything Pillow opens, or SVG).
        filename: Original file name, used only to help detect SVG.
        box_w_mm: Maximum width of the stitched design in mm.
        box_h_mm: Maximum height of the stitched design in mm.
        crop: Optional ``(x0, y0, x1, y1)`` fractions of the full image (0..1).

    Returns:
        ``(rgb, alpha, width_mm, height_mm)``: ``rgb`` is an HxWx3 uint8 array,
        ``alpha`` an HxW uint8 array, and the sizes are the real fitted size
        (W / PX_PER_MM, H / PX_PER_MM), which is at most the box on each side.

    Raises:
        ValueError: If either box side is not positive.
    """
    if box_w_mm <= 0 or box_h_mm <= 0:
        raise ValueError("size must be positive")
    svg = is_svg(data, filename)
    if svg:
        _ensure_cairo_on_path()
        import cairosvg

        clean = sanitize_svg(data)
        # a render at the SVG's own size, only to learn its natural size and aspect
        probe = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=clean)))
        w0, h0 = probe.size
    else:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGBA")
        w0, h0 = img.size

    fx0, fy0, fx1, fy1 = crop or (0.0, 0.0, 1.0, 1.0)
    cw, ch = w0 * (fx1 - fx0), h0 * (fy1 - fy0)       # region size in natural px
    scale = min(box_w_mm / cw, box_h_mm / ch)         # mm per natural px
    W = max(1, round(cw * scale * PX_PER_MM))
    H = max(1, round(ch * scale * PX_PER_MM))

    if svg:
        # Render the whole image so the crop region comes out at W x H (capped), then cut it out.
        fw, fh = W / (fx1 - fx0), H / (fy1 - fy0)
        shrink = min(1.0, (MAX_RENDER_PX / (fw * fh)) ** 0.5)
        fw, fh = max(1, round(fw * shrink)), max(1, round(fh * shrink))
        png = cairosvg.svg2png(bytestring=clean, output_width=fw, output_height=fh)
        img = Image.open(io.BytesIO(png)).convert("RGBA")
        fw, fh = img.size
        img = img.crop((round(fx0 * fw), round(fy0 * fh), round(fx1 * fw), round(fy1 * fh)))
    elif crop:
        img = img.crop((round(fx0 * w0), round(fy0 * h0), round(fx1 * w0), round(fy1 * h0)))
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)

    arr = np.asarray(img)
    return arr[..., :3].copy(), arr[..., 3].copy(), W / PX_PER_MM, H / PX_PER_MM
