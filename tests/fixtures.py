"""Synthetic test inputs: a flat 3-color logo and a messy traced SVG.

Every image is generated in code (no binary files in the repo), so the inputs
are deterministic and each one targets specific pipeline behavior. The PNG
fixtures are drawn at 4x and downsampled with Lanczos so their edges are
anti-aliased like real exported art, which exercises palette clustering and
sliver cleanup. Sizes quoted below assume the test's output size (about
100 mm wide).

    flat_logo_png      clean 3-color logo on white: the happy path
    messy_traced_svg   an auto-trace full of junk: SVG cleanup
    ringed_badge_png   white margin plus white interior: background removal
    thin_outline_png   0.6 mm ring on transparent: thin lines and hairlines
    script_stroke_png  3 mm stroke plus a disc: satin columns

These are helpers, not tests; tests/test_pipeline.py and tests/test_export.py
import them.
"""
from __future__ import annotations

import base64
import io
import math
import random

from PIL import Image, ImageDraw

DARK_GREEN = "#1b5e20"
LIGHT_GREEN = "#8bc34a"


def flat_logo_png(size: int = 600) -> bytes:
    """White background, dark green disc, light green ring + bar. Drawn at 4x and
    downsampled so the edges are anti-aliased like a real exported logo.

    Concentric circles (dark green, light green ring, dark green center) with a
    white bar across the middle. Detects as 3 colors (white, DARK_GREEN,
    LIGHT_GREEN); on white fabric the white is knocked out, leaving two thread
    layers whose colors touch. Used for the basic pipeline path, overlap and
    pull compensation, knockout and restore on a fabric change, low-contrast
    warnings and the machine summary.

    Args:
        size: output width and height in pixels.

    Returns:
        PNG bytes.
    """
    s = size * 4
    img = Image.new("RGB", (s, s), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([s * 0.12, s * 0.12, s * 0.88, s * 0.88], fill=DARK_GREEN)
    d.ellipse([s * 0.25, s * 0.25, s * 0.75, s * 0.75], fill=LIGHT_GREEN)
    d.ellipse([s * 0.33, s * 0.33, s * 0.67, s * 0.67], fill=DARK_GREEN)
    d.polygon([(s * 0.30, s * 0.47), (s * 0.70, s * 0.40), (s * 0.70, s * 0.53), (s * 0.30, s * 0.60)],
              fill="white")
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def messy_traced_svg(specks: int = 400, seed: int = 1) -> bytes:
    """Like an auto-trace: overlapping near-duplicate fills, hundreds of specks,
    a stroked outline, a hidden layer and an embedded bitmap.

    On a 200 x 200 white canvas: an embedded PNG, a display:none red group and
    an opacity 0 blue rect (none of which may reach the palette), a navy badge
    traced as two near-identical navies (#1f2a44 and #202b46, which should
    merge), a red stripe drawn only as a stroke (which must become a filled
    shape), a gold star, and many tiny specks (design colors, white and near
    variants) that cleanup must remove. The white margin is detected as
    background. Expected result: 3 thread colors (navy, red, gold).

    Args:
        specks: number of speck circles to scatter.
        seed: random seed, so the output is identical on every run.

    Returns:
        SVG bytes.
    """
    rnd = random.Random(seed)
    tiny_png = base64.b64encode(flat_logo_png(16)).decode()
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'width="200" height="200" viewBox="0 0 200 200">',
        '<rect width="200" height="200" fill="#ffffff"/>',
        # embedded bitmap and hidden layer — must not appear in the output
        f'<image x="0" y="0" width="200" height="200" xlink:href="data:image/png;base64,{tiny_png}"/>',
        '<g style="display:none"><rect x="10" y="10" width="180" height="180" fill="#ff0000"/></g>',
        '<rect x="20" y="20" width="60" height="60" fill="#0000ff" opacity="0"/>',
    ]
    # main navy badge, traced as two overlapping near-identical colors
    parts.append('<path fill="#1f2a44" d="M100 18 L178 60 L178 140 L100 182 L22 140 L22 60 Z"/>')
    parts.append('<path fill="#202b46" d="M100 30 L166 66 L166 100 L34 100 L34 66 Z"/>')
    # red stripe drawn as a stroke only (must become a filled shape)
    parts.append('<path d="M45 120 L155 120" stroke="#c62828" stroke-width="14" fill="none"/>')
    # gold star-ish shape
    pts = []
    for i in range(10):
        r = 34 if i % 2 == 0 else 15
        a = math.pi / 2 + i * math.pi / 5
        pts.append(f"{100 + r * math.cos(a):.2f},{78 - r * math.sin(a):.2f}")
    parts.append(f'<polygon fill="#e0b030" points="{" ".join(pts)}"/>')
    # specks in every color, all well under 1 mm² at 100 mm output
    colors = ["#1f2a44", "#c62828", "#e0b030", "#ffffff", "#203050", "#c02020"]
    for _ in range(specks):
        x, y, r = rnd.uniform(0, 200), rnd.uniform(0, 200), rnd.uniform(0.2, 0.9)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{rnd.choice(colors)}"/>')
    parts.append("</svg>")
    return "\n".join(parts).encode()


def ringed_badge_png(size: int = 600) -> bytes:
    """Crab-like case: a white margin around a black-ringed badge whose interior is
    also white, with a gold square inside. The margin is background; the white
    interior is part of the design.

    Exercises background removal: only the white region connected to the image
    edge (found by flood fill) is removed and cropped away, while the same
    white inside the black ring stays a thread color, e.g. to stitch on navy.

    Args:
        size: output width and height in pixels.

    Returns:
        PNG bytes.
    """
    s = size * 4
    img = Image.new("RGB", (s, s), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([s * 0.25, s * 0.25, s * 0.75, s * 0.75], fill="black")
    d.ellipse([s * 0.29, s * 0.29, s * 0.71, s * 0.71], fill="white")
    d.rectangle([s * 0.42, s * 0.42, s * 0.58, s * 0.58], fill="#e0b030")
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def thin_outline_png(size: int = 800) -> bytes:
    """A gold disc with a ~0.6 mm black outline ring around it (at 100 mm), on transparent.

    The ring is too thin to fill, so with thin_lines on it should become a
    triple running stitch, with thin_lines off it should raise a narrow_stitch
    warning, and in satin mode it should become a satin column widened to
    1 mm. A 4 mm black bar across the middle is wide enough to stay a fill.
    The transparent background also covers alpha handling.

    Args:
        size: output width and height in pixels.

    Returns:
        PNG bytes (RGBA).
    """
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([s * 0.02, s * 0.02, s * 0.98, s * 0.98], fill="black")
    d.ellipse([s * 0.026, s * 0.026, s * 0.974, s * 0.974], fill="#e0b030")
    d.rectangle([s * 0.3, s * 0.48, s * 0.7, s * 0.52], fill="black")      # a 4 mm bar: stays a fill
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def script_stroke_png(size: int = 800) -> bytes:
    """A ~3 mm wide navy S-curve stroke (at 99 mm wide) and a solid 20 mm disc, on white.

    Stands in for script lettering: the steady-width stroke should become a
    satin column when satin is on (and makes /api/analyze recommend satin),
    while the disc is too wide for satin and stays a fill. The image is twice
    as wide as it is tall.

    Args:
        size: output width in pixels (the height is size // 2).

    Returns:
        PNG bytes.
    """
    s = size * 4
    img = Image.new("RGB", (s, s // 2), "white")
    d = ImageDraw.Draw(img)
    pts = [(s * (0.05 + 0.6 * t / 100), s * (0.25 + 0.15 * math.sin(t / 100 * 2 * math.pi))) for t in range(101)]
    d.line(pts, fill="#1f2a44", width=int(s * 0.03), joint="curve")
    d.ellipse([s * 0.72, s * 0.15, s * 0.92, s * 0.35], fill="#1f2a44")
    img = img.resize((size, size // 2), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
