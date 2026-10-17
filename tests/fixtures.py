"""Synthetic test inputs: a flat 3-color logo and a messy traced SVG."""
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
    downsampled so the edges are anti-aliased like a real exported logo."""
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
    a stroked outline, a hidden layer and an embedded bitmap."""
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
    interior is part of the design."""
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
