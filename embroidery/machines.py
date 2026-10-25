"""Home embroidery machines: file formats by brand, standard hoop sizes, and the
checks that depend on them (fit, canvas orientation).

Where it sits: pure data plus small helpers, no I/O. ``app.py`` serves the formats and
hoops to the UI (``/api/machine``) and validates requests with :func:`fits`,
:func:`max_design`, :data:`MAX_HOOP_MM` and :data:`MAX_PULL_COMP_MM`. ``pipeline`` turns
the hoop to match the design with :func:`oriented` (the SVG canvas is the hoop), and
``export`` uses :data:`FORMATS` to pick the Ink/Stitch output format, check the file
signature and decide whether the file carries thread colors.

Units: all sizes are millimetres. Hoop sizes are the usable stitch field, given
portrait (width <= height); a hoop can be used either way round.

Main entry points: :func:`fmt`, :func:`fits`, :func:`max_design`, :func:`oriented`,
:func:`hoop_label`.
"""
from __future__ import annotations

# Ink/Stitch writes all of these (through pyembroidery). `machines` is shown in the UI.
# `signature`: magic bytes the exported file must start with (None = not checked).
# `palette`: the brand thread palette the file stores colors as (None = the file's colors
# aren't meaningful thread names, e.g. .dst stores no colors at all).
FORMATS = {
    "pes": {"label": "Brother / Baby Lock (.pes)", "machines": "Brother SE400, SE600, PE800, SE1900, Baby Lock…",
            "signature": b"#PES", "palette": "Brother"},
    "jef": {"label": "Janome / Elna (.jef)", "machines": "Janome Memory Craft, Elna eXpressive…",
            "signature": None, "palette": "Janome"},
    "vp3": {"label": "Husqvarna Viking / Pfaff (.vp3)", "machines": "Husqvarna Viking, Pfaff creative…",
            "signature": None, "palette": None},
    "exp": {"label": "Bernina (.exp)", "machines": "Bernina with embroidery module",
            "signature": None, "palette": None},
    "xxx": {"label": "Singer (.xxx)", "machines": "Singer Futura, SE9180…",
            "signature": None, "palette": None},
    "dst": {"label": "Tajima (.dst, most commercial machines)", "machines": "Commercial multi-needle machines",
            "signature": b"LA:", "palette": None},
}
DEFAULT_FORMAT = "pes"

# Usable stitch fields of the standard hoop sizes, in mm (width × height, portrait).
HOOPS = {
    "4x4": {"label": "4×4 in (100 × 100 mm)", "w": 100.0, "h": 100.0},
    "5x7": {"label": "5×7 in (130 × 180 mm)", "w": 130.0, "h": 180.0},
    "6x10": {"label": "6×10 in (160 × 260 mm)", "w": 160.0, "h": 260.0},
    "8x8": {"label": "8×8 in (200 × 200 mm)", "w": 200.0, "h": 200.0},
}
DEFAULT_HOOP = "4x4"
MAX_HOOP_MM = 400.0       # largest custom hoop side accepted by the API
DEFAULT_SPEED_SPM = 400   # Brother SE400; most home machines are 400–850
EDGE_MARGIN_MM = 0.5      # kept free at each hoop edge: pull compensation stitches past the design's edge
MAX_PULL_COMP_MM = EDGE_MARGIN_MM   # so the largest pull compensation still stays inside the hoop


def fits(w: float, h: float, hoop_w: float, hoop_h: float) -> bool:
    """A design fits if it fits either way round (hoops can be used rotated).

    Args:
        w, h: Design size in mm.
        hoop_w, hoop_h: Hoop (or max design) size in mm.

    Returns:
        True if the design fits, allowing 0.05 mm of rounding slack.
    """
    eps = 0.05
    return (w <= hoop_w + eps and h <= hoop_h + eps) or (w <= hoop_h + eps and h <= hoop_w + eps)


def max_design(hoop_w: float, hoop_h: float) -> tuple[float, float]:
    """Largest design for a hoop, leaving the edge margin for pull compensation.

    Returns:
        ``(w, h)`` in mm, e.g. ``(99.0, 99.0)`` for the 4x4 hoop.
    """
    return hoop_w - 2 * EDGE_MARGIN_MM, hoop_h - 2 * EDGE_MARGIN_MM


def oriented(w: float, h: float, hoop_w: float, hoop_h: float) -> tuple[float, float]:
    """Hoop dimensions turned to match the design: landscape design, landscape hoop.

    The hoop is only swapped when the design and hoop orientations differ, the hoop
    isn't square, and the design fits the swapped hoop.

    Args:
        w, h: Design size in mm.
        hoop_w, hoop_h: Hoop size in mm, as configured.

    Returns:
        ``(hoop_w, hoop_h)``, possibly swapped. Used as the SVG canvas size.
    """
    if (w > h) != (hoop_w > hoop_h) and hoop_w != hoop_h and fits(w, h, hoop_h, hoop_w):
        return hoop_h, hoop_w
    return hoop_w, hoop_h


def fmt(code: str | None) -> str:
    """Normalize a format code (``None`` means :data:`DEFAULT_FORMAT`, case-insensitive).

    Returns:
        The lowercase key into :data:`FORMATS`, also the Ink/Stitch ``--format`` value
        and the file extension.

    Raises:
        ValueError: If the format isn't in :data:`FORMATS`.
    """
    code = (code or DEFAULT_FORMAT).lower()
    if code not in FORMATS:
        raise ValueError(f"Unsupported format: {code}")
    return code


def hoop_label(w: float, h: float) -> str:
    """Human-readable size for messages, e.g. ``"100 × 100 mm"``."""
    return f"{w:g} × {h:g} mm"
