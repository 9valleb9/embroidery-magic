"""Home embroidery machines: file formats by brand, standard hoop sizes, and the
checks that depend on them (fit, canvas orientation)."""
from __future__ import annotations

# Ink/Stitch writes all of these (through pyembroidery). `machines` is shown in the UI.
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
MAX_HOOP_MM = 400.0
DEFAULT_SPEED_SPM = 400   # Brother SE400; most home machines are 400–850


def fits(w: float, h: float, hoop_w: float, hoop_h: float) -> bool:
    """A design fits if it fits either way round (hoops can be used rotated)."""
    eps = 0.05
    return (w <= hoop_w + eps and h <= hoop_h + eps) or (w <= hoop_h + eps and h <= hoop_w + eps)


def oriented(w: float, h: float, hoop_w: float, hoop_h: float) -> tuple[float, float]:
    """Hoop dimensions turned to match the design: landscape design, landscape hoop."""
    if (w > h) != (hoop_w > hoop_h) and hoop_w != hoop_h and fits(w, h, hoop_h, hoop_w):
        return hoop_h, hoop_w
    return hoop_w, hoop_h


def fmt(code: str | None) -> str:
    code = (code or DEFAULT_FORMAT).lower()
    if code not in FORMATS:
        raise ValueError(f"Unsupported format: {code}")
    return code


def hoop_label(w: float, h: float) -> str:
    return f"{w:g} × {h:g} mm"
