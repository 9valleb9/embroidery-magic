"""Color math: CIE Lab, ΔE2000, thread names, bobbin and contrast helpers."""
from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

MERGE_DE = 5.0          # near-identical design colors merge by default (fewer thread changes)
KNOCKOUT_DE = 8.0       # design color this close to the fabric -> left as fabric
LOW_CONTRAST_DE = 15.0  # stitched color this close to the fabric -> barely shows
LIGHT_THREAD_L = 70.0
DARK_FABRIC_L = 35.0

FABRIC_PRESETS = {
    "White": "#ffffff",
    "Black": "#141414",
    "Navy": "#1f2a44",
    "Gray": "#8a8d91",
    "Tan leather": "#b5835a",
    "Red": "#b3202a",
}

# Names used for thread layers ("1 Dark green"). Nearest by ΔE2000.
NAMED_COLORS = {
    "White": "#ffffff", "Cream": "#f3ead3", "Light gray": "#c8c8c8", "Gray": "#8a8a8a",
    "Dark gray": "#4a4a4a", "Black": "#111111", "Red": "#d32f2f", "Dark red": "#8b1a1a",
    "Burgundy": "#6d1a36", "Pink": "#f48fb1", "Hot pink": "#e91e63", "Orange": "#f57c00",
    "Peach": "#ffcc99", "Yellow": "#fdd835", "Gold": "#c9a227", "Tan": "#c8a27a",
    "Brown": "#6d4c41", "Dark brown": "#3e2723", "Light green": "#8bc34a", "Green": "#2e9e44",
    "Dark green": "#1b5e20", "Olive": "#6b6b1f", "Lime": "#c6e03b", "Mint": "#a8e6cf",
    "Teal": "#00897b", "Light blue": "#90caf9", "Sky blue": "#4fc3f7", "Blue": "#1e63d6",
    "Royal blue": "#2a3fa8", "Navy": "#1f2a44", "Purple": "#7b1fa2", "Lavender": "#c5b3e6",
    "Maroon": "#5a1020", "Silver": "#b8bcc2",
}


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"bad hex color: {h!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(max(0, min(255, c)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_lab(h: str) -> np.ndarray:
    rgb = np.array(hex_to_rgb(h), dtype=np.float64).reshape(1, 1, 3) / 255.0
    return rgb2lab(rgb).reshape(3)


def lab_to_hex(lab) -> str:
    rgb = lab2rgb(np.asarray(lab, dtype=np.float64).reshape(1, 1, 3)).reshape(3)
    return rgb_to_hex(rgb * 255)


def delta_e(h1: str, h2: str) -> float:
    return float(deltaE_ciede2000(hex_to_lab(h1), hex_to_lab(h2)))


def lightness(h: str) -> float:
    return float(hex_to_lab(h)[0])


_NAMED_LABS = {name: hex_to_lab(hx) for name, hx in NAMED_COLORS.items()}


def color_name(h: str) -> str:
    lab = hex_to_lab(h)
    return min(_NAMED_LABS, key=lambda n: float(deltaE_ciede2000(lab, _NAMED_LABS[n])))


def fabric_name(h: str) -> str:
    for name, hx in FABRIC_PRESETS.items():
        if hx.lower() == h.lower():
            return name
    return color_name(h)


def fabric_label(h: str) -> str:
    """For messages: the preset name, or the nearest name plus the hex for custom fabrics."""
    for name, hx in FABRIC_PRESETS.items():
        if hx.lower() == h.lower():
            return name
    return f"{color_name(h)} ({h.lower()})"


def bobbin_for(fabric_hex: str) -> tuple[str, str]:
    """White or black bobbin thread, whichever is closer to the fabric."""
    if delta_e(fabric_hex, "#ffffff") <= delta_e(fabric_hex, "#000000"):
        return "White", "#ffffff"
    return "Black", "#000000"


def contrasting_bw(fabric_hex: str) -> tuple[str, str]:
    """Black or white, whichever stands out more against the fabric."""
    if delta_e(fabric_hex, "#ffffff") >= delta_e(fabric_hex, "#000000"):
        return "White", "#ffffff"
    return "Black", "#111111"


def contrast_swap(thread_hex: str, fabric_hex: str, target_de: float = 25.0) -> str:
    """Suggest a replacement thread: same hue, lightness pushed away from the fabric."""
    lab = hex_to_lab(thread_hex)
    fab_l = lightness(fabric_hex)
    step = 4.0 if fab_l < 50 else -4.0
    cand = lab.copy()
    for _ in range(30):
        cand[0] = float(np.clip(cand[0] + step, 0, 100))
        hx = lab_to_hex(cand)
        if delta_e(hx, fabric_hex) >= target_de:
            return hx
        if cand[0] in (0.0, 100.0):
            break
    return contrasting_bw(fabric_hex)[1]


def is_light_on_dark(thread_hex: str, fabric_hex: str) -> bool:
    return lightness(thread_hex) > LIGHT_THREAD_L and lightness(fabric_hex) < DARK_FABRIC_L
