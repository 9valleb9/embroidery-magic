"""Color math: CIE Lab, ΔE2000, thread names, bobbin and contrast helpers.

Where it sits: a shared utility used by most of the flow. ``palette`` formats its
cluster colors with :func:`rgb_to_hex`, ``pipeline`` uses the ΔE thresholds below for its color decisions (merge, fabric
knockout, low-contrast warnings, light-on-dark adjustments), names thread layers with
:func:`color_name`, and picks the bobbin and outline colors here. ``export`` uses
:func:`hex_to_rgb` to paint the stitch-out preview.

Key concepts:

- Colors travel through the app as ``"#rrggbb"`` hex strings (sRGB).
- Lab values are CIE L*a*b* from scikit-image (D65): L* runs 0 (black) to 100 (white).
- All color differences are CIEDE2000 (``skimage.color.deltaE_ciede2000``). Roughly,
  ΔE < 2 is hard to see, ΔE 5 is a near match, ΔE 25+ is clearly different.

Main entry points: :func:`delta_e`, :func:`color_name`, :func:`fabric_label`,
:func:`bobbin_for`, :func:`contrasting_bw`, :func:`contrast_swap`,
:func:`is_light_on_dark`.

Gotcha: converting Lab back to hex (:func:`lab_to_hex`) clips out-of-gamut colors, so a
round trip through Lab is not exact for saturated colors.
"""
from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

MERGE_DE = 5.0          # near-identical design colors merge by default (fewer thread changes)
KNOCKOUT_DE = 8.0       # design color this close to the fabric -> left as fabric
LOW_CONTRAST_DE = 15.0  # stitched color this close to the fabric -> barely shows
LIGHT_THREAD_L = 70.0   # thread L* above this counts as "light" for the light-on-dark rule
DARK_FABRIC_L = 35.0    # fabric L* below this counts as "dark" for the light-on-dark rule

# Fabric color presets offered in the UI. A fabric hex that matches one exactly is
# called by this name in messages and metadata.
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
    """Parse a hex color (``"#rrggbb"``, ``"rrggbb"`` or short ``"#rgb"``) into 0-255 RGB.

    Args:
        h: The hex string. Surrounding whitespace and a leading ``#`` are ignored.

    Returns:
        ``(r, g, b)`` integers in 0-255.

    Raises:
        ValueError: If the string is not 3 or 6 hex digits.
    """
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"bad hex color: {h!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    """Format an RGB triple (0-255, floats allowed) as lowercase ``"#rrggbb"``, clamping and rounding."""
    r, g, b = (int(round(max(0, min(255, c)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_lab(h: str) -> np.ndarray:
    """Convert a hex color to CIE Lab.

    Args:
        h: Hex color string (see :func:`hex_to_rgb`).

    Returns:
        A float64 array ``[L*, a*, b*]`` of shape (3,).
    """
    rgb = np.array(hex_to_rgb(h), dtype=np.float64).reshape(1, 1, 3) / 255.0
    return rgb2lab(rgb).reshape(3)


def lab_to_hex(lab) -> str:
    """Convert a CIE Lab triple back to a hex color (out-of-gamut values are clipped)."""
    rgb = lab2rgb(np.asarray(lab, dtype=np.float64).reshape(1, 1, 3)).reshape(3)
    return rgb_to_hex(rgb * 255)


def delta_e(h1: str, h2: str) -> float:
    """CIEDE2000 color difference between two hex colors (0 = identical)."""
    return float(deltaE_ciede2000(hex_to_lab(h1), hex_to_lab(h2)))


def lightness(h: str) -> float:
    """CIE L* (0 black to 100 white) of a hex color."""
    return float(hex_to_lab(h)[0])


# NAMED_COLORS converted to Lab once at import, so color_name doesn't redo it per call.
_NAMED_LABS = {name: hex_to_lab(hx) for name, hx in NAMED_COLORS.items()}


def color_name(h: str) -> str:
    """Name of the nearest entry in :data:`NAMED_COLORS` by ΔE2000 (used for thread layer names)."""
    lab = hex_to_lab(h)
    return min(_NAMED_LABS, key=lambda n: float(deltaE_ciede2000(lab, _NAMED_LABS[n])))


def fabric_name(h: str) -> str:
    """Name for a fabric color: the preset name on an exact (case-insensitive) match, else the nearest named color."""
    for name, hx in FABRIC_PRESETS.items():
        if hx.lower() == h.lower():
            return name
    return color_name(h)


def fabric_label(h: str) -> str:
    """For messages: the preset name, or the nearest name plus the hex for custom fabrics.

    Unlike :func:`fabric_name`, a custom fabric keeps its hex (``"Navy (#202b40)"``) so the
    user can tell it apart from the preset of the same name.
    """
    for name, hx in FABRIC_PRESETS.items():
        if hx.lower() == h.lower():
            return name
    return f"{color_name(h)} ({h.lower()})"


def bobbin_for(fabric_hex: str) -> tuple[str, str]:
    """White or black bobbin thread, whichever is closer to the fabric.

    The bobbin thread can peek through on the top side, so it should blend with the
    fabric. Ties go to white.

    Returns:
        ``(name, hex)``: ``("White", "#ffffff")`` or ``("Black", "#000000")``.
    """
    if delta_e(fabric_hex, "#ffffff") <= delta_e(fabric_hex, "#000000"):
        return "White", "#ffffff"
    return "Black", "#000000"


def contrasting_bw(fabric_hex: str) -> tuple[str, str]:
    """Black or white, whichever stands out more against the fabric.

    Used for the 1 mm outline fix, the design border and as the last-resort swap color.
    Ties go to white. Black is ``#111111`` (the "Black" named color), not pure black.

    Returns:
        ``(name, hex)``: ``("White", "#ffffff")`` or ``("Black", "#111111")``.
    """
    if delta_e(fabric_hex, "#ffffff") >= delta_e(fabric_hex, "#000000"):
        return "White", "#ffffff"
    return "Black", "#111111"


def contrast_swap(thread_hex: str, fabric_hex: str, target_de: float = 25.0) -> str:
    """Suggest a replacement thread: same hue, lightness pushed away from the fabric.

    Steps L* by 4 at a time (up on dark fabric, L* < 50; down on light fabric), keeping
    a* and b*, until the candidate is at least ``target_de`` from the fabric. Gives up
    after 30 steps or when L* hits 0 or 100.

    Args:
        thread_hex: The low-contrast thread color.
        fabric_hex: The fabric color.
        target_de: ΔE2000 the replacement must reach.

    Returns:
        The replacement hex, or black/white from :func:`contrasting_bw` if no lightness
        shift reaches the target.
    """
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
    """True for light thread (L* > 70) on dark fabric (L* < 35).

    Such colors always get fill underlay and rows no lighter than 0.25 mm (see
    ``vectorize.stitch_params``), so the dark fabric doesn't show through.
    """
    return lightness(thread_hex) > LIGHT_THREAD_L and lightness(fabric_hex) < DARK_FABRIC_L
