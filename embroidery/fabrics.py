"""Fabric types: the stitch settings each one gets, and what to put under and on top of it.

The numbers are common starting points from machine-embroidery practice, not measured
stitch-outs. `pull_comp_mm` is a default the UI loads into its slider (the user can move
it); `row_spacing_mm` (None = Ink/Stitch's 0.25 mm default) and `underlay` are written
onto every tatami fill.

Where it sits: pure data, no I/O. ``app.py`` sends :data:`FABRICS` to the UI and rejects
unknown keys. ``pipeline`` looks up the chosen type with :func:`get` and passes its
settings to ``vectorize``: ``row_spacing_mm`` also goes on contour fills (which have no
underlay) and scales satin zig-zag spacing, and ``underlay`` also drives satin zig-zag
underlay. The stabilizer, needle and label go into the SVG metadata (``svg_out``) and
the "Before you stitch" box, where :func:`density_label` describes the density.

Profile keys (all sizes in mm):

- ``label``: name shown in the UI and metadata.
- ``pull_comp_mm``: default pull compensation (capped by ``machines.MAX_PULL_COMP_MM``).
- ``row_spacing_mm``: fill row spacing; smaller is denser. None = standard 0.25 mm.
- ``underlay``: whether tatami fills (and wide satins) get underlay. Light thread on dark fabric gets underlay
  regardless (``colors.is_light_on_dark``).
- ``stabilizer``, ``needle``, ``extras``: advice text for the user.
- ``max_height_mm`` (optional): warn when the design's shorter side is taller than this
  (a beanie's folded cuff).
- ``stitch_warn`` (optional): warn above this many estimated stitches.
"""
from __future__ import annotations

FABRICS = {
    "cotton": {"label": "Cotton / woven", "pull_comp_mm": 0.2, "row_spacing_mm": None, "underlay": False,
               "stabilizer": "Tear-away", "needle": "75/11 sharp", "extras": []},
    "denim": {"label": "Denim / canvas", "pull_comp_mm": 0.2, "row_spacing_mm": None, "underlay": True,
              "stabilizer": "Heavy tear-away", "needle": "90/14 jeans", "extras": []},
    "tshirt": {"label": "T-shirt knit", "pull_comp_mm": 0.35, "row_spacing_mm": 0.3, "underlay": True,
               "stabilizer": "Cut-away", "needle": "75/11 ballpoint",
               "extras": ["Hoop the shirt smooth but not stretched."]},
    "polo": {"label": "Polo / piqué", "pull_comp_mm": 0.35, "row_spacing_mm": None, "underlay": True,
             "stabilizer": "Cut-away", "needle": "75/11 ballpoint",
             "extras": ["Water-soluble topping keeps stitches from sinking into the knit texture."]},
    "fleece": {"label": "Fleece / sweatshirt", "pull_comp_mm": 0.4, "row_spacing_mm": None, "underlay": True,
               "stabilizer": "Cut-away", "needle": "75/11 ballpoint",
               "extras": ["Water-soluble topping keeps stitches on top of the pile."]},
    "beanie": {"label": "Beanie / knit hat (acrylic, polyester)", "pull_comp_mm": 0.4, "row_spacing_mm": None,
               "underlay": True, "stabilizer": "Cut-away (no-show mesh)", "needle": "75/11 ballpoint",
               "extras": ["Water-soluble topping so stitches don't sink between the ribs.",
                          "Hoop the folded cuff flat without stretching it."],
               "max_height_mm": 55},
    "performance": {"label": "Performance polyester (athletic, jerseys)", "pull_comp_mm": 0.3,
                    "row_spacing_mm": 0.3, "underlay": True, "stabilizer": "No-show mesh cut-away",
                    "needle": "75/11 ballpoint",
                    "extras": ["Keep the design light: dense fills pucker thin, slippery polyester."],
                    "stitch_warn": 8000},
    "towel": {"label": "Towel / terry", "pull_comp_mm": 0.3, "row_spacing_mm": 0.22, "underlay": True,
              "stabilizer": "Tear-away", "needle": "75/11 sharp",
              "extras": ["Water-soluble topping keeps stitches on top of the loops."]},
    "felt": {"label": "Felt", "pull_comp_mm": 0.15, "row_spacing_mm": 0.3, "underlay": False,
             "stabilizer": "Tear-away, or none on thick felt", "needle": "75/11 sharp", "extras": []},
    "leather": {"label": "Leather / vinyl", "pull_comp_mm": 0.15, "row_spacing_mm": 0.3, "underlay": False,
                "stabilizer": "Tear-away", "needle": "Leather needle (80/12 or 90/14)",
                "extras": ["Every needle hole is permanent: don't pin inside the design, and test on a scrap."]},
}
DEFAULT_FABRIC = "cotton"
STANDARD_ROW_SPACING_MM = 0.25   # Ink/Stitch's default tatami row spacing


def get(key: str | None) -> dict:
    """Look up a fabric profile.

    Args:
        key: A :data:`FABRICS` key, case-insensitive. ``None`` or empty means
            :data:`DEFAULT_FABRIC`.

    Returns:
        A new dict with the profile's fields plus ``"key"`` (the normalized key).

    Raises:
        ValueError: If the key is unknown.
    """
    key = (key or DEFAULT_FABRIC).lower()
    if key not in FABRICS:
        raise ValueError(f"Unknown fabric type: {key}")
    return {"key": key, **FABRICS[key]}


def density_label(row_spacing_mm: float | None) -> str:
    """Describe a row spacing against the standard, e.g. ``"lighter (0.3 mm rows)"``.

    Smaller spacing means more rows, so it reads as "denser". ``None`` (or 0) is the
    standard 0.25 mm.
    """
    rs = row_spacing_mm or STANDARD_ROW_SPACING_MM
    if rs < STANDARD_ROW_SPACING_MM:
        return f"denser ({rs:g} mm rows)"
    if rs > STANDARD_ROW_SPACING_MM:
        return f"lighter ({rs:g} mm rows)"
    return f"standard ({rs:g} mm rows)"
