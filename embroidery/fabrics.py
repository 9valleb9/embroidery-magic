"""Fabric types: the stitch settings each one gets, and what to put under and on top of it.

The numbers are common starting points from machine-embroidery practice, not measured
stitch-outs. `pull_comp_mm` is a default the UI loads into its slider (the user can move
it); `row_spacing_mm` (None = Ink/Stitch's 0.25 mm default) and `underlay` are written
onto every tatami fill."""
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
    key = (key or DEFAULT_FABRIC).lower()
    if key not in FABRICS:
        raise ValueError(f"Unknown fabric type: {key}")
    return {"key": key, **FABRICS[key]}


def density_label(row_spacing_mm: float | None) -> str:
    rs = row_spacing_mm or STANDARD_ROW_SPACING_MM
    if rs < STANDARD_ROW_SPACING_MM:
        return f"denser ({rs:g} mm rows)"
    if rs > STANDARD_ROW_SPACING_MM:
        return f"lighter ({rs:g} mm rows)"
    return f"standard ({rs:g} mm rows)"
