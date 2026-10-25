"""Machine-file route: verification logic, plus real Ink/Stitch exports when installed."""
import io

import pyembroidery as pe
import pytest

from embroidery import export, pipeline
from tests.fixtures import flat_logo_png, script_stroke_png, thin_outline_png


def _pes(pattern: pe.EmbPattern) -> bytes:
    buf = io.BytesIO()
    pe.write_pes(pattern, buf)
    return buf.getvalue()


def _square(p, x0, y0, size, step=20):
    """Running stitch around a square (units of 0.1 mm), every stitch <= step."""
    for x in range(x0, x0 + size + 1, step):
        p.add_stitch_absolute(pe.STITCH, x, y0)
    for y in range(y0 + step, y0 + size + 1, step):
        p.add_stitch_absolute(pe.STITCH, x0 + size, y)
    for x in range(x0 + size - step, x0 - 1, -step):
        p.add_stitch_absolute(pe.STITCH, x, y0 + size)


def test_analyze_pes_finds_hand_trims_and_colors():
    p = pe.EmbPattern()
    p.add_thread(pe.EmbThread("#000080"))
    p.add_thread(pe.EmbThread("#ff0000"))
    _square(p, 0, 0, 200)                      # color 1, shape A
    p.add_stitch_absolute(pe.JUMP, 600, 0)     # jump within color 1 -> hand trim
    _square(p, 600, 0, 200)                    # color 1, shape B
    p.add_command(pe.COLOR_CHANGE)
    p.add_stitch_absolute(pe.JUMP, 300, 300)   # first move of a new color: not a hand trim
    _square(p, 300, 300, 200)
    p.add_command(pe.END)

    r = export.analyze_file(_pes(p), "pes", expected_colors=2)
    assert r["ok"], r["problems"]
    assert r["signature"].startswith("#PES")
    assert r["color_blocks"] == 2 and r["thread_changes"] == 1
    assert len(r["trims"]) == 1 and r["trims"][0]["color"] == 1
    assert r["trims"][0]["length_mm"] == pytest.approx(63.2, abs=0.2)   # (0,20 mm) -> (60,0 mm)
    assert r["minutes"] == round(r["stitches"] / 400, 1)


def test_analyze_pes_flags_oversize():
    p = pe.EmbPattern()
    p.add_thread(pe.EmbThread("#000000"))
    _square(p, 0, 0, 1100, step=50)            # 110 mm wide
    p.add_command(pe.END)
    r = export.analyze_file(_pes(p))
    assert not r["ok"] and any("100 × 100 mm hoop" in x for x in r["problems"])
    # the same 110 mm design fits a 5×7 hoop
    assert export.analyze_file(_pes(p), hoop=(130, 180))["ok"]


def test_analyze_jef_and_dst():
    p = pe.EmbPattern()
    p.add_thread(pe.EmbThread("#000080"))
    p.add_thread(pe.EmbThread("#ff0000"))
    _square(p, 0, 0, 200)
    p.add_command(pe.COLOR_CHANGE)
    _square(p, 300, 300, 200)
    p.add_command(pe.END)
    for fmt, writer in (("jef", pe.write_jef), ("dst", pe.write_dst)):
        buf = io.BytesIO()
        writer(p, buf)
        r = export.analyze_file(buf.getvalue(), fmt, expected_colors=2)
        assert r["ok"], (fmt, r["problems"])
        assert r["format"] == fmt and r["color_blocks"] == 2
    assert r["palette"] is None and r["threads"][0]["hex"] is None   # DST stores no colors


@pytest.mark.inkstitch
@pytest.mark.skipif(not export.find_inkstitch(), reason="Ink/Stitch not installed")
def test_inkstitch_export_is_se400_ready():
    s = pipeline.new_session(flat_logo_png(), "logo.png", 99, 99, None)
    g = pipeline.generate(s, {"fabric_hex": "#ffffff",
                              "colors": [{"id": i, "hex": h} for i, h in enumerate(s.hexes)]})
    data = export.export_file(g["svg"], "pes")
    r = export.analyze_file(data, "pes", g["machine"]["thread_blocks"])
    assert r["ok"], r["problems"]
    assert r["extent_mm"][0] <= 100 and r["extent_mm"][1] <= 100
    assert r["color_blocks"] == 2 and r["stitches"] > 1000
    # the geometry-based estimate shown before export should be in the right ballpark
    assert 0.6 < g["machine"]["estimated_stitches"] / r["stitches"] < 1.6


@pytest.mark.inkstitch
@pytest.mark.skipif(not export.find_inkstitch(), reason="Ink/Stitch not installed")
def test_inkstitch_exports_outline_lines():
    s = pipeline.new_session(thin_outline_png(), "outline.png", 99, 99, None)
    g = pipeline.generate(s, {"fabric_hex": "#ffffff",
                              "colors": [{"id": i, "hex": h} for i, h in enumerate(s.hexes)]})
    assert any(l["lines"] for l in g["layers"])
    r = export.analyze_file(export.export_file(g["svg"], "pes"), "pes", g["machine"]["thread_blocks"])
    assert r["ok"], r["problems"]
    black = next(t for t in r["threads"] if t["hex"] < "#303030")
    # a ~300 mm ring as a triple run of 2 mm stitches is ~450 stitches; a fill would be far more
    assert 250 < black["stitches"] < 2500


@pytest.mark.inkstitch
@pytest.mark.skipif(not export.find_inkstitch(), reason="Ink/Stitch not installed")
@pytest.mark.parametrize("fmt", ["jef", "dst"])
def test_inkstitch_exports_other_formats_on_bigger_hoop(fmt):
    s = pipeline.new_session(flat_logo_png(), "logo.png", 120, 120, None)
    g = pipeline.generate(s, {"fabric_hex": "#ffffff", "hoop_w": 130, "hoop_h": 180,
                              "colors": [{"id": i, "hex": h} for i, h in enumerate(s.hexes)]})
    r = export.analyze_file(export.export_file(g["svg"], fmt), fmt, g["machine"]["thread_blocks"],
                            hoop=(130, 180))
    assert r["ok"], r["problems"]
    assert 115 < max(r["extent_mm"]) <= 121


@pytest.mark.inkstitch
@pytest.mark.skipif(not export.find_inkstitch(), reason="Ink/Stitch not installed")
def test_inkstitch_exports_satin_columns():
    s = pipeline.new_session(script_stroke_png(), "script.png", 99, 99, None)
    g = pipeline.generate(s, {"fabric_hex": "#ffffff", "satin": True,
                              "colors": [{"id": i, "hex": h} for i, h in enumerate(s.hexes)]})
    assert any(l["satins"] for l in g["layers"])
    r = export.analyze_file(export.export_file(g["svg"], "pes"), "pes", g["machine"]["thread_blocks"])
    assert r["ok"], r["problems"]
    assert r["longest_stitch_mm"] <= export.MAX_STITCH_MM
