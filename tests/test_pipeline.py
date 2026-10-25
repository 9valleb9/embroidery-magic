import math
import re
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from app import app
from embroidery import pipeline
from tests.fixtures import flat_logo_png, messy_traced_svg, ringed_badge_png, script_stroke_png, thin_outline_png

SVG = "{http://www.w3.org/2000/svg}"
INK = "{http://www.inkscape.org/namespaces/inkscape}"
STITCH = "{http://inkstitch.org/namespace}"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def analyze(client, data, name, w=99, h=99, remove_bg="auto", hoop=(100, 100)):
    r = client.post("/api/analyze", files={"file": (name, data)},
                    data={"width_mm": w, "height_mm": h, "remove_bg": remove_bg,
                          "hoop_w": hoop[0], "hoop_h": hoop[1]})
    assert r.status_code == 200, r.text
    return r.json()


def request(session, fabric="#ffffff", **extra):
    return {"session_id": session["session_id"], "fabric_hex": fabric,
            "colors": [{"id": p["id"], "hex": p["hex"]} for p in session["palette"]], **extra}


def layers_of(svg_text):
    root = ET.fromstring(svg_text)
    return root, [g for g in root.iter(f"{SVG}g") if g.get(f"{INK}groupmode") == "layer"]


def check_svg(svg_text, w_mm, h_mm, hoop=(100, 100)):
    """The canvas is the hoop's stitch field (default 4×4 = 100×100 mm); the w×h design is centered on it."""
    root, layers = layers_of(svg_text)
    hw, hh = hoop
    assert root.get("width") == f"{hw:g}mm"
    assert root.get("height") == f"{hh:g}mm"
    assert root.get("viewBox") == f"0 0 {hw:g} {hh:g}"
    paths = list(root.iter(f"{SVG}path"))
    assert paths, "no shapes were written"
    nums = [float(v) for p in paths for v in re.findall(r"-?\d+(?:\.\d+)?", p.get("d"))]
    xs, ys = nums[0::2], nums[1::2]
    assert abs((min(xs) + max(xs)) / 2 - hw / 2) < 1.5 and abs((min(ys) + max(ys)) / 2 - hh / 2) < 1.5
    assert max(xs) - min(xs) <= w_mm + 0.2 and max(ys) - min(ys) <= h_mm + 0.2
    for p in paths:
        if p.get(f"{STITCH}stroke_method"):       # outline mode: running-stitch centerline
            assert p.get(f"{STITCH}stroke_method") == "running_stitch"
            assert "fill:none" in p.get("style")
            continue
        method = p.get(f"{STITCH}fill_method")
        assert method in ("tatami_fill", "contour_fill"), p.get("id")
        if method == "tatami_fill":
            angle = int(p.get(f"{STITCH}angle"))
            assert angle % 5 == 0 and 0 <= angle < 180
        style = p.get("style")
        assert "fill-rule:evenodd" in style and "stroke:none" in style
    for i, g in enumerate(layers, 1):
        assert g.get(f"{INK}label").startswith(f"{i} ")
    return root, layers


def test_flat_logo_svg(client):
    s = analyze(client, flat_logo_png(), "logo.png")
    assert s["k"] == 3 and (s["width_mm"], s["height_mm"]) == (99, 99)
    r = client.post("/api/generate", json=request(s)).json()
    root, layers = check_svg(r["svg"], 99, 99)
    # white background is the fabric -> two thread layers, each with a few clean shapes
    assert [g.get(f"{INK}label") for g in layers] == ["1 Dark green", "2 Light green"]
    for g in layers:
        assert 1 <= len(list(g.iter(f"{SVG}path"))) <= 4
    assert "not stitched — fabric" in r["svg"]


def test_messy_trace_is_cleaned(client):
    s = analyze(client, messy_traced_svg(), "trace.svg")
    hexes = [p["hex"] for p in s["palette"]]
    # navy, red, gold — the white margin is removed as background
    assert s["k"] == 3 and s["background"]["removed"]
    assert not any(h.startswith("#ff00") or h == "#0000ff" for h in hexes), "hidden content leaked in"
    r = client.post("/api/generate", json=request(s)).json()
    root, layers = check_svg(r["svg"], 99, 99)
    assert len(layers) == 3                       # navy, red, gold (white = fabric)
    for g in layers:                              # 400 specks gone: a few shapes per color
        assert len(list(g.iter(f"{SVG}path"))) <= 3


def test_aspect_ratio_kept_and_centered(client):
    s = analyze(client, messy_traced_svg(), "trace.svg", w=99, h=50)
    assert s["height_mm"] == 50 and s["width_mm"] <= 50
    check_svg(client.post("/api/generate", json=request(s)).json()["svg"], s["width_mm"], s["height_mm"])


def test_larger_than_hoop_is_blocked(client):
    r = client.post("/api/analyze", files={"file": ("logo.png", flat_logo_png())},
                    data={"width_mm": 110, "height_mm": 100})
    assert r.status_code == 400 and "100" in r.json()["detail"]


def test_bigger_hoop_rotates_to_fit_landscape_design(client):
    # 170×85 doesn't fit a 5×7 hoop upright (130 wide) but does rotated (180×130)
    s = analyze(client, messy_traced_svg(), "trace.svg", w=170, h=85, hoop=(130, 180))
    assert s["height_mm"] == 85
    r = client.post("/api/generate", json=request(s, hoop_w=130, hoop_h=180, speed_spm=800)).json()
    check_svg(r["svg"], s["width_mm"], s["height_mm"], hoop=(130, 180) if s["width_mm"] <= s["height_mm"] else (180, 130))
    assert r["machine"]["speed_spm"] == 800
    assert r["machine"]["estimated_minutes"] == round(r["machine"]["estimated_stitches"] / 800, 1)


def test_hoop_limits_and_formats_validated(client):
    r = client.post("/api/analyze", files={"file": ("logo.png", flat_logo_png())},
                    data={"width_mm": 150, "height_mm": 150, "hoop_w": 130, "hoop_h": 180})
    assert r.status_code == 400 and "130 × 180" in r.json()["detail"]
    s = analyze(client, flat_logo_png(), "logo.png")
    assert client.post("/api/generate", json=request(s, format="doc")).status_code == 400
    assert client.post("/api/generate", json=request(s, format="jef")).status_code == 200


def test_svg_is_stamped_for_current_inkstitch(client):
    s = analyze(client, flat_logo_png(), "logo.png")
    svg = client.post("/api/generate", json=request(s)).json()["svg"]
    # without the stamp Ink/Stitch rewrites params as legacy and opens an "update?" dialog
    assert "<inkstitch:inkstitch_svg_version>4</inkstitch:inkstitch_svg_version>" in svg


def test_pull_compensation_and_overlap(client):
    s = analyze(client, flat_logo_png(), "logo.png")        # dark and light green touch
    req = request(s, fabric="#ffffff")
    plain = client.post("/api/generate", json={**req, "overlap_mm": 0, "pull_comp_mm": 0}).json()
    tuned = client.post("/api/generate", json={**req, "overlap_mm": 0.5, "pull_comp_mm": 0.35}).json()
    root, _ = layers_of(tuned["svg"])
    tatami = [p for p in root.iter(f"{SVG}path") if p.get(f"{STITCH}fill_method") == "tatami_fill"]
    assert tatami and all(p.get(f"{STITCH}pull_compensation_mm") == "0.35" for p in tatami)
    root0, _ = layers_of(plain["svg"])
    assert not any(p.get(f"{STITCH}pull_compensation_mm") for p in root0.iter(f"{SVG}path"))
    # overlap makes colors reach under each other (more stitched area) without growing the outer edge
    assert tuned["machine"]["estimated_stitches"] > plain["machine"]["estimated_stitches"]

    def extent(svg):
        nums = [float(v) for p in layers_of(svg)[0].iter(f"{SVG}path") for v in re.findall(r"-?\d+(?:\.\d+)?", p.get("d"))]
        return max(nums[0::2]) - min(nums[0::2])
    assert extent(tuned["svg"]) <= extent(plain["svg"]) + 0.05
    assert client.post("/api/generate", json={**req, "overlap_mm": 5}).status_code == 400


def test_overlap_closes_hairline_fabric_gaps():
    """Two colors with a 0.3 mm fabric line between them: overlap 0.3 closes it."""
    from PIL import Image, ImageDraw
    import io
    img = Image.new("RGB", (1000, 1000), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 498, 900], fill="#1b5e20")
    d.rectangle([502, 100, 900, 900], fill="#c62828")       # 3 px = 0.3 mm white line (1000 px = 100 mm)
    buf = io.BytesIO(); img.save(buf, "PNG")
    s = pipeline.new_session(buf.getvalue(), "gap.png", 100, 100, None, remove_bg="off", sliver_mm=0.2)
    req = {"fabric_hex": "#ffffff", "colors": [{"id": i, "hex": h} for i, h in enumerate(s.hexes)]}
    gaps = lambda r: [w for w in r["warnings"] if w["code"] == "narrow_gap"]
    assert gaps(pipeline.generate(s, {**req, "overlap_mm": 0}))
    assert not gaps(pipeline.generate(s, {**req, "overlap_mm": 0.3}))


def test_sliver_cleanup_absorbs_thin_bands():
    """A 0.5 mm fabric line between two colors survives the default cleanup (and is
    flagged as a gap that will close up) but is absorbed at sliver_mm=0.7."""
    from PIL import Image, ImageDraw
    import io
    img = Image.new("RGB", (1000, 1000), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 497, 900], fill="#1b5e20")
    d.rectangle([503, 100, 900, 900], fill="#c62828")       # 5 px = 0.5 mm (1000 px = 100 mm)
    buf = io.BytesIO(); img.save(buf, "PNG")

    def gaps(sliver):
        s = pipeline.new_session(buf.getvalue(), "gap.png", 100, 100, None, remove_bg="off", sliver_mm=sliver)
        r = pipeline.generate(s, {"fabric_hex": "#ffffff", "overlap_mm": 0,
                                  "colors": [{"id": i, "hex": h} for i, h in enumerate(s.hexes)]})
        return [w for w in r["warnings"] if w["code"] == "narrow_gap"]
    assert gaps(0.35)
    assert not gaps(0.7)


def test_machine_summary(client):
    s = analyze(client, flat_logo_png(), "logo.png")
    m = client.post("/api/generate", json=request(s)).json()["machine"]
    assert m["thread_changes"] == 1 and m["estimated_stitches"] > 1000
    assert m["estimated_minutes"] == round(m["estimated_stitches"] / 400, 1)


def test_fabric_change_restores_knockout(client):
    s = analyze(client, flat_logo_png(), "logo.png")
    req = request(s)
    white = next(p["id"] for p in s["palette"] if p["hex"] == "#ffffff")
    plan = client.post("/api/plan", json=req).json()
    assert next(c for c in plan["colors"] if c["id"] == white)["fabric"] is True

    for c in req["colors"]:
        c["was_fabric"] = next(d["fabric"] for d in plan["colors"] if d["id"] == c["id"])
    req.update(fabric_hex="#1f2a44", prev_fabric_hex="#ffffff")
    r = client.post("/api/generate", json=req).json()
    root, layers = check_svg(r["svg"], 99, 99)
    labels = [g.get(f"{INK}label") for g in layers]
    assert any(l.endswith("White (fabric was white)") for l in labels)
    assert any(w["code"] == "restored" for w in r["warnings"])
    assert r["summary"]["bobbin_name"] == "Black"
    # light thread on dark fabric: underlay + dense rows on tatami shapes
    white_layer = next(g for g in layers if "White" in g.get(f"{INK}label"))
    for p in white_layer.iter(f"{SVG}path"):
        if p.get(f"{STITCH}fill_method") == "tatami_fill":
            assert p.get(f"{STITCH}fill_underlay") == "true"
            assert p.get(f"{STITCH}row_spacing_mm") == "0.25"


def test_low_contrast_warning_and_outline_fix(client):
    s = analyze(client, flat_logo_png(), "logo.png")
    dark = next(p["id"] for p in s["palette"] if p["hex"] == "#1b5e20")
    # ΔE 4: so close it's knocked out as fabric automatically
    plan = client.post("/api/plan", json=request(s, fabric="#2a6a2e")).json()
    assert next(c for c in plan["colors"] if c["id"] == dark)["fabric"] is True
    # ΔE ~10: stitched, but warned with one-click fixes
    req = request(s, fabric="#2e7d32")
    plan = client.post("/api/plan", json=req).json()
    warn = next(w for w in plan["warnings"] if w["code"] == "low_contrast")
    assert warn["color_id"] == dark
    assert {f["type"] for f in warn["fixes"]} == {"swap", "outline"}
    req["outlines"] = [warn["color_id"]]
    r = client.post("/api/generate", json=req).json()
    root, layers = check_svg(r["svg"], 99, 99)
    outline = next(g for g in layers if "outline" in g.get(f"{INK}label"))
    assert all(p.get(f"{STITCH}fill_method") == "contour_fill" for p in outline.iter(f"{SVG}path"))


def test_stitch_params_rules():
    from shapely.geometry import Point, box
    from embroidery.vectorize import stitch_params

    assert stitch_params(box(0, 0, 40, 0.8))["fill_method"] == "contour_fill"           # thin
    assert stitch_params(box(0, 0, 40, 0.8), row_spacing_mm=0.22, underlay=True) == \
        {"fill_method": "contour_fill", "row_spacing_mm": "0.22"}                         # density, no underlay
    ring = Point(0, 0).buffer(20).difference(Point(0, 0).buffer(17))
    assert stitch_params(ring)["fill_method"] == "contour_fill"                         # ring
    square = stitch_params(box(0, 0, 30, 20))
    assert square == {"fill_method": "tatami_fill", "angle": "45"}                        # never 0/90 on boxy shapes
    from shapely import affinity
    slanted = stitch_params(affinity.rotate(box(0, 0, 40, 10), 30, origin="centroid"))
    assert slanted["angle"] in ("150", "30")
    assert stitch_params(box(0, 0, 30, 20), light_on_dark=True)["row_spacing_mm"] == "0.25"
    knit = stitch_params(box(0, 0, 30, 20), row_spacing_mm=0.3, underlay=True)
    assert knit["row_spacing_mm"] == "0.3" and knit["fill_underlay"] == "true"
    # light on dark never gets lighter rows than 0.25 mm, whatever the fabric asks for
    assert stitch_params(box(0, 0, 30, 20), light_on_dark=True, row_spacing_mm=0.3)["row_spacing_mm"] == "0.25"
    assert stitch_params(box(0, 0, 30, 20), light_on_dark=True, row_spacing_mm=0.22)["row_spacing_mm"] == "0.22"


def test_fabric_type_sets_stitches_and_advice(client):
    from embroidery import fabrics
    s = analyze(client, flat_logo_png(), "logo.png")
    cotton = client.post("/api/generate", json=request(s)).json()
    assert cotton["fabric"]["key"] == "cotton" and cotton["fabric"]["stabilizer"] == "Tear-away"
    root, _ = layers_of(cotton["svg"])
    tatami = [p for p in root.iter(f"{SVG}path") if p.get(f"{STITCH}fill_method") == "tatami_fill"]
    assert tatami and not any(p.get(f"{STITCH}fill_underlay") or p.get(f"{STITCH}row_spacing_mm") for p in tatami)

    towel = client.post("/api/generate", json={**request(s), "fabric_type": "towel"}).json()
    root, _ = layers_of(towel["svg"])
    tatami = [p for p in root.iter(f"{SVG}path") if p.get(f"{STITCH}fill_method") == "tatami_fill"]
    assert all(p.get(f"{STITCH}row_spacing_mm") == "0.22" and p.get(f"{STITCH}fill_underlay") == "true"
               for p in tatami)
    assert towel["machine"]["estimated_stitches"] > cotton["machine"]["estimated_stitches"]   # denser rows
    assert any("topping" in n for n in towel["fabric"]["notes"])
    design = next(root.iter("{urn:embroidery-magic}design"))
    assert design.get("fabric-type") == fabrics.FABRICS["towel"]["label"] and design.get("stabilizer")

    perf = client.post("/api/generate", json={**request(s), "fabric_type": "performance"}).json()
    assert any(w["code"] == "fabric_stitches" for w in perf["warnings"]) == \
        (perf["machine"]["estimated_stitches"] > 8000)
    assert client.post("/api/generate", json={**request(s), "fabric_type": "burlap"}).status_code == 400


def test_beanie_warns_when_too_tall(client):
    s = analyze(client, flat_logo_png(), "logo.png", w=80, h=80)
    r = client.post("/api/generate", json={**request(s), "fabric_type": "beanie"}).json()
    assert r["fabric"]["key"] == "beanie"
    assert min(r["width_mm"], r["height_mm"]) > 55
    assert any(w["code"] == "fabric_height" for w in r["warnings"])
    small = analyze(client, flat_logo_png(), "logo.png", w=50, h=50)
    r = client.post("/api/generate", json={**request(small), "fabric_type": "beanie"}).json()
    assert not any(w["code"] == "fabric_height" for w in r["warnings"])


def test_bad_upload(client):
    r = client.post("/api/analyze", files={"file": ("x.png", b"not an image")})
    assert r.status_code == 422


def test_background_removed_but_interior_kept_on_navy(client):
    s = analyze(client, ringed_badge_png(), "badge.png")
    assert s["background"]["hex"] == "#ffffff" and s["background"]["removed"]
    assert s["width_mm"] == 99                        # cropped to the badge, fills the size
    white = next(p for p in s["palette"] if p["hex"] == "#ffffff")
    r = client.post("/api/generate", json=request(s, fabric="#1f2a44")).json()
    root, layers = check_svg(r["svg"], 99, 99)
    labels = [g.get(f"{INK}label") for g in layers]
    assert any("White" in l for l in labels)          # interior white is stitched on navy...
    # ...but only the interior: a 42 mm-radius disc minus the 32 mm gold square ≈ 4,517 mm² at
    # 100 mm wide, scaled to the 99 mm design
    assert abs(white["area_mm2"] - (3.14159 * 42 ** 2 - 32 ** 2) * 0.99 ** 2) < 250

    kept = analyze(client, ringed_badge_png(), "badge.png", remove_bg="off")
    assert not kept["background"]["removed"] and kept["width_mm"] == 99
    white_kept = next(p for p in kept["palette"] if p["hex"] == "#ffffff")
    assert white_kept["area_mm2"] > 5_000              # the margin is part of the design again


def test_thin_outline_becomes_running_stitch(client):
    s = analyze(client, thin_outline_png(), "outline.png")

    r = client.post("/api/generate", json=request(s)).json()
    root, layers = check_svg(r["svg"], 99, 99)
    g = next(g for g in layers if "Black" in g.get(f"{INK}label"))
    strokes = [p for p in g.iter(f"{SVG}path") if p.get(f"{STITCH}stroke_method")]
    fills = [p for p in g.iter(f"{SVG}path") if p.get(f"{STITCH}fill_method")]
    assert 1 <= len(strokes) <= 3                  # the ring, as one (or a few) continuous runs
    assert strokes[0].get(f"{STITCH}bean_stitch_repeats") == "1"
    assert len(fills) == 1                         # the 4 mm bar is still a fill
    assert not any(w["code"] == "narrow_stitch" and "Black" in w["message"] for w in r["warnings"])

    off = client.post("/api/generate", json=request(s, thin_lines=False)).json()
    root, layers = check_svg(off["svg"], 99, 99)
    assert not any(p.get(f"{STITCH}stroke_method") for p in root.iter(f"{SVG}path"))
    assert any(w["code"] == "narrow_stitch" and "Black" in w["message"] for w in off["warnings"])


def test_satin_mode_turns_strokes_into_satin_columns(client):
    s = analyze(client, script_stroke_png(), "script.png")
    off = client.post("/api/generate", json=request(s)).json()
    assert "satin_column" not in off["svg"] and off["layers"][0]["satins"] == 0
    on = client.post("/api/generate", json={**request(s), "satin": True}).json()
    layer = on["layers"][0]
    assert layer["satins"] >= 1 and layer["shapes"] - layer["satins"] - layer["lines"] >= 1   # the disc stays a fill
    root, _ = layers_of(on["svg"])
    satins = [p for p in root.iter(f"{SVG}path") if p.get(f"{STITCH}satin_column") == "true"]
    assert len(satins) == layer["satins"]
    for p in satins:   # two rails with paired nodes, drawn as a stroke
        rails = [r for r in p.get("d").split("M ") if r.strip()]
        assert len(rails) == 2 and len(rails[0].split(" L ")) == len(rails[1].split(" L "))
        assert "fill:none" in p.get("style") and p.get(f"{STITCH}zigzag_spacing_mm") == "0.4"
    gaps = lambda r: sum(int(w["message"].split()[0]) for w in r["warnings"] if w["code"] == "narrow_gap")
    assert gaps(on) <= gaps(off)                     # columns meet the rest of the shape without new gaps


def test_satin_column_geometry():
    import cv2
    import numpy as np
    from embroidery.raster import PX_PER_MM
    from embroidery.vectorize import satin_params, split_satin

    m = np.zeros((60 * PX_PER_MM, 80 * PX_PER_MM), np.uint8)
    pts = np.array([[(10 + t * 0.6) * PX_PER_MM, (30 + 12 * np.sin(t / 6)) * PX_PER_MM] for t in range(70)], np.int32)
    cv2.polylines(m, [pts], False, 1, 3 * PX_PER_MM)                     # a 3 mm stroke
    cv2.circle(m, (65 * PX_PER_MM, 15 * PX_PER_MM), 8 * PX_PER_MM, 1, -1)   # a 16 mm blob
    cv2.rectangle(m, (5 * PX_PER_MM, 50 * PX_PER_MM), (75 * PX_PER_MM, 59 * PX_PER_MM), 1, -1)  # 9 mm bar: too wide
    rest, sats = split_satin(m.astype(bool))
    assert len(sats) == 1 and abs(sats[0].width_mm - 3) < 0.4
    assert len(sats[0].rail1) == len(sats[0].rail2)
    left = rest.sum() / PX_PER_MM ** 2
    assert abs(left - (math.pi * 64 + 70 * 9)) < 40                      # blob + bar stay fills, stroke is gone

    # a stroke ending in a small serif: the serif stays a fill, but no strip along the column's edges does
    m = np.zeros((40 * PX_PER_MM, 60 * PX_PER_MM), np.uint8)
    cv2.rectangle(m, (10 * PX_PER_MM, 18 * PX_PER_MM), (50 * PX_PER_MM, 22 * PX_PER_MM), 1, -1)  # 4 mm bar
    cv2.rectangle(m, (48 * PX_PER_MM, 14 * PX_PER_MM), (52 * PX_PER_MM, 26 * PX_PER_MM), 1, -1)  # end serif
    rest, sats = split_satin(m.astype(bool))
    assert sats
    cols = np.flatnonzero(rest.any(axis=0)) / PX_PER_MM
    assert not cols.size or cols.min() > 40                                 # any fill is at the serif end only
    towel = satin_params(3.0, 0.3, 0.22, underlay=True)
    assert towel["zigzag_spacing_mm"] == "0.35" and towel["zigzag_underlay"] == "true"
    assert satin_params(1.5)["contour_underlay"] == "false"


def test_satin_mode_stitches_hairlines_as_1mm_satin(client):
    s = analyze(client, thin_outline_png(), "outline.png")        # a ~0.6 mm outline ring
    off = client.post("/api/generate", json=request(s)).json()
    assert sum(l["lines"] for l in off["layers"]) >= 1
    on = client.post("/api/generate", json={**request(s), "satin": True}).json()
    assert sum(l["lines"] for l in on["layers"]) == 0               # no loose-looking running stitch
    root, _ = layers_of(on["svg"])
    widths = []
    for p in root.iter(f"{SVG}path"):
        if p.get(f"{STITCH}satin_column") != "true":
            continue
        r1, r2 = [[tuple(map(float, xy.split())) for xy in r.strip().split(" L ")]
                  for r in p.get("d").split("M ") if r.strip()]
        widths.append(min(math.dist(a, b) for a, b in zip(r1, r2)))
    assert widths and min(widths) >= 0.95                            # hairlines widened to 1 mm
