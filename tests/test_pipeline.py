import re
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from app import app
from embroidery import pipeline
from tests.fixtures import flat_logo_png, messy_traced_svg

SVG = "{http://www.w3.org/2000/svg}"
INK = "{http://www.inkscape.org/namespaces/inkscape}"
STITCH = "{http://inkstitch.org/namespace}"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def analyze(client, data, name, w=100, h=100):
    r = client.post("/api/analyze", files={"file": (name, data)}, data={"width_mm": w, "height_mm": h})
    assert r.status_code == 200, r.text
    return r.json()


def request(session, fabric="#ffffff", **extra):
    return {"session_id": session["session_id"], "fabric_hex": fabric,
            "colors": [{"id": p["id"], "hex": p["hex"]} for p in session["palette"]], **extra}


def layers_of(svg_text):
    root = ET.fromstring(svg_text)
    return root, [g for g in root.iter(f"{SVG}g") if g.get(f"{INK}groupmode") == "layer"]


def check_svg(svg_text, w_mm, h_mm):
    """The canvas is always the SE400's 100×100 mm hoop; the w×h design is centered on it."""
    root, layers = layers_of(svg_text)
    assert root.get("width") == "100mm"
    assert root.get("height") == "100mm"
    assert root.get("viewBox") == "0 0 100 100"
    paths = list(root.iter(f"{SVG}path"))
    assert paths, "no shapes were written"
    nums = [float(v) for p in paths for v in re.findall(r"-?\d+(?:\.\d+)?", p.get("d"))]
    xs, ys = nums[0::2], nums[1::2]
    assert abs((min(xs) + max(xs)) / 2 - 50) < 1.5 and abs((min(ys) + max(ys)) / 2 - 50) < 1.5
    assert max(xs) - min(xs) <= w_mm + 0.2 and max(ys) - min(ys) <= h_mm + 0.2
    for p in paths:
        method = p.get(f"{STITCH}fill_method")
        assert method in ("auto_fill", "contour_fill"), p.get("id")
        if method == "auto_fill":
            angle = int(p.get(f"{STITCH}angle"))
            assert angle % 5 == 0 and 0 <= angle < 180
        style = p.get("style")
        assert "fill-rule:evenodd" in style and "stroke:none" in style
    for i, g in enumerate(layers, 1):
        assert g.get(f"{INK}label").startswith(f"{i} ")
    return root, layers


def test_flat_logo_svg(client):
    s = analyze(client, flat_logo_png(), "logo.png")
    assert s["k"] == 3 and (s["width_mm"], s["height_mm"]) == (100, 100)
    r = client.post("/api/generate", json=request(s)).json()
    root, layers = check_svg(r["svg"], 100, 100)
    # white background is the fabric -> two thread layers, each with a few clean shapes
    assert [g.get(f"{INK}label") for g in layers] == ["1 Dark green", "2 Light green"]
    for g in layers:
        assert 1 <= len(list(g.iter(f"{SVG}path"))) <= 4
    assert "not stitched — fabric" in r["svg"]


def test_messy_trace_is_cleaned(client):
    s = analyze(client, messy_traced_svg(), "trace.svg")
    hexes = [p["hex"] for p in s["palette"]]
    assert s["k"] == 4
    assert not any(h.startswith("#ff00") or h == "#0000ff" for h in hexes), "hidden content leaked in"
    r = client.post("/api/generate", json=request(s)).json()
    root, layers = check_svg(r["svg"], 100, 100)
    assert len(layers) == 3                       # navy, red, gold (white = fabric)
    for g in layers:                              # 400 specks gone: a few shapes per color
        assert len(list(g.iter(f"{SVG}path"))) <= 3


def test_aspect_ratio_kept_and_centered(client):
    s = analyze(client, messy_traced_svg(), "trace.svg", w=100, h=50)
    assert (s["width_mm"], s["height_mm"]) == (50, 50)
    check_svg(client.post("/api/generate", json=request(s)).json()["svg"], 50, 50)


def test_larger_than_hoop_is_blocked(client):
    r = client.post("/api/analyze", files={"file": ("logo.png", flat_logo_png())},
                    data={"width_mm": 110, "height_mm": 100})
    assert r.status_code == 400 and "100" in r.json()["detail"]


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
    root, layers = check_svg(r["svg"], 100, 100)
    labels = [g.get(f"{INK}label") for g in layers]
    assert any(l.endswith("White (fabric was white)") for l in labels)
    assert any(w["code"] == "restored" for w in r["warnings"])
    assert r["summary"]["bobbin_name"] == "Black"
    # light thread on dark fabric: underlay + dense rows on tatami shapes
    white_layer = next(g for g in layers if "White" in g.get(f"{INK}label"))
    for p in white_layer.iter(f"{SVG}path"):
        if p.get(f"{STITCH}fill_method") == "auto_fill":
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
    root, layers = check_svg(r["svg"], 100, 100)
    outline = next(g for g in layers if "outline" in g.get(f"{INK}label"))
    assert all(p.get(f"{STITCH}fill_method") == "contour_fill" for p in outline.iter(f"{SVG}path"))


def test_stitch_params_rules():
    from shapely.geometry import Point, box
    from embroidery.vectorize import stitch_params

    assert stitch_params(box(0, 0, 40, 0.8))["fill_method"] == "contour_fill"           # thin
    ring = Point(0, 0).buffer(20).difference(Point(0, 0).buffer(17))
    assert stitch_params(ring)["fill_method"] == "contour_fill"                         # ring
    square = stitch_params(box(0, 0, 30, 20))
    assert square == {"fill_method": "auto_fill", "angle": "45"}                        # never 0/90 on boxy shapes
    from shapely import affinity
    slanted = stitch_params(affinity.rotate(box(0, 0, 40, 10), 30, origin="centroid"))
    assert slanted["angle"] in ("150", "30")
    assert stitch_params(box(0, 0, 30, 20), light_on_dark=True)["row_spacing_mm"] == "0.25"


def test_bad_upload(client):
    r = client.post("/api/analyze", files={"file": ("x.png", b"not an image")})
    assert r.status_code == 422
