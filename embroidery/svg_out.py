"""Step 6 — write the Ink/Stitch-ready SVG."""
from __future__ import annotations

from xml.sax.saxutils import quoteattr

from shapely.geometry import Polygon

NS = {
    "": "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
    "inkstitch": "http://inkstitch.org/namespace",
    "em": "urn:embroidery-magic",
}


def _num(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def path_d(poly: Polygon) -> str:
    parts = []
    for ring in [poly.exterior, *poly.interiors]:
        pts = list(ring.coords)[:-1]
        parts.append("M " + " L ".join(f"{_num(x)} {_num(y)}" for x, y in pts) + " Z")
    return " ".join(parts)


def write_svg(width_mm: float, height_mm: float, layers: list[dict], meta: dict) -> str:
    """layers: [{label, hex, shapes: [(Polygon, {param: value})]}] in stitch order.
    meta: {fabric_hex, fabric_name, bobbin_name, bobbin_hex, threads: [{order, hex, name, status}]}"""
    w, h = _num(width_mm), _num(height_mm)
    ns = " ".join(f'xmlns{":" + k if k else ""}="{v}"' for k, v in NS.items())
    out = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg {ns} version="1.1" width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}">',
        "  <title>Embroidery design</title>",
        "  <metadata>",
        f'    <em:design fabric={quoteattr(meta["fabric_hex"])} fabric-name={quoteattr(meta["fabric_name"])} '
        f'bobbin={quoteattr(meta["bobbin_name"])} bobbin-hex={quoteattr(meta["bobbin_hex"])}>',
    ]
    for t in meta["threads"]:
        order = f' order="{t["order"]}"' if t.get("order") else ""
        out.append(f'      <em:thread{order} hex={quoteattr(t["hex"])} name={quoteattr(t["name"])} '
                   f'status={quoteattr(t["status"])}/>')
    out += [
        "    </em:design>",
        "  </metadata>",
        '  <sodipodi:namedview id="namedview" inkscape:document-units="mm"/>',
    ]
    n = 0
    for li, layer in enumerate(layers, 1):
        out.append(f'  <g inkscape:groupmode="layer" inkscape:label={quoteattr(layer["label"])} id="layer{li}">')
        for poly, params in layer["shapes"]:
            n += 1
            attrs = " ".join(f'inkstitch:{k}={quoteattr(v)}' for k, v in params.items())
            out.append(f'    <path id="path{n}" style="fill:{layer["hex"]};fill-rule:evenodd;stroke:none" '
                       f'd="{path_d(poly)}" {attrs}/>')
        out.append("  </g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"
