"""Step 6 — write the Ink/Stitch-ready SVG.

Where it sits: called by ``pipeline.generate`` once layers are decided, ordered and
placed on the hoop. The returned SVG text is offered as a download and is the input to
``export`` (the Ink/Stitch CLI turns it into a machine file).

What the SVG looks like:

- ``width``/``height`` in mm and a matching mm ``viewBox``, so 1 user unit = 1 mm. The
  canvas is the hoop's stitch field (``pipeline`` centers the design on it).
- Namespaces for SVG, Inkscape, Sodipodi, Ink/Stitch, plus ``em`` =
  ``urn:embroidery-magic`` for this app's own metadata.
- ``<metadata>`` holds an ``<em:design>`` element (fabric, bobbin, fabric type,
  stabilizer, needle) with one ``<em:thread>`` per stitched thread and per color left
  as fabric (no ``order``, status "not stitched"), and the
  ``<inkstitch:inkstitch_svg_version>`` stamp (see :data:`INKSTITCH_SVG_VERSION`).
- One Inkscape layer (``<g inkscape:groupmode="layer">``) per thread, in stitch order.
  Ink/Stitch stitches in document order, so layer order is stitch order.
- Each shape is a ``<path>`` with its stitch settings as ``inkstitch:*`` attributes.

Main entry point: :func:`write_svg`.

Gotcha: Ink/Stitch parameter values are strings; ``vectorize`` already formats them
(``"0.25"``, ``"true"``). Parameter keys starting with ``_`` are private to this app and
are not written as attributes.
"""
from __future__ import annotations

from xml.sax.saxutils import quoteattr

from shapely.geometry import Polygon

# XML namespaces declared on the root <svg>; "" is the default (SVG) namespace.
NS = {
    "": "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
    "inkstitch": "http://inkstitch.org/namespace",
    "em": "urn:embroidery-magic",
}


INKSTITCH_SVG_VERSION = 4   # Ink/Stitch 3.x


def _num(v: float) -> str:
    """Format a coordinate with at most 3 decimals (0.001 mm), trailing zeros and ``-0`` dropped."""
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def path_d(poly: Polygon) -> str:
    """SVG path data for a polygon: one closed subpath for the exterior and one per hole.

    Holes only render as holes because the path is drawn with ``fill-rule:evenodd``
    (ring direction doesn't matter).

    Args:
        poly: A Shapely polygon in mm.

    Returns:
        The ``d`` attribute value, e.g. ``"M 0 0 L 10 0 L 10 10 Z M 2 2 L 4 2 L 4 4 Z"``.
    """
    parts = []
    for ring in [poly.exterior, *poly.interiors]:
        pts = list(ring.coords)[:-1]   # Shapely repeats the first point at the end; Z closes it instead
        parts.append("M " + " L ".join(f"{_num(x)} {_num(y)}" for x, y in pts) + " Z")
    return " ".join(parts)


def write_svg(width_mm: float, height_mm: float, layers: list[dict], meta: dict) -> str:
    """layers: [{label, hex, shapes: [(Polygon | LineString, {param: value})]}] in stitch order.
    meta: {fabric_hex, fabric_name, bobbin_name, bobbin_hex, threads: [{order, hex, name, status}],
    optional fabric_type, stabilizer, needle}

    Builds the whole document as text (no XML library), escaping attribute values with
    ``quoteattr``. Each shape becomes one of three kinds of path:

    - a satin column, when ``params`` has ``"_satin"`` (a ``vectorize.Satin``): one path
      with two subpaths, the two rails, stroked in the thread color. The geometry in the
      tuple is ignored in this case;
    - a running-stitch line, when the geometry is a ``LineString``: a stroked centerline;
    - a fill, otherwise: a filled ``evenodd`` path from :func:`path_d`.

    Args:
        width_mm: Canvas width in mm (the oriented hoop width).
        height_mm: Canvas height in mm.
        layers: Thread layers in stitch order. ``label`` is the Inkscape layer name
            (``"1 Dark green"``), ``hex`` the thread color, ``shapes`` the geometry and
            Ink/Stitch params. Params whose key starts with ``_`` are skipped as
            attributes. Extra layer keys are ignored.
        meta: Design metadata written into ``<em:design>``. A thread's ``order`` is
            omitted when falsy (fabric colors are listed but not stitched).

    Returns:
        The SVG document as a string, ending with a newline.
    """
    w, h = _num(width_mm), _num(height_mm)
    ns = " ".join(f'xmlns{":" + k if k else ""}="{v}"' for k, v in NS.items())
    out = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg {ns} version="1.1" width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}">',
        "  <title>Embroidery design</title>",
        "  <metadata>",
        f'    <em:design fabric={quoteattr(meta["fabric_hex"])} fabric-name={quoteattr(meta["fabric_name"])} '
        f'bobbin={quoteattr(meta["bobbin_name"])} bobbin-hex={quoteattr(meta["bobbin_hex"])}'
        + "".join(f' {k.replace("_", "-")}={quoteattr(meta[k])}' for k in ("fabric_type", "stabilizer", "needle") if meta.get(k))
        + '>',
    ]
    for t in meta["threads"]:
        order = f' order="{t["order"]}"' if t.get("order") else ""
        out.append(f'      <em:thread{order} hex={quoteattr(t["hex"])} name={quoteattr(t["name"])} '
                   f'status={quoteattr(t["status"])}/>')
    out += [
        "    </em:design>",
        # Current Ink/Stitch param names. Without this stamp Ink/Stitch treats the file as
        # legacy: it rewrites params and asks (in a dialog) whether to update it.
        f"    <inkstitch:inkstitch_svg_version>{INKSTITCH_SVG_VERSION}</inkstitch:inkstitch_svg_version>",
        "  </metadata>",
        '  <sodipodi:namedview id="namedview" inkscape:document-units="mm"/>',
    ]
    n = 0   # path ids are numbered across all layers so they stay unique in the document
    for li, layer in enumerate(layers, 1):
        out.append(f'  <g inkscape:groupmode="layer" inkscape:label={quoteattr(layer["label"])} id="layer{li}">')
        for geom, params in layer["shapes"]:
            n += 1
            attrs = " ".join(f'inkstitch:{k}={quoteattr(v)}' for k, v in params.items() if not k.startswith("_"))
            if "_satin" in params:               # satin column: two rails, nodes paired one to one
                sat = params["_satin"]
                d = " ".join("M " + " L ".join(f"{_num(x)} {_num(y)}" for x, y in rail) for rail in (sat.rail1, sat.rail2))
                out.append(f'    <path id="path{n}" style="fill:none;stroke:{layer["hex"]};stroke-width:0.3" '
                           f'd="{d}" {attrs}/>')
            elif geom.geom_type == "LineString":   # outline mode: running-stitch centerline
                d = "M " + " L ".join(f"{_num(x)} {_num(y)}" for x, y in geom.coords)
                out.append(f'    <path id="path{n}" style="fill:none;stroke:{layer["hex"]};stroke-width:0.3" '
                           f'd="{d}" {attrs}/>')
            else:
                out.append(f'    <path id="path{n}" style="fill:{layer["hex"]};fill-rule:evenodd;stroke:none" '
                           f'd="{path_d(geom)}" {attrs}/>')
        out.append("  </g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"
