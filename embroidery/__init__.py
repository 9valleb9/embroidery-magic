"""Embroidery Magic: turn a logo image into an Ink/Stitch SVG and a machine embroidery file.

The web server (``app.py``) is a thin FastAPI layer over this package. A design goes
through these modules, in this order:

1. ``raster``: rasterize the upload (PNG, JPG or a sanitized SVG) at a fixed
   pixels-per-mm of the final design size.
2. ``palette``: k-means palette in CIE Lab on flat pixels, a label map of every pixel's
   nearest palette color, label-map clean-up and photo/gradient detection.
3. ``vectorize``: label masks to Shapely polygons (holes kept), stitch settings per shape
   (tatami or contour fill, angle, density, underlay, pull compensation), thin-line
   outlines, outline bands and satin columns.
4. ``pipeline``: analysis sessions (geometry that is traced once), fabric-aware color
   decisions (knockouts, merges, warnings, stitch order), shape ordering, previews and
   gap checks, and generation of the final SVG plus a machine summary.
5. ``svg_out``: writes the Ink/Stitch-ready SVG (mm units, one layer per thread, the
   ``urn:embroidery-magic`` metadata block and the ``inkstitch_svg_version`` stamp).
6. ``export``: runs the Ink/Stitch CLI in a background job to write the machine file
   (.pes, .jef, .vp3, .exp, .xxx, .dst), then reads it back with pyembroidery to check
   it and to draw the stitch-out preview with the hand-trim map.

Supporting modules used throughout:

- ``colors``: hex/Lab conversion, ΔE2000 and the thresholds built on it, thread layer
  names, fabric presets, bobbin choice and contrast fixes.
- ``machines``: file formats by brand, standard hoop sizes, hoop fit and orientation.
- ``fabrics``: fabric-type profiles (pull compensation, row spacing, underlay,
  stabilizer, needle, notes and limits).

Units: sizes and coordinates in the SVG and in the public settings are millimetres
(raster work happens in pixels at ``raster.PX_PER_MM``). pyembroidery coordinates are in
0.1 mm and are converted back to mm in ``export``.
"""
