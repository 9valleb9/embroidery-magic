# Embroidery Magic

A local web app that turns a logo (PNG, JPG or SVG) into an **Ink/Stitch-ready embroidery SVG** and a verified **.pes for the Brother SE400**: one Inkscape layer per thread color, clean filled shapes, and per-shape stitch settings. It also makes fabric-aware decisions: what to leave unstitched, what will barely show, and which bobbin to use.

## Setup (macOS)

```bash
brew install cairo            # needed by cairosvg for SVG input
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For direct .pes export, install [Inkscape](https://inkscape.org) and the [Ink/Stitch](https://inkstitch.org) extension. The app finds Ink/Stitch in the standard macOS extension folder; set `INKSTITCH_BIN=/path/to/inkstitch` to point it somewhere else. Without Ink/Stitch you still get the SVG, plus steps for exporting the .pes by hand.

## Run

```bash
python app.py
```

This opens <http://127.0.0.1:8765>. Set `PORT=9000` to use a different port, or `NO_BROWSER=1` to skip opening the browser.

Try it with the files in `samples/`: a flat 3-color logo and a messy auto-traced SVG with 400 specks, a hidden layer, an embedded bitmap and a stroke-only shape.

## Using it

1. **Drop an image.** Set the final size (the design is fitted inside the W×H box with its aspect ratio kept) and the number of thread colors. *Auto* suggests the fewest colors that cover every flat area.
2. **Pick the fabric.** Use a preset (White, Black, Navy, Gray, Tan leather, Red) or the color picker. The live preview always renders on the fabric.
3. **Edit the threads.** You can change a color's hex, **Merge into…** another color, toggle **Stitch / Fabric**, and drag rows to set the stitch order. Colors within ΔE2000 < 8 of the fabric are knocked out automatically, and *reset to auto* undoes a manual override.
4. **Generate.** You get a stitched preview, a *Problem areas* view (red = fabric gaps < 0.6 mm, orange = stitched areas < 1 mm wide), warnings with one-click fixes, a summary (fabric, bobbin, thread list) and the SVG download.

**Remove background** (shown when one color fills most of the image's edge, on by default): the region of that color connected to the image edge — found by flood fill — is never stitched, and the design is cropped to fill your size. Areas of the same color *inside* the design stay a normal thread color (e.g. white parts of a logo on navy fabric). Turn it off for full-frame designs like patches.

**Stitch thin lines as outlines** (thread panel, on by default): features under 1 mm wide, such as a logo's outline, can't hold a fill. Their centerline (skeleton) is stitched as a triple (bean) running stitch in the same color: `inkstitch:stroke_method="running_stitch"`, `bean_stitch_repeats="1"`, 2 mm stitches. Line pieces that meet are joined into continuous runs and stitched next to the fills they touch. Short thin bits (corner slivers under 2 mm) stay part of the fill. It usually adds a few hand trims; turn it off to compare.

Changing the fabric only re-runs the **color decisions**. Shapes, palette edits, size and a custom stitch order are kept. Changing size or color count re-analyzes the image.

## Brother SE400

| Constraint | How the app handles it |
|---|---|
| 100×100 mm stitch field | The default size is 100×100 mm, and anything larger is blocked in the UI and the API. The SVG canvas is the hoop (100×100 mm) with the design centered on (50, 50). |
| .pes format | **Export .pes for SE400** runs `inkstitch --extension=output --format=pes` in the background (about 20–60 s; Ink/Stitch is an x86 build running under Rosetta). It then reads the file back with pyembroidery and checks the `#PES` signature, that the stitches fit 100×100 mm, that the longest stitch is ≤ 12.1 mm (Brother limit 12.7), and that each color appears as exactly one block. |
| Single needle | It shows the thread-change count. Colors with ΔE < 5 are merged by default (unmerge them in the thread list), and each color is stitched in one block. A warning appears if an outline or border would bring a thread color back later. |
| 400 stitches/min | Stitch time = stitches ÷ 400. A geometry estimate (about 1.9 stitches/mm²) shows right after Generate; the exact count from the .pes replaces it after export. |
| No auto-trim | Shapes within a color are ordered nearest-neighbor (plus a 2-opt pass), starting where the previous color ended. Every remaining jump inside a color is listed as a **hand trim** with its position and length, and numbered in red on the **Stitch-out** tab. |
| Long runs | A warning appears above about 25,000 stitches. |

A .pes stores colors as entries in Brother's thread palette. At each color stop the SE400 shows the *nearest Brother name* (navy may read "Dark Gray"); the app lists your actual thread next to it.

## Pipeline

| Step | What happens |
|---|---|
| Rasterize | 20 px per mm of final size. For SVG input, `<image>`, `<script>`, `<foreignObject>` and hidden elements (`display:none`, `visibility:hidden`, `opacity:0`) are removed before cairosvg renders. Strokes rasterize as pixels, so they come out as filled outlines. |
| Palette | k-means in CIE Lab on **flat** pixels only (low local variance, so anti-aliased edges are ignored). Every pixel is then snapped to its nearest palette color. |
| Clean up | A 9 px majority filter on the label map (the label-safe form of a median blur: a median over label ids can invent a third color where two meet). Then a per-color morphological open and close with a 0.35 mm disk. Specks under 1 mm² are absorbed into their surroundings, so dropping them never leaves a gap. |
| Vectorize | `cv2.findContours(RETR_CCOMP)` keeps holes. It drops parts under 1 mm² and holes under 0.8 mm², simplifies at 0.04 mm and fixes geometry with `buffer(0)`. Each color grows by 1 px so neighbors overlap slightly instead of leaving hairline seams. |
| Fabric | Fabric-marked colors get no layer; they stay as holes in the neighboring colors. |
| SVG | `width/height` in mm with a mm `viewBox`, inkscape + inkstitch namespaces, one layer per thread named `1 Dark green`, `2 Light green`, … in stitch order. Every shape is `fill-rule:evenodd; stroke:none`. |
| Stitches | **Thin shapes or rings** (area ÷ perimeter < 1.2 mm, or one hole covering more than half the outline) get `inkstitch:fill_method="contour_fill"`. **Everything else** gets `auto_fill` (tatami), with `inkstitch:angle` taken from the minimum rotated rectangle's long axis and rounded to 5°. Shapes whose edges are mostly horizontal or vertical get 45° instead of 0°/90°. **Light thread (L\* > 70) on dark fabric (L\* < 35)** also gets `fill_underlay="true"` and `row_spacing_mm="0.25"` on tatami shapes. Everything else is left at Ink/Stitch defaults. |

Angles use Ink/Stitch's convention: degrees, counter-clockwise from horizontal as seen on screen.

### Fabric rules

All color comparisons use ΔE2000 (`skimage.color.deltaE_ciede2000`).

- **Auto knockout:** a color with ΔE < 8 to the fabric becomes fabric. You can override it per color.
- **Restored knockouts:** a color that was fabric before but no longer matches the new fabric becomes a thread layer named like `White (fabric was white)`, with a warning.
- **Low contrast:** a stitched color with ΔE < 15 gets a warning plus two fixes: *swap thread color* (same hue, lightness pushed away from the fabric) or *add a 1 mm outline* (a contour-fill layer in black or white, whichever contrasts more with the fabric).
- **Design edge:** if the outermost color is close to the fabric, the app suggests a thin 1 mm border around the design silhouette.
- **Suggested stitch order:** the largest area goes first. The rest follow with colors touching the fabric and larger areas first, and lighter colors ahead of darker ones wherever two colors touch.
- **Bobbin:** white or black, whichever is closer to the fabric.

The fabric color, bobbin and thread list (fabric colors marked *not stitched — fabric*) are written to the SVG's `<metadata>` (`urn:embroidery-magic` namespace) and shown in the UI.

### Warnings

- Stitched areas narrower than 1 mm.
- Fabric gaps narrower than 0.6 mm, which will likely close up (shown in red on the preview).
- More than 8 colors.
- Photo or gradient input.
- Low contrast, design edge, restored knockouts, and the light-on-dark adjustments.

## Tests

```bash
pytest                      # includes a real Ink/Stitch export when installed (~20–60 s)
pytest -m "not inkstitch"   # fast run
```

The tests check that the output SVG parses, has the right mm dimensions and `viewBox`, and that every path has an inkstitch fill setting. They also cover the flat-logo and messy-trace fixtures (a few clean shapes per color), aspect ratio, the fabric restore flow, and the low-contrast warning with its outline fix, plus the stitch-setting rules.

## Layout

```
app.py                  FastAPI server + `python app.py` entry point
embroidery/raster.py    rasterize + SVG sanitizing
embroidery/palette.py   k-means palette, label-map cleanup, photo detection
embroidery/vectorize.py contours -> polygons, stitch settings, outline bands
embroidery/colors.py    Lab / ΔE2000, thread names, bobbin, contrast swaps
embroidery/pipeline.py  sessions, fabric decisions, stitch order, generation, preview + gap checks
embroidery/svg_out.py   Ink/Stitch SVG writer
embroidery/export.py    SE400: Ink/Stitch .pes export, pyembroidery verification, hand-trim list
static/index.html       single-page UI (no build step)
tests/                  pytest + synthetic fixtures
samples/                example inputs
```

## License

MIT — see [LICENSE](LICENSE).
