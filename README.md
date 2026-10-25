<img src="static/logo.svg" alt="Embroidery Magic logo" width="96">

# Embroidery Magic

[![tests](https://github.com/9valleb9/embroidery-magic/actions/workflows/test.yml/badge.svg)](https://github.com/9valleb9/embroidery-magic/actions/workflows/test.yml)

**Turn a logo into a machine embroidery file — free, open source, runs on your computer.**

Drop in a PNG, JPG or SVG logo, pick your machine, hoop and fabric, and get a stitch file you can put on a USB stick: **.pes** (Brother, Baby Lock), **.jef** (Janome, Elna), **.vp3** (Husqvarna Viking, Pfaff), **.exp** (Bernina), **.xxx** (Singer) or **.dst**. Every file is read back and checked before you download it.

![A flat logo, its stitch plan, and the stitch-out preview read back from the exported .pes](docs/hero.png)

<!-- TODO: add a photo of a real stitch-out (docs/stitched.jpg) -->

Made for home single-needle machines, where the auto-digitizers get the small stuff wrong:

- **Fabric-aware.** Pick the fabric type (cotton, knit, fleece, beanie, performance polyester, towel, leather…) and the stitch settings follow, with the stabilizer and needle to use. Colors that match the fabric aren't stitched. Low-contrast threads get flagged, with a one-click swap or outline. White-on-dark gets underlay so it doesn't sink in. The bobbin color is suggested.
- **Fewest thread changes.** Near-identical colors are merged, every color is stitched in one go, and you get the change list with the name your machine's screen will show.
- **A trim map.** Most home machines don't cut the jumps between shapes. You get every jump you'll snip by hand, numbered on a full-hoop stitch-out preview.
- **Problem areas before you stitch.** Gaps that will close up are marked red, slivers too thin to fill are marked orange, and thin outlines become clean triple running stitches. Script letters and swashes can stitch as satin columns.
- **Background removal.** A solid background behind the logo is dropped, and the design is cropped to fill your size.

## Try it

**With Docker** (includes Ink/Stitch, so machine-file export just works):

```bash
docker run --rm -p 8765:8765 $(docker build -q https://github.com/9valleb9/embroidery-magic.git)
```

Then open <http://localhost:8765>.

**With Python** (macOS shown, see [Requirements](#requirements) for Linux and Windows):

```bash
# cairo is needed by cairosvg for SVG input
brew install cairo
git clone https://github.com/9valleb9/embroidery-magic.git && cd embroidery-magic
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
# opens http://127.0.0.1:8765
python app.py
```

Next time, just `cd embroidery-magic && source .venv/bin/activate && python app.py`. After `git pull`, run `pip install -r requirements.txt` again and hard-reload the page.

## Requirements

| What | Version | Needed for |
|---|---|---|
| Python | 3.11 or newer (tested on 3.11 and 3.13) | Everything |
| Python packages | `pip install -r requirements.txt`: FastAPI, uvicorn, OpenCV (headless), Shapely 2, NumPy, Pillow, CairoSVG, scikit-learn, scikit-image, SciPy, pyembroidery, plus pytest/httpx for the tests | Everything |
| Cairo library | macOS `brew install cairo` · Debian/Ubuntu `sudo apt install libcairo2` · Windows: the GTK runtime (or use Docker) | SVG input (PNG/JPG work without it) |
| [Ink/Stitch](https://inkstitch.org/docs/install/) | 3.3.0 (3.x needed) | Exporting .pes/.jef/.vp3/.exp/.xxx/.dst directly. Without it you get the Ink/Stitch SVG and steps to export by hand in Inkscape |
| Xvfb | Linux only: `sudo apt install xvfb`, then point `INKSTITCH_BIN` at a wrapper that runs `xvfb-run -a .../inkstitch "$@"` | Ink/Stitch needs a display even when exporting from the command line |
| A web browser | Any current one | The UI |

Ink/Stitch is found in the standard Inkscape extension folders on macOS and Linux, or on `PATH`. On Windows, set `INKSTITCH_BIN` to its `inkstitch.exe`. Docker includes everything above.

Settings (environment variables):

| Variable | Default | |
|---|---|---|
| `INKSTITCH_BIN` | auto-detected | Path to the `inkstitch` executable |
| `PORT` | `8765` | Port to serve on |
| `HOST` | `127.0.0.1` | Use `0.0.0.0` to reach it from another device |
| `NO_BROWSER` | unset | Set to `1` to not open a browser tab on start |

Sample inputs are in `samples/`: the mountain badge above, a flat 3-color logo, and a messy auto-traced SVG with 400 specks, a hidden layer, an embedded bitmap and a stroke-only shape.

## Using it

![The app: machine and hoop on the left, thread list in the middle, stitch-out preview read back from the .pes on the right](docs/screenshot.jpg)

1. **Pick your machine and drop an image.** Set the format, hoop, **design final size** and the number of thread colors. The design is fitted inside the W×H box with its aspect ratio kept and centered in the hoop; the line under the box shows its real size (e.g. *99 × 38.45 mm · hoop 100 × 100 mm*), and the live preview draws it inside a dashed hoop outline. *Auto* suggests the fewest colors that cover every flat area.
2. **Pick the fabric type and color.** The type sets the stitch settings (see [Fabric types](#fabric-types)). For the color, use a preset (White, Black, Navy, Gray, Tan leather, Red) or the color picker. The live preview always renders on the fabric color.
3. **Edit the threads.** You can change a color's hex, **Merge into…** another color, toggle **Stitch / Fabric**, and drag rows to set the stitch order. Colors within ΔE2000 < 8 of the fabric are knocked out automatically, and *reset to auto* undoes a manual override.
4. **Generate.** You get a stitched preview, a *Problem areas* view (red = fabric gaps < 0.6 mm, orange = stitched areas < 1 mm wide), warnings with one-click fixes, a **Before you stitch** box (stabilizer, needle, topping, the settings the fabric applied, and notes about this design), a summary (fabric, bobbin, thread list) and the SVG download.

**Remove background** (shown when one color fills most of the image's edge, on by default): the region of that color connected to the image edge — found by flood fill — is never stitched, and the design is cropped to fill your size. Areas of the same color *inside* the design stay a normal thread color (e.g. white parts of a logo on navy fabric). Turn it off for full-frame designs like patches.

**Stitch thin lines as outlines** (thread panel, on by default): features under 1 mm wide, such as a logo's outline, can't hold a fill. Their centerline (skeleton) is stitched as a triple (bean) running stitch in the same color: `inkstitch:stroke_method="running_stitch"`, `bean_stitch_repeats="1"`, 2 mm stitches. Line pieces that meet are joined into continuous runs and stitched next to the fills they touch. Short thin bits (corner slivers under 2 mm) stay part of the fill. It usually adds a few hand trims; turn it off to compare.

**Stitch narrow strokes as satin** (thread panel, off by default): parts of a shape that are a steady 1–7 mm wide, like script letters, swashes and bars, are stitched as satin columns instead of fills. Each column follows the shape's centerline (its skeleton): pieces that continue each other through a junction (turning less than 50°) form one column, and sharp corners start a new one. The column's two rails sit on the stroke's edges with their nodes paired one to one, so Ink/Stitch sews straight across the stroke. Wider areas, blobs and junction centers stay fills, reaching 0.3 mm under the columns so no fabric shows where they meet. Hairlines under 1 mm (what *Stitch thin lines as outlines* would sew as a triple running stitch) become 1 mm satin columns along their centerline instead, so they match the satin lettering rather than reading as loose threads. That makes them slightly bolder than in the artwork. Satin columns get center-walk underlay, contour underlay from 2 mm wide, and zig-zag underlay from 3 mm on fabrics that use underlay. The zig-zag spacing follows the fabric's density (0.4 mm at standard). Satin usually means more stitches and a few more hand trims than fills: on a script logo at 175 mm on cotton, about 10,700 stitches and 24 trims, against 7,100 and 15 as fills.

**Overlap & pull compensation** (thread panel, three sliders, remembered between visits):

- **Color overlap** (default 0.3 mm): each color reaches this far under the colors it touches, and fills fabric gaps narrower than twice this, so neighbors overlap instead of leaving slivers of fabric between them. It never grows the design's outer edge, and it shows up in the previews and the gap check.
- **Pull compensation** (set by the fabric type, max 0.5): written as Ink/Stitch's `pull_compensation_mm` on tatami fills. Each row of stitches is stretched this far past the shape's edge, because stitching pulls the fabric in. Moving the slider marks it *custom*; *reset to fabric* puts the fabric's value back, and picking a fabric type always does.
- **Absorb slivers under** (default 0.35 mm): color bands narrower than this, such as anti-aliased rims around a shape or a hairline between two colors, are merged into the neighboring color during clean-up. Changing it re-analyzes the image. Set it too high and real thin lines disappear too.

### Fabric types

Picking a fabric type loads its pull compensation into the slider and writes its fill density (`row_spacing_mm`) onto every fill and its underlay (`fill_underlay`) onto tatami fills (Ink/Stitch's contour fill has no underlay). *Standard* density is Ink/Stitch's default 0.25 mm rows. The stabilizer, needle and extras appear in **Before you stitch** and are written to the SVG's metadata. These are common starting points, not measured stitch-outs: test on a scrap of the same fabric.

| Fabric type | Pull comp | Fill density | Underlay | Stabilizer · needle · extras |
|---|---|---|---|---|
| Cotton / woven (default) | 0.2 mm | standard | off | tear-away · 75/11 sharp |
| Denim / canvas | 0.2 mm | standard | on | heavy tear-away · 90/14 jeans |
| T-shirt knit | 0.35 mm | lighter (0.3 mm) | on | cut-away · 75/11 ballpoint |
| Polo / piqué | 0.35 mm | standard | on | cut-away · 75/11 ballpoint · water-soluble topping |
| Fleece / sweatshirt | 0.4 mm | standard | on | cut-away · 75/11 ballpoint · water-soluble topping |
| Beanie / knit hat (acrylic, polyester) | 0.4 mm | standard | on | no-show mesh cut-away · 75/11 ballpoint · water-soluble topping · warns above 55 mm tall (a folded cuff) |
| Performance polyester (athletic, jerseys) | 0.3 mm | lighter (0.3 mm) | on | no-show mesh cut-away · 75/11 ballpoint · warns above 8,000 stitches |
| Towel / terry | 0.3 mm | denser (0.22 mm) | on | tear-away · 75/11 sharp · water-soluble topping |
| Felt | 0.15 mm | lighter (0.3 mm) | off | tear-away or none · 75/11 sharp |
| Leather / vinyl | 0.15 mm | lighter (0.3 mm) | off | tear-away · leather needle · no pins inside the design |

The light-on-dark rule still applies on top: light thread on dark fabric always gets underlay and rows no lighter than 0.25 mm. The stitch estimate follows the density. **Before you stitch** also counts the thin features that will stitch as outlines and, when the hoop has room, says how big the design can go so more of them hold as fills.

Changing the fabric color only re-runs the **color decisions**. Shapes, palette edits, size and a custom stitch order are kept. Changing size or color count re-analyzes the image.

## Machines

Pick the **file format** for your brand, your **hoop** (4×4, 5×7, 6×10, 8×8 or a custom size) and your machine's **speed** (for the time estimate). These are remembered between visits.

| Constraint | How the app handles it |
|---|---|
| Hoop size | The design must fit the hoop's stitch field, either way round, with 0.5 mm kept free at each edge because pull compensation stitches slightly past the design's edge (so 99 × 99 mm max on a 4×4). The SVG canvas is the hoop, turned to match the design, with the design centered on it. |
| File format | **Export** runs `inkstitch --extension=output --format=<fmt>` in the background (about 20–60 s). It then reads the file back with pyembroidery and checks the signature (.pes, .dst), that the stitches fit the hoop, that the longest stitch is ≤ 12.1 mm, and that each color is exactly one block. |
| Single needle | It shows the thread-change count. Colors with ΔE < 5 are merged by default (unmerge them in the thread list), and each color is stitched in one block. A warning appears if an outline or border would bring a thread color back later. |
| Stitch time | Stitches ÷ your machine's speed (default 400 spm, the Brother SE400). A geometry estimate (about 1.9 stitches/mm²) shows right after Generate; the exact count from the file replaces it after export. |
| Jump trims | Shapes within a color are ordered nearest-neighbor (plus a 2-opt pass), starting where the previous color ended. Every remaining jump inside a color is listed with its position and length, and numbered in red on the **Stitch-out** tab. |
| Long runs | A warning appears above about 25,000 stitches. |

.pes and .jef store colors as entries in Brother's or Janome's thread palette, so the machine's screen shows the *nearest palette name* at each color stop (navy may read "Dark Gray"). The app lists your actual thread next to it. .dst stores no colors at all; the machine just stops, and you load threads in the listed order.

Built around the Brother SE400 (4×4 hoop, no jump trimming). Every format is verified by reading the exported file back, not by stitching it on each brand's machine. If you stitch one out, please open an issue with how it went (photos welcome).

## Pipeline

| Step | What happens |
|---|---|
| Rasterize | 20 px per mm of final size. For SVG input, `<image>`, `<script>`, `<foreignObject>` and hidden elements (`display:none`, `visibility:hidden`, `opacity:0`) are removed before cairosvg renders. Strokes rasterize as pixels, so they come out as filled outlines. |
| Palette | k-means in CIE Lab on **flat** pixels only (low local variance, so anti-aliased edges are ignored). Every pixel is then snapped to its nearest palette color. |
| Clean up | A 9 px majority filter on the label map (the label-safe form of a median blur: a median over label ids can invent a third color where two meet). Then a per-color morphological open and close with a disk the size of the sliver setting (default 0.35 mm). Specks under 1 mm² are absorbed into their surroundings, so dropping them never leaves a gap. |
| Vectorize | `cv2.findContours(RETR_CCOMP)` keeps holes. It drops parts under 1 mm² and holes under 0.8 mm², simplifies at 0.04 mm and fixes geometry with `buffer(0)`. Each color grows by 1 px so neighbors overlap slightly instead of leaving hairline seams. |
| Fabric | Fabric-marked colors get no layer; they stay as holes in the neighboring colors. |
| SVG | `width/height` in mm with a mm `viewBox`, inkscape + inkstitch namespaces, stamped `inkstitch_svg_version` 4 (Ink/Stitch 3.x parameter names; without it Ink/Stitch treats the file as legacy, rewrites its parameters and asks whether to update it), one layer per thread named `1 Dark green`, `2 Light green`, … in stitch order. Every shape is `fill-rule:evenodd; stroke:none`. |
| Stitches | **Thin shapes or rings** (area ÷ perimeter < 1.2 mm, or one hole covering more than half the outline) get `inkstitch:fill_method="contour_fill"`. **Everything else** gets `tatami_fill`, with `inkstitch:angle` taken from the minimum rotated rectangle's long axis and rounded to 5°. Shapes whose edges are mostly horizontal or vertical get 45° instead of 0°/90°. All fills get the fabric type's `row_spacing_mm`; tatami shapes also get its `fill_underlay` and the pull compensation setting. With satin on, steady 1–7 mm strokes become `inkstitch:satin_column="true"` paths (two rails, equal node counts) with `zigzag_spacing_mm`, underlay and pull compensation, and the rest of the shape is filled as above. **Light thread (L\* > 70) on dark fabric (L\* < 35)** always gets `fill_underlay="true"` and rows no lighter than `row_spacing_mm="0.25"`. Everything else is left at Ink/Stitch defaults. |

Angles use Ink/Stitch's convention: degrees, counter-clockwise from horizontal as seen on screen.

### Fabric rules

All color comparisons use ΔE2000 (`skimage.color.deltaE_ciede2000`).

- **Auto knockout:** a color with ΔE < 8 to the fabric becomes fabric. You can override it per color.
- **Restored knockouts:** a color that was fabric before but no longer matches the new fabric becomes a thread layer named like `White (fabric was white)`, with a warning.
- **Low contrast:** a stitched color with ΔE < 15 gets a warning plus two fixes: *swap thread color* (same hue, lightness pushed away from the fabric) or *add a 1 mm outline* (a contour-fill layer in black or white, whichever contrasts more with the fabric).
- **Design edge:** if the outermost color is close to the fabric, the app suggests a thin 1 mm border around the design silhouette.
- **Suggested stitch order:** the largest area goes first. The rest follow with colors touching the fabric and larger areas first, and lighter colors ahead of darker ones wherever two colors touch.
- **Bobbin:** white or black, whichever is closer to the fabric.

The fabric color, fabric type, stabilizer, needle, bobbin and thread list (fabric colors marked *not stitched — fabric*) are written to the SVG's `<metadata>` (`urn:embroidery-magic` namespace) and shown in the UI.

### Warnings

- Stitched areas narrower than 1 mm.
- Fabric gaps narrower than 0.6 mm, which will likely close up (shown in red on the preview).
- More than 8 colors.
- Photo or gradient input.
- Low contrast, design edge, restored knockouts, and the light-on-dark adjustments.
- Fabric limits: a beanie design taller than 55 mm, or more than 8,000 stitches on performance polyester.

## Tests

```bash
pytest                      # includes real Ink/Stitch exports when installed (a few minutes)
pytest -m "not inkstitch"   # fast run
```

The tests check that the output SVG parses, has the right mm dimensions and `viewBox`, and that every path has an inkstitch fill setting. They also cover the flat-logo and messy-trace fixtures (a few clean shapes per color), aspect ratio, the fabric restore flow, and the low-contrast warning with its outline fix, plus the stitch-setting rules and the fabric types (density, underlay, metadata, beanie and polyester limits), plus satin columns (rail pairing, what stays a fill, a real Ink/Stitch export).

## Layout

```
app.py                  FastAPI server + `python app.py` entry point
embroidery/raster.py    rasterize + SVG sanitizing
embroidery/palette.py   k-means palette, label-map cleanup, photo detection
embroidery/vectorize.py contours -> polygons, stitch settings, outline bands, satin columns
embroidery/colors.py    Lab / ΔE2000, thread names, bobbin, contrast swaps
embroidery/pipeline.py  sessions, fabric decisions, stitch order, generation, preview + gap checks
embroidery/svg_out.py   Ink/Stitch SVG writer
embroidery/machines.py  file formats by brand, hoop sizes, fit + orientation
embroidery/fabrics.py   fabric types: stitch settings, stabilizer, needle, notes
embroidery/export.py    Ink/Stitch machine-file export, pyembroidery verification, jump-trim list
static/index.html       single-page UI (no build step)
static/logo.svg         app logo (header, browser tab, README)
Dockerfile              app + Ink/Stitch in one image
tests/                  pytest + synthetic fixtures
samples/                example inputs
```

## License

MIT — see [LICENSE](LICENSE).
