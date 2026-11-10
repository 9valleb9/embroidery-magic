"""Shape fixes for lettering: bolder strokes, dot repair, corner tips, and the checks
that keep letters readable when they change.

Where it sits: ``pipeline._build_geometry`` calls :func:`apply` on the merged label map
before tracing, so every fix shows up in the fills, satins, preview and checks alike.
Nothing here knows about thread, fabric or Ink/Stitch: it takes a label map (one id per
thread color, ``TRANSPARENT`` for empty) and returns an edited one plus notes for the UI.

Why: a logo traced from art made for vinyl or print stitches thinner than it looks,
because satin columns pull in and knit fabric swallows thin strokes. Growing the shapes
fixes that but breaks other things: letters merge, counters (the eye of an "e") close,
the insides of "n", "m" and "h" seal off, neighboring serifs fuse into a bar, flared
ends balloon into blobs, and small dots crowd their stems. Each of those is handled
here, automatically.

How bolder works (:func:`_grow`):

- Every separate shape (a letter, a dot, a swash) grows only into its own territory:
  background pixels nearer to it than to any other shape, keeping ``min_gap_mm`` from
  the midline between them. Letters can't merge, and a gap already narrower than the
  minimum is left as it is rather than closed further.
- Pockets of a shape (concave spaces: the inside of an "n", the notch between two
  serifs) and its holes (counters) keep a channel ``min_inside_mm`` wide along their
  center line. A counter already narrower than that gets a slot of that width carved
  along its center line.
- A hole that appears where there was none (two feet growing together under an "n")
  is reopened with a straight channel down past the baseline.
- End caps that grow wider than ``CAP_GROWTH`` times their stroke become rounded
  rectangles as tall as the stroke.
- Before growing, a letter whose inside is already narrower than ``min_inside_mm`` is
  widened (:func:`_spread_narrow`), moving the letters after it.

Other fixes: :func:`_repair_dots` redraws a small dot above a stem (the dot of an "i")
as a circle when it is a dash or crowds its stem, and :func:`_fill_corner_tips` closes
the points of sharp inside corners without merging shapes or touching counters.

Units: the label map is at ``PX_PER_MM``; all public sizes are mm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .palette import TRANSPARENT
from .raster import PX_PER_MM

COUNTER_MIN_MM2 = 0.4  # enclosed spaces smaller than this are specks where strokes cross, not counters
POCKET_DEPTH_MM = 0.8  # pockets shallower than this are inside corners, not spaces to keep open
POCKET_MM = 3.0        # concave spaces narrower than this belong to their shape (closing disk)
CAP_GROWTH = 1.35      # an end cap this many times wider than its stroke after growing is rebuilt
CAP_RADIUS_MM = 0.3    # corner radius of a rebuilt end cap
DOT_MAX_MM = 3.5       # a separate shape up to this size, above a stem, is a dot
DOT_CLEARANCE_MM = 1.0 # a dot keeps this much space from everything else
DOT_DASH_ASPECT = 1.5  # a dot this many times wider than tall is a dash, redrawn as a circle
SPECK_MM2 = 0.15       # pieces smaller than this left by the edits are removed
TIP_FILL_MM = 0.6      # corner tips narrower than this are filled (closing disk)


@dataclass(frozen=True)
class Fixes:
    """Requested shape fixes. Frozen so it can be part of the geometry cache key.

    Attributes:
        bolder_mm: Grow every shape outward by this much (mm per side).
        bolder_pct: Or by this percentage of the median stroke width (total width).
            Used when ``bolder_mm`` is 0.
        repair_dots: Redraw dashes and crowded dots above stems as round dots.
        fill_corner_tips: Fill the points of sharp inside corners.
        open_gaps: Widen gaps narrower than ``min_gap_mm`` (between letters, inside
            counters, between serifs) to ``min_gap_mm`` by trimming the shapes beside them.
        min_gap_mm: Gap kept between separate shapes while growing.
        min_inside_mm: Channel kept through counters and pockets while growing.
    """
    bolder_mm: float = 0.0
    bolder_pct: float = 0.0
    repair_dots: bool = True
    fill_corner_tips: bool = False
    open_gaps: bool = False
    min_gap_mm: float = 0.8
    min_inside_mm: float = 1.0


@dataclass
class Report:
    """What :func:`apply` did, for the UI. ``notes`` are info lines, ``warnings`` problems."""
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stroke: dict | None = None          # {min, median, max} stroke width in mm, before fixes
    grow_mm: float = 0.0


def disk(mm: float) -> np.ndarray:
    """Elliptical structuring element ``mm`` across (odd, at least 3 px)."""
    d = max(3, round(mm * PX_PER_MM) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def stroke_widths(mask: np.ndarray) -> dict | None:
    """Stroke width in mm (2 x distance to the edge along the skeleton): 5th percentile,
    median and 95th percentile, reported as min / median / max. Percentiles keep stroke
    tips (where the width runs to zero) and junctions out of the extremes.

    Returns:
        ``{"min", "median", "max"}`` rounded to 0.05 mm, or None for an empty mask.
    """
    from skimage.morphology import skeletonize

    if not mask.any():
        return None
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    w = 2 * dist[skeletonize(mask)] / PX_PER_MM
    w = w[w > 1.5 / PX_PER_MM]
    if not len(w):
        return None
    r = lambda v: round(float(v) * 20) / 20
    return {"min": r(np.percentile(w, 5)), "median": r(np.median(w)), "max": r(np.percentile(w, 95))}


def _holes(comp: np.ndarray) -> np.ndarray:
    """Background pixels enclosed by ``comp`` (its counters), as a bool mask."""
    inv = (~comp).astype(np.uint8)
    n, cc = cv2.connectedComponents(inv, connectivity=4)
    border = set(np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]])).tolist())
    return (cc > 0) & ~np.isin(cc, list(border))


def _count_holes(comp: np.ndarray) -> int:
    """Number of separate counters in ``comp``."""
    h = _holes(comp)
    return cv2.connectedComponents(h.astype(np.uint8), connectivity=4)[0] - 1 if h.any() else 0


def _channels(comp: np.ndarray, width_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Where one shape may not grow, so its pockets and counters stay open.

    Pockets are the concave spaces a ``POCKET_MM`` closing fills (the inside of an "n",
    the notch between two serifs) that reach at least ``POCKET_DEPTH_MM`` in from the
    closed outline; shallower ones are inside corners, which growth may round off.
    Holes are counters. Inside them, a pixel at distance
    ``d`` from the shape is protected when ``d`` is within ``width_mm / 2`` of the
    space's local half-width (the largest such distance within ``POCKET_MM / 2``), so
    growing leaves a channel at least ``width_mm`` wide down the middle of every space
    that had one. Growth on the shape's outer edge (past a pocket's mouth) is untouched.

    Returns:
        (protected mask the shape must not grow into, slot mask to carve out of the
        shape where a counter is already narrower than ``width_mm``).
    """
    from skimage.morphology import skeletonize

    c8 = comp.astype(np.uint8)
    closed = cv2.morphologyEx(c8, cv2.MORPH_CLOSE, disk(POCKET_MM)).astype(bool)
    holes = _holes(comp)
    pockets = closed & ~comp & ~holes
    # a real pocket is deep; the wedge in the inside corner where a serif meets a stem is
    # a "pocket" of the closing too, but shallow, and growth should round it off
    depth = cv2.distanceTransform(closed.astype(np.uint8), cv2.DIST_L2, 5)
    n, cc = cv2.connectedComponents(pockets.astype(np.uint8), connectivity=8)
    if n > 1:
        deep = np.zeros(n, bool)
        np.maximum.at(deep, cc[pockets], depth[pockets] >= POCKET_DEPTH_MM * PX_PER_MM)
        deep[0] = False
        pockets = deep[cc]
    inside = pockets | holes
    if not inside.any():
        z = np.zeros_like(comp)
        return z, z
    d = cv2.distanceTransform((~comp).astype(np.uint8), cv2.DIST_L2, 5) * inside
    ridge = cv2.dilate(d, disk(POCKET_MM))                    # local half-width of the space
    channel = inside & (d >= ridge - width_mm / 2 * PX_PER_MM)
    slot = np.zeros_like(comp)
    if holes.any():
        n, cc = cv2.connectedComponents(holes.astype(np.uint8), connectivity=4)
        dist = cv2.distanceTransform(holes.astype(np.uint8), cv2.DIST_L2, 5)
        for i in range(1, n):
            h = cc == i
            ys, xs = np.nonzero(h)
            if len(ys) < COUNTER_MIN_MM2 * PX_PER_MM ** 2 or max(np.ptp(ys), np.ptp(xs)) < PX_PER_MM:
                continue                                     # a speck where strokes cross, not a counter
            if 2 * dist[h].max() / PX_PER_MM < width_mm:     # counter already too narrow
                slot |= cv2.dilate((skeletonize(h) | (h & (dist == dist[h].max()))).astype(np.uint8),
                                   disk(width_mm)).astype(bool)
    return channel, slot & comp


def _grow(fg: np.ndarray, cc: np.ndarray, n: int, r_mm: float, fx: Fixes, rep: Report) -> np.ndarray:
    """Grow every shape by ``r_mm`` into its own territory, protecting gaps, counters
    and pockets. Returns the new component map (0 = background, i = shape i)."""
    h, w = fg.shape
    r_px = r_mm * PX_PER_MM
    gap_px = fx.min_gap_mm * PX_PER_MM
    pad = int(r_px + gap_px) + 3
    best1 = np.full((h, w), np.inf, np.float32)
    best1_id = np.zeros((h, w), np.int32)
    best2 = np.full((h, w), np.inf, np.float32)
    boxes = []
    for i in range(1, n):
        ys, xs = np.nonzero(cc == i)
        y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
        boxes.append((i, y0, y1, x0, x1))
        d = cv2.distanceTransform((cc[y0:y1, x0:x1] != i).astype(np.uint8), cv2.DIST_L2, 5)
        b1, b1i, b2 = best1[y0:y1, x0:x1], best1_id[y0:y1, x0:x1], best2[y0:y1, x0:x1]
        closer = d < b1
        b2[:] = np.where(closer, b1, np.minimum(b2, d))
        b1[:] = np.where(closer, d, b1)
        b1i[:] = np.where(closer, i, b1i)
    # a background pixel joins its nearest shape when within reach and far enough from
    # the midline to the next shape that the two grown shapes keep the minimum gap
    with np.errstate(invalid="ignore"):         # inf - inf far from every shape
        claim = ~fg & (best1 <= r_px) & (best2 - best1 >= gap_px + 1)
    out = np.where(fg, cc, 0).astype(np.int32)
    out[claim] = best1_id[claim]
    for i, y0, y1, x0, x1 in boxes:
        orig = cc[y0:y1, x0:x1] == i
        channel, slot = _channels(orig, fx.min_inside_mm)
        sub = out[y0:y1, x0:x1]
        keep = sub.copy()
        # try the full protection, then without carving slots, then without channels:
        # a cut must never split the shape
        for use_slot, use_channel in ((True, True), (False, True), (False, False)):
            sub[:] = keep
            grown = (sub == i) & ~orig
            if use_channel:
                sub[grown & channel] = 0
            if use_slot and slot.any():
                sub[slot] = 0
            reopened = _reopen(sub, i, orig, r_px, fx.min_inside_mm)
            _rebuild_caps(sub, i, orig, r_px)
            # growth a cut left stranded off the shape is dropped; only a split of the
            # shape itself (a slot through a thin stroke) counts
            k, pieces = cv2.connectedComponents((sub == i).astype(np.uint8), connectivity=8)
            on_shape = set(np.unique(pieces[orig & (sub == i)]).tolist()) - {0}
            sub[(sub == i) & ~np.isin(pieces, list(on_shape))] = 0
            if len(on_shape) <= 1:
                break
        if reopened:
            rep.notes.append("Reopened a space that growing closed off.")
        if use_slot and slot.any():
            rep.notes.append(f"Opened a counter narrower than {fx.min_inside_mm:g} mm with a "
                             f"{fx.min_inside_mm:g} mm slot.")
        if not use_channel:
            rep.warnings.append("Couldn't keep every space in one shape open without splitting it; "
                                "check its counters on the preview.")
    return out


def _reopen(sub: np.ndarray, i: int, orig: np.ndarray, r_px: float, width_mm: float) -> bool:
    """Reopen spaces that growing sealed off (feet growing together under an "n", two
    loops of a swash touching): a straight vertical channel from the bottom of each new
    hole down past the shape, taking back growth only. If the space still doesn't open
    (it opens sideways), the growth around it is undone. Speck-sized new holes (corners
    the growth wrapped around) are filled instead.

    Returns:
        True when a real space was reopened.
    """
    now = sub == i
    new_holes = _holes(now) & ~_holes(orig)
    if not new_holes.any():
        return False
    n, cc = cv2.connectedComponents(new_holes.astype(np.uint8), connectivity=4)
    big = np.zeros_like(new_holes)
    for k in range(1, n):
        h = cc == k
        if h.sum() < COUNTER_MIN_MM2 * PX_PER_MM ** 2:
            sub[h] = i                     # a speck left in a corner: fill it
        else:
            big |= h
    if not big.any():
        return False
    # a straight channel from the bottom of each new hole down past the shape, at least
    # width_mm wide and as wide as the hole near its bottom; it only takes back growth,
    # never the original shape
    n, cc = cv2.connectedComponents(big.astype(np.uint8), connectivity=4)
    bottom = sub.shape[0]
    for k in range(1, n):
        ys, xs = np.nonzero(cc == k)
        low = ys >= ys.max() - round(0.3 * PX_PER_MM)
        x0, x1 = xs[low].min(), xs[low].max()
        half = max(round(width_mm * PX_PER_MM / 2), (x1 - x0 + 1) // 2)
        cx = (x0 + x1) // 2
        band = np.zeros_like(big)
        band[ys.min():bottom, max(0, cx - half):cx + half + 1] = True
        sub[band & (sub == i) & ~orig] = 0
    if _count_holes(sub == i) > _count_holes(orig):       # the space opens sideways: undo growth around it
        near = cv2.dilate(big.astype(np.uint8), disk(2 * (r_px / PX_PER_MM) + width_mm)).astype(bool)
        sub[near & (sub == i) & ~orig] = 0
    return True


def _rebuild_caps(sub: np.ndarray, i: int, orig: np.ndarray, r_px: float) -> None:
    """Turn end caps that ballooned while growing back into rounded rectangles.

    For each free end of the shape's skeleton, the cap's width (twice the largest
    distance to the edge within one stroke width of the end) is compared with the
    stroke's width just before it. When the cap is over ``CAP_GROWTH`` times wider, the
    grown pixels around the end are replaced by a rectangle as wide as the stroke,
    reaching the cap's end, with ``CAP_RADIUS_MM`` corners.
    """
    from skimage.morphology import skeletonize

    if r_px <= 0:
        return
    now = sub == i
    dist = cv2.distanceTransform(now.astype(np.uint8), cv2.DIST_L2, 5)
    skel = skeletonize(now)
    nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32)) - skel
    ends = np.argwhere(skel & (nb == 1))
    for y, x in ends:
        # walk back along the skeleton about three stroke widths
        path = [(y, x)]
        seen = {(y, x)}
        for _ in range(int(6 * max(dist[y, x], 3))):
            cy, cx = path[-1]
            nxt = [(cy + dy, cx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if (dy or dx) and 0 <= cy + dy < skel.shape[0] and 0 <= cx + dx < skel.shape[1]
                   and skel[cy + dy, cx + dx] and (cy + dy, cx + dx) not in seen]
            if len(nxt) != 1:
                break
            path.append(nxt[0])
            seen.add(nxt[0])
        if len(path) < 8:
            continue
        ws = np.array([dist[p] for p in path])
        stroke = float(np.median(ws[len(ws) // 2:]))
        cap_len = max(3, int(stroke * 2))
        cap = float(ws[:cap_len].max())
        if stroke < 2 or cap < CAP_GROWTH * stroke:
            continue
        # rectangle along the end's direction, as wide as the stroke
        p0 = np.array(path[min(len(path) - 1, cap_len)], float)
        tip = np.array(path[0], float)
        d = tip - p0
        d /= max(np.hypot(*d), 1e-9)
        tip = tip + d * stroke                                # the skeleton stops short of the end
        nrm = np.array([-d[1], d[0]])
        corners = np.array([p0 + nrm * stroke, tip + nrm * stroke, tip - nrm * stroke, p0 - nrm * stroke])
        rect = np.zeros_like(now, np.uint8)
        cv2.fillPoly(rect, [corners[:, ::-1].round().astype(np.int32)], 1)
        rad = CAP_RADIUS_MM * PX_PER_MM
        if rad >= 1:
            k = disk(2 * CAP_RADIUS_MM)
            rect = cv2.dilate(cv2.erode(rect, k), k)
        zone = np.zeros_like(now, np.uint8)
        cv2.circle(zone, (int(tip[1]), int(tip[0])), int(cap_len + stroke), 1, -1)
        zone = zone.astype(bool) & ~orig                      # only ever remove grown pixels
        sub[zone & now & ~rect.astype(bool)] = 0


def _spread_narrow(labels: np.ndarray, min_mm: float, rep: Report) -> np.ndarray:
    """Widen letters whose inside space is under ``min_mm`` (a tight "n" or "m").

    Growing can't open such a space (it's protected, but already too narrow), so the
    letter is cut down the middle of the space and its right half moved right until the
    space is ``min_mm`` wide; the cut is filled by repeating the column of pixels at it.
    Letters after it on the same line (starting right of the cut, overlapping it
    vertically by half their height) move with it, keeping the word's spacing. A spread
    that would push something into a shape that stays put (a swash under the word) is
    skipped with a warning.

    Returns:
        The label map, wider by the total spread.
    """
    n, cc = _components(labels)
    todo = []
    for i in range(1, n):
        ys, xs = np.nonzero(cc == i)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        comp = cc[y0:y1, x0:x1] == i
        pad = round(POCKET_MM * PX_PER_MM)
        c = np.pad(comp, pad)
        closed = cv2.morphologyEx(c.astype(np.uint8), cv2.MORPH_CLOSE, disk(POCKET_MM)).astype(bool)
        pockets = closed & ~c & ~_holes(c)
        depth = cv2.distanceTransform(closed.astype(np.uint8), cv2.DIST_L2, 5)
        k, pc = cv2.connectedComponents(pockets.astype(np.uint8), connectivity=8)
        for j in range(1, k):
            pj = pc == j
            if depth[pj].max() < POCKET_DEPTH_MM * PX_PER_MM:
                continue
            py, px = np.nonzero(pj)
            if np.ptp(py) < 1.5 * PX_PER_MM or np.ptp(py) < 2 * np.ptp(px):
                continue                                       # not a tall inside space
            width = np.median(pj.sum(axis=1)[pj.any(axis=1)])
            need = int(np.ceil(min_mm * PX_PER_MM - width))
            if need > 0:
                todo.append((x0 - pad + int(np.median(px)), need, i, (y0, y1)))
    if not todo:
        return labels
    out = labels
    for cut, k, i, (y0, y1) in sorted(todo, reverse=True):   # right to left: cuts stay valid
        n, cc = _components(out)
        h, w = out.shape
        fg = out != TRANSPARENT
        # the letter: the shape crossing the cut inside its own rows
        band = cc[y0:y1, cut]
        ids = [v for v in np.unique(band) if v]
        if not ids:
            continue
        letter = ids[0]
        movers = {letter}
        for v in range(1, n):
            ys, xs = np.nonzero(cc == v)
            ov = min(ys.max(), y1) - max(ys.min(), y0)
            if xs.min() >= cut and ov >= 0.5 * min(np.ptp(ys) + 1, y1 - y0):
                movers.add(v)
        moving = np.isin(cc, list(movers)) & (np.arange(w)[None, :] >= cut)
        new = np.full((h, w + k), TRANSPARENT, np.uint8)
        new[:, :w][fg & ~moving] = out[fg & ~moving]
        ys, xs = np.nonzero(moving)
        if (new[ys, xs + k] != TRANSPARENT).any():
            rep.warnings.append("A letter's inside is under "
                                f"{min_mm:g} mm, but widening it would run into a shape below or beside it.")
            continue
        new[ys, xs + k] = out[ys, xs]
        col = (cc[:, cut] == letter)
        for d in range(k):                                      # fill the cut by repeating its column
            new[col, cut + d] = out[col, cut]
        out = new
        rep.notes.append(f"Widened a letter by {k / PX_PER_MM:.2f} mm so its inside is {min_mm:g} mm open; "
                         "the letters after it moved with it.")
    return out


def _find_dots(n: int, cc: np.ndarray) -> list[tuple[int, int]]:
    """Dots in the original design: separate shapes at most ``DOT_MAX_MM`` across, with
    another shape right below (the stem: starting within 4 mm, overlapping sideways).

    Returns:
        One (y, x) pixel inside each dot, to find it again after growing.
    """
    stats = []
    for i in range(1, n):
        ys, xs = np.nonzero(cc == i)
        stats.append((i, ys.min(), ys.max(), xs.min(), xs.max(), (ys[0], xs[0])))
    lim = DOT_MAX_MM * PX_PER_MM
    out = []
    for i, y0, y1, x0, x1, px in stats:
        if max(y1 - y0, x1 - x0) > lim:
            continue
        if any(s[0] != i and s[1] > y1 and s[1] - y1 < 4 * PX_PER_MM and min(x1, s[4]) - max(x0, s[3]) > 0
               for s in stats):
            out.append(px)
    return out


def _repair_dots(labels: np.ndarray, seeds: list[tuple[int, int]], rep: Report) -> None:
    """Redraw dots above stems as circles: dashes always, round dots only when they
    crowd something.

    ``seeds`` are pixels inside each dot (from :func:`_find_dots`). The circle is
    1.05 x the stem's width (0.9 to 3 mm), centered over the stem, and moved up until it
    has ``DOT_CLEARANCE_MM`` around it. Edits ``labels`` in place, keeping the dot's color.
    """
    n, cc = _components(labels)
    fg = cc > 0
    stats = {}
    for i in range(1, n):
        ys, xs = np.nonzero(cc == i)
        stats[i] = (ys.min(), ys.max(), xs.min(), xs.max())
    for sy, sx in seeds:
        i = int(cc[sy, sx])
        if not i:
            continue
        y0, y1, x0, x1 = stats[i]
        below = [(k, v) for k, v in stats.items() if k != i and v[0] > y1 and v[0] - y1 < 4 * PX_PER_MM
                 and min(x1, v[3]) - max(x0, v[2]) > 0]
        if not below:
            continue
        stem = cc == min(below, key=lambda kv: kv[1][0])[0]
        sy0 = np.nonzero(stem.any(axis=1))[0].min()
        top = stem[sy0:sy0 + round(1.5 * PX_PER_MM)]
        cols = np.nonzero(top.any(axis=0))[0]
        rows_w = top.sum(axis=1)
        stem_w = float(np.median(rows_w[rows_w > 0])) / PX_PER_MM
        dot = cc == i
        color = int(np.bincount(labels[dot]).argmax())
        others = fg & ~dot
        clear_d = cv2.distanceTransform((~others).astype(np.uint8), cv2.DIST_L2, 5)
        clearance = clear_d[dot].min() / PX_PER_MM
        dash = (x1 - x0 + 1) > DOT_DASH_ASPECT * (y1 - y0 + 1)
        if not dash and clearance >= DOT_CLEARANCE_MM:
            continue
        diam = float(np.clip(1.05 * stem_w, 0.9, 3.0))
        rad = diam * PX_PER_MM / 2
        cx = (cols.min() + cols.max()) / 2
        cy = (y0 + y1) / 2
        need = DOT_CLEARANCE_MM * PX_PER_MM + rad
        while cy - rad > 0 and clear_d[int(cy), int(cx)] < need:
            cy -= 1
        labels[dot] = TRANSPARENT
        cv2.circle(labels, (int(round(cx)), int(round(cy))), int(round(rad)), color, -1)
        rep.notes.append(f"Redrew {'a dash' if dash else 'a crowded dot'} above a stem as a round "
                         f"{diam:.1f} mm dot with {DOT_CLEARANCE_MM:g} mm clearance.")


def _fill_corner_tips(labels: np.ndarray, rep: Report, width_mm: float = TIP_FILL_MM) -> None:
    """Fill the points of sharp inside corners narrower than ``width_mm`` (a closing),
    per color, without merging shapes, changing the number of shapes or touching
    counters. Edits ``labels`` in place."""
    filled = 0
    for c in np.unique(labels):
        if c == TRANSPARENT:
            continue
        m = labels == c
        add = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, disk(width_mm)).astype(bool) & ~m
        add &= labels == TRANSPARENT
        if not add.any():
            continue
        n, cc = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        holes = _holes(m)
        na, acc = cv2.connectedComponents(add.astype(np.uint8), connectivity=8)
        for k in range(1, na):
            blob = acc == k
            if (blob & holes).any():
                continue                                       # never shrink a counter
            touch = cc[cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)]
            if len(set(touch[touch > 0].tolist())) != 1:
                continue                                       # would join two shapes
            labels[blob] = c
            filled += 1
    if filled:
        rep.notes.append(f"Filled {filled} corner tip(s).")


def _open_gaps(labels: np.ndarray, width_mm: float, rep: Report) -> None:
    """Widen every gap narrower than ``width_mm`` (at least 1 mm long and not a wedge
    open at one end only, so corner tips don't count) to ``width_mm``: a channel that wide along the gap's center line
    is cleared, trimming the shapes beside it. A channel that would split a shape is
    skipped. Edits ``labels`` in place.
    """
    from skimage.morphology import skeletonize

    bg = labels == TRANSPARENT
    wide = cv2.morphologyEx(bg.astype(np.uint8), cv2.MORPH_OPEN, disk(width_mm)).astype(bool)
    narrow = bg & ~wide
    n, cc, st, _ = cv2.connectedComponentsWithStats(narrow.astype(np.uint8), connectivity=8)
    opened = skipped = 0
    pad = round(2 * width_mm * PX_PER_MM)
    h, w = labels.shape
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < 0.1 * PX_PER_MM ** 2:
            continue
        box = cc[y:y + bh, x:x + bw] == i
        ys, xs = np.nonzero(box)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
        if max(rw, rh) + 1 < PX_PER_MM:
            continue                                   # a corner tip
        y0, y1, x0, x1 = max(0, y - pad), min(h, y + bh + pad), max(0, x - pad), min(w, x + bw + pad)
        ring = cv2.dilate((cc[y0:y1, x0:x1] == i).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if cv2.connectedComponents((ring & wide[y0:y1, x0:x1]).astype(np.uint8), connectivity=8)[0] - 1 == 1:
            continue                                   # a wedge where two strokes meet: a corner tip
        gap = cc[y0:y1, x0:x1] == i
        # a hair wider than asked, so the pixel grid doesn't leave it just under the minimum
        channel = cv2.dilate(skeletonize(gap).astype(np.uint8), disk(width_mm + 0.15)).astype(bool)
        sub = labels[y0:y1, x0:x1]
        cut = channel & (sub != TRANSPARENT)
        if not cut.any():
            continue
        before = _components(sub)[0]
        trial = sub.copy()
        trial[cut] = TRANSPARENT
        if _components(trial)[0] > before:
            skipped += 1                               # the channel would cut a stroke in two
            continue
        sub[cut] = TRANSPARENT
        opened += 1
    if opened:
        rep.notes.append(f"Opened {opened} narrow gap(s) to {width_mm:g} mm by trimming the shapes beside them.")
    if skipped:
        rep.warnings.append(f"{skipped} narrow gap(s) couldn't be opened without cutting a stroke in two.")


def _components(labels: np.ndarray) -> tuple[int, np.ndarray]:
    """Separate shapes of the design (all colors together), 8-connected."""
    return cv2.connectedComponents((labels != TRANSPARENT).astype(np.uint8), connectivity=8)


def apply(labels: np.ndarray, fx: Fixes, crop: bool = True) -> tuple[np.ndarray, Report]:
    """Apply the requested fixes to a label map.

    Args:
        labels: HxW uint8 label map at ``PX_PER_MM`` (not modified).
        fx: The fixes to apply.
        crop: Crop a grown map to the design (off keeps the padded canvas, for comparing).

    Returns:
        ``(labels, report)``. When growing, the map is padded so shapes near the edge
        have room, then cropped to the design plus 2 px; it can come out larger than
        the input. ``report.grow_mm`` is the growth actually applied.
    """
    rep = Report()
    fg = labels != TRANSPARENT
    rep.stroke = stroke_widths(fg)
    out = labels.copy()
    n0, _ = _components(out)

    r_mm = fx.bolder_mm
    if not r_mm and fx.bolder_pct and rep.stroke:
        r_mm = fx.bolder_pct / 100 * rep.stroke["median"] / 2
    r_mm = max(0.0, min(r_mm, 1.5))
    if r_mm > 0:
        out = _spread_narrow(out, fx.min_inside_mm, rep)
    pad = int((r_mm + fx.min_gap_mm) * PX_PER_MM) + 4 if r_mm > 0 else 0
    dots = _find_dots(*_components(out)) if fx.repair_dots else []
    if r_mm > 0:
        out = np.pad(out, pad, constant_values=TRANSPARENT)
        n, cc = _components(out)
        grown = _grow(out != TRANSPARENT, cc, n, r_mm, fx, rep)
        # new pixels take the color of the nearest original pixel of their shape
        new = (grown > 0) & (out == TRANSPARENT)
        if new.any():
            _, idx = cv2.distanceTransformWithLabels((out == TRANSPARENT).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
            src = np.zeros(idx.max() + 1, np.uint8)
            ys, xs = np.nonzero(out != TRANSPARENT)
            src[idx[ys, xs]] = out[ys, xs]
            out[new] = src[idx[new]]
        out[(grown == 0) & (out != TRANSPARENT)] = TRANSPARENT     # channels and slots carved out
        rep.grow_mm = round(r_mm, 2)
        rep.notes.insert(0, f"Made strokes bolder by {r_mm:.2f} mm per side "
                            f"(about {2 * r_mm / rep.stroke['median'] * 100:.0f}% of the median stroke)."
                         if rep.stroke else f"Made strokes bolder by {r_mm:.2f} mm per side.")
    if dots:
        _repair_dots(out, [(y + pad, x + pad) for y, x in dots], rep)
    if fx.fill_corner_tips:
        # as wide as the narrowest gap the fabric keeps, so the gap check finds none left
        _fill_corner_tips(out, rep, max(TIP_FILL_MM, fx.min_gap_mm))
    if fx.open_gaps:
        _open_gaps(out, fx.min_gap_mm, rep)

    # specks the edits left behind (a corner cut off by a channel) aren't stitchable
    n, cc, st, _ = cv2.connectedComponentsWithStats((out != TRANSPARENT).astype(np.uint8), connectivity=8)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < SPECK_MM2 * PX_PER_MM ** 2:
            out[cc == k] = TRANSPARENT
    n1, _ = _components(out)
    if n1 != n0:
        rep.warnings.append(f"The fixes changed the number of separate shapes from {n0 - 1} to {n1 - 1}: "
                            "check the preview for letters that merged or split.")
    if r_mm > 0 and crop:
        ys, xs = np.nonzero(out != TRANSPARENT)
        if len(ys):
            m = 2
            out = out[max(0, ys.min() - m):ys.max() + 1 + m, max(0, xs.min() - m):xs.max() + 1 + m]
    return out, rep
