"""Hand edits to a session's label map: split letters that touch or overlap, rebuild
the edge a cut leaves, and move a letter together with the letters after it.

Where it sits: ``app.py``'s ``/api/edit`` calls :func:`apply_op` on the session's label
map (the analysis result, before merges), keeps an undo history on the session and
drops the cached geometry, so the next plan traces the edited shapes.

Why: in some logos letters overlap or touch (an "m" running into an "S", a "t" touching
an "h"). In thread they merge into one blob. Cutting them apart and nudging them gives
each letter its own outline.

Operations (all coordinates in mm on the label map, which is at ``PX_PER_MM``):

- ``cut``: clear a ``CUT_MM`` wide line between two points (only where there's ink).
- ``rebuild``: on one side of a cut line, mirror the piece's pixels near the cut across
  the line, so a stroke the cut sliced lengthwise gets back its full width (its hidden
  edge rebuilt from its visible half). Mirrored pixels never come within ``CUT_MM`` of
  another shape.
- ``move``: move the shape under a point by (dx, dy), and with it the shapes after it
  in the same word: those starting to its right on the same line, up to the first gap
  wider than ``WORD_GAP_MM`` and 1.8 times the line's typical letter gap. The point must be on ink; a click on background is an
  error rather than a silent no-op.
"""
from __future__ import annotations

import cv2
import numpy as np

from .palette import TRANSPARENT
from .raster import PX_PER_MM

CUT_MM = 0.4          # width of a cut line
WORD_GAP_MM = 2.0     # a horizontal gap wider than this between letters ends a word
REBUILD_MM = 2.5      # how far from the cut the piece is mirrored


class EditError(ValueError):
    """A request that can't be applied (point off the design, nothing to cut)."""


def _px(v_mm: float) -> int:
    return int(round(v_mm * PX_PER_MM))


def _components(labels: np.ndarray) -> tuple[int, np.ndarray]:
    return cv2.connectedComponents((labels != TRANSPARENT).astype(np.uint8), connectivity=8)


def cut(labels: np.ndarray, p0: tuple[float, float], p1: tuple[float, float]) -> np.ndarray:
    """Clear a ``CUT_MM`` line from p0 to p1 (mm, x/y). Raises EditError if it crosses no ink."""
    line = np.zeros(labels.shape, np.uint8)
    cv2.line(line, (_px(p0[0]), _px(p0[1])), (_px(p1[0]), _px(p1[1])), 1, max(1, _px(CUT_MM)))
    hit = line.astype(bool) & (labels != TRANSPARENT)
    if not hit.any():
        raise EditError("The cut line doesn't cross the design. Draw it across where the letters touch.")
    out = labels.copy()
    out[hit] = TRANSPARENT
    return out


def _side(shape: tuple[int, int], p0, p1) -> np.ndarray:
    """Signed side of every pixel relative to the line p0->p1 (positive = left of it)."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    x0, y0, x1, y1 = (_px(v) for v in (*p0, *p1))
    return (x1 - x0) * (yy - y0) - (y1 - y0) * (xx - x0)


def side_of(p0, p1, at) -> int:
    """Which side of the line p0->p1 (mm) the point ``at`` is on: +1 or -1, as
    :func:`rebuild` counts them."""
    x0, y0, x1, y1 = (float(_px(v)) for v in (*p0, *p1))
    d = np.array([x1 - x0, y1 - y0])
    nrm = np.array([-d[1], d[0]])
    return 1 if (_px(at[0]) - x0) * nrm[0] + (_px(at[1]) - y0) * nrm[1] >= 0 else -1


def rebuild(labels: np.ndarray, p0, p1, side: int) -> np.ndarray:
    """Rebuild the cut edge of the piece on one side of the line (side = +1 or -1, see
    :func:`side_of`).

    The pixels on that side within ``REBUILD_MM`` of the line are mirrored across it and
    added, except where they'd come within ``CUT_MM`` of anything on the other side.
    """
    h, w = labels.shape
    x0, y0, x1, y1 = (float(_px(v)) for v in (*p0, *p1))
    d = np.array([x1 - x0, y1 - y0])
    ln = np.hypot(*d)
    if ln < 1:
        raise EditError("The cut line is too short.")
    d /= ln
    nrm = np.array([-d[1], d[0]])
    ink = labels != TRANSPARENT
    ys, xs = np.nonzero(ink)
    rel = np.stack([xs - x0, ys - y0], axis=1)
    dist = rel @ nrm                                           # signed distance to the line, px
    along = rel @ d
    s = np.sign(dist)
    sel = (s == side) & (np.abs(dist) <= REBUILD_MM * PX_PER_MM) & (along >= -REBUILD_MM * PX_PER_MM) \
        & (along <= ln + REBUILD_MM * PX_PER_MM)
    if not sel.any():
        raise EditError("Nothing on that side of the cut to rebuild.")
    mx = np.round(xs[sel] - 2 * dist[sel] * nrm[0]).astype(int)
    my = np.round(ys[sel] - 2 * dist[sel] * nrm[1]).astype(int)
    ok = (mx >= 0) & (mx < w) & (my >= 0) & (my < h)
    mx, my, vals = mx[ok], my[ok], labels[ys[sel][ok], xs[sel][ok]]
    yy, xx = np.mgrid[:h, :w]
    other = ink & (np.sign((xx - x0) * nrm[0] + (yy - y0) * nrm[1]) != side)
    keep_off = cv2.dilate(other.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * _px(CUT_MM) + 1,) * 2)).astype(bool)
    free = ~keep_off[my, mx]
    out = labels.copy()
    out[my[free], mx[free]] = vals[free]
    # mirrored pixels land on a rounded grid: close the pinholes they leave
    added = np.zeros((h, w), np.uint8)
    added[my[free], mx[free]] = 1
    holes = cv2.morphologyEx(added, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool) & (out == TRANSPARENT)
    if holes.any():
        out[holes] = int(np.bincount(vals[free]).argmax()) if free.any() else out[holes]
    return out


def followers(cc: np.ndarray, n: int, sel: int) -> list[int]:
    """The shape ``sel`` plus the shapes after it in the same word (see module docs)."""
    boxes = {}
    for i in range(1, n):
        ys, xs = np.nonzero(cc == i)
        boxes[i] = (xs.min(), xs.max(), ys.min(), ys.max())
    x0, x1, y0, y1 = boxes[sel]
    line = [i for i, (a, b, c, d) in boxes.items()
            if i != sel and a > x0 and min(d, y1) - max(c, y0) >= 0.5 * min(d - c, y1 - y0)]
    # a word ends at a gap clearly wider than the line's usual letter spacing
    row = sorted([i for i, (a, b, c, d) in boxes.items() if min(d, y1) - max(c, y0) >= 0.5 * min(d - c, y1 - y0)],
                 key=lambda i: boxes[i][0])
    gaps = [boxes[b][0] - boxes[a][1] for a, b in zip(row, row[1:]) if boxes[b][0] > boxes[a][1]]
    limit = max(WORD_GAP_MM * PX_PER_MM, 1.8 * float(np.median(gaps)) if gaps else 0)
    out, right = [sel], x1
    for i in sorted(line, key=lambda i: boxes[i][0]):
        if boxes[i][0] - right > limit:
            break
        out.append(i)
        right = max(right, boxes[i][1])
    return out


def apply_op(labels: np.ndarray, op: dict, last_cut: tuple | None) -> tuple[np.ndarray, str, tuple | None]:
    """Run one edit request on a label map.

    Args:
        labels: The current label map (not modified).
        op: ``{"op": "cut", "x0", "y0", "x1", "y1"}``, ``{"op": "rebuild", "x", "y"}``
            (the piece clicked, beside the last cut) or ``{"op": "move", "x", "y", "dx",
            "dy"}``, all in mm.
        last_cut: The previous cut's (p0, p1), needed by ``rebuild``.

    Returns:
        (new label map, a short note for the UI, the cut line to remember).

    Raises:
        EditError: For a request that can't be applied.
    """
    kind = op.get("op")
    if kind == "cut":
        p0, p1 = (op["x0"], op["y0"]), (op["x1"], op["y1"])
        out = cut(labels, p0, p1)
        n0, n1 = _components(labels)[0] - 1, _components(out)[0] - 1
        note = (f"Cut: {n0} shape(s) became {n1}." if n1 > n0
                else "Cut, but the shapes still touch somewhere else: draw another cut there.")
        return out, note, (p0, p1)
    if kind == "rebuild":
        if not last_cut:
            raise EditError("Cut first, then click the piece whose edge to rebuild.")
        at = (op["x"], op["y"])
        h, w = labels.shape
        if not (0 <= _px(at[0]) < w and 0 <= _px(at[1]) < h) or labels[_px(at[1]), _px(at[0])] == TRANSPARENT:
            raise EditError("That spot isn't on a letter. Click the piece whose cut edge to rebuild.")
        out = rebuild(labels, *last_cut, side_of(*last_cut, at))
        return out, "Rebuilt the cut edge from the piece's other side.", last_cut
    if kind == "move":
        out, k = move(labels, (op["x"], op["y"]), float(op.get("dx", 0)), float(op.get("dy", 0)))
        return out, f"Moved {k} shape(s) (the letter and the rest of its word).", None
    raise EditError(f"Unknown edit: {kind}")


def move(labels: np.ndarray, at: tuple[float, float], dx: float, dy: float) -> tuple[np.ndarray, int]:
    """Move the shape under ``at`` (mm) and the rest of its word by (dx, dy) mm.

    The map grows on a side the shapes would leave (so coordinates stay put otherwise).
    Raises EditError when ``at`` isn't on ink,
    or the move would land on a shape that stays put.

    Returns:
        (new label map, number of shapes moved).
    """
    h, w = labels.shape
    x, y = _px(at[0]), _px(at[1])
    if not (0 <= x < w and 0 <= y < h) or labels[y, x] == TRANSPARENT:
        raise EditError("That spot isn't on a letter. Click right on the letter you want to move.")
    n, cc = _components(labels)
    ids = followers(cc, n, int(cc[y, x]))
    mx, my = _px(dx), _px(dy)
    moving = np.isin(cc, ids)
    ys, xs = np.nonzero(moving)
    # grow the canvas only on a side the moved shapes would leave
    pl, pr = max(0, -(xs.min() + mx)), max(0, xs.max() + mx - (w - 1))
    pt, pb = max(0, -(ys.min() + my)), max(0, ys.max() + my - (h - 1))
    big = np.pad(labels, ((pt, pb), (pl, pr)), constant_values=TRANSPARENT)
    mv = np.pad(moving, ((pt, pb), (pl, pr)))
    ys, xs = np.nonzero(mv)
    vals = big[ys, xs].copy()
    big[ys, xs] = TRANSPARENT
    ty, tx = ys + my, xs + mx
    if (big[ty, tx] != TRANSPARENT).any():
        raise EditError("Moving it that far would run into another shape.")
    big[ty, tx] = vals
    return big, len(ids)
