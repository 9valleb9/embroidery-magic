"""Tests for embroidery/lettering.py: bolder strokes and the protections that keep
letters readable, dot repair, corner tips and stroke widths.

Shapes are drawn straight into a label map (id 0 = the thread color, TRANSPARENT =
empty) at raster.PX_PER_MM, so each test controls the exact geometry.
"""
import cv2
import numpy as np

from embroidery import lettering as L
from embroidery.palette import TRANSPARENT
from embroidery.raster import PX_PER_MM as P


def canvas(w_mm=40, h_mm=30):
    """An empty label map w x h mm."""
    return np.full((h_mm * P, w_mm * P), TRANSPARENT, np.uint8)


def mm(*v):
    """mm -> px, as ints."""
    return tuple(int(round(x * P)) for x in v)


def shapes(lab):
    """Number of separate shapes (8-connected)."""
    return cv2.connectedComponents((lab != TRANSPARENT).astype(np.uint8), connectivity=8)[0] - 1


def test_stroke_widths():
    lab = canvas()
    cv2.rectangle(lab, mm(5, 10), mm(35, 12), 0, -1)              # a 2 mm bar
    st = L.stroke_widths(lab != TRANSPARENT)
    assert abs(st["median"] - 2.0) <= 0.15


def test_bolder_keeps_letters_apart():
    """Two bars 1 mm apart grow 0.3 mm per side: they stay separate, at least 0.8 mm
    apart, while their outer sides grow the full amount."""
    lab = canvas()
    cv2.rectangle(lab, mm(5, 5), mm(15, 25), 0, -1)
    cv2.rectangle(lab, mm(16, 5), mm(26, 25), 0, -1)
    out, rep = L.apply(lab, L.Fixes(bolder_mm=0.3, repair_dots=False), crop=False)
    assert shapes(out) == 2 and not rep.warnings
    pad = (out.shape[1] - lab.shape[1]) // 2
    row = out[pad + 15 * P] != TRANSPARENT
    xs = np.flatnonzero(row)
    gap = np.diff(xs).max() - 1
    assert gap >= 0.78 * P                                          # the 1 mm gap kept at least 0.8 mm
    assert (xs[-1] - xs[0]) / P >= 21 + 0.55                        # outer sides grew ~0.3 mm each


def test_bolder_keeps_counters_open():
    """A ring with a 1.6 mm counter grows 0.4 mm per side: the counter shrinks to no
    less than 1 mm across (it would close to 0.8 mm unprotected)."""
    lab = canvas()
    cv2.circle(lab, mm(20, 15), int(3 * P), 0, -1)
    cv2.circle(lab, mm(20, 15), int(0.8 * P), TRANSPARENT, -1)
    out, _ = L.apply(lab, L.Fixes(bolder_mm=0.4, repair_dots=False), crop=False)
    hole = L._holes(out != TRANSPARENT)
    assert hole.any()
    d = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
    assert 2 * d.max() / P >= 0.95


def test_bolder_does_not_seal_the_inside_of_an_n():
    """Feet 0.8 mm apart under an arch: growing would close the bottom into a hole."""
    lab = canvas()
    cv2.rectangle(lab, mm(10, 8), mm(12, 22), 0, -1)              # left stem
    cv2.rectangle(lab, mm(15, 8), mm(17, 22), 0, -1)              # right stem
    cv2.rectangle(lab, mm(10, 8), mm(17, 10), 0, -1)              # arch
    cv2.rectangle(lab, mm(9, 21), mm(13.1, 22), 0, -1)            # feet, 0.8 mm apart
    cv2.rectangle(lab, mm(13.9, 21), mm(18, 22), 0, -1)
    out, _ = L.apply(lab, L.Fixes(bolder_mm=0.4, repair_dots=False), crop=False)
    assert shapes(out) == 1
    assert not L._holes(out != TRANSPARENT).any()


def test_dash_above_a_stem_becomes_a_round_dot():
    lab = canvas()
    cv2.rectangle(lab, mm(10, 12), mm(11.6, 25), 0, -1)           # stem, 1.6 mm wide
    cv2.rectangle(lab, mm(9.6, 9.5), mm(12, 10.2), 0, -1)         # a dash for a dot
    out, rep = L.apply(lab, L.Fixes(), crop=False)
    assert shapes(out) == 2 and any("dash" in n for n in rep.notes)
    n, cc, st, _ = cv2.connectedComponentsWithStats((out != TRANSPARENT).astype(np.uint8), connectivity=8)
    dot = min(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA])
    w, h = st[dot, cv2.CC_STAT_WIDTH], st[dot, cv2.CC_STAT_HEIGHT]
    assert abs(w - h) <= 2                                          # round
    gap = st[[i for i in range(1, n) if i != dot][0], cv2.CC_STAT_TOP] - (st[dot, cv2.CC_STAT_TOP] + h)
    assert gap >= 0.95 * P                                          # 1 mm above the stem


def test_fill_corner_tips_keeps_shapes_and_counters():
    """A V-notch's point gets filled; a separate shape nearby isn't joined and a small
    counter isn't touched."""
    lab = canvas()
    pts = np.array([mm(5, 5), mm(25, 5), mm(25, 20), mm(15, 8), mm(5, 20)], np.int32)  # a deep V notch
    cv2.fillPoly(lab, [pts], 0)
    cv2.rectangle(lab, mm(26.3, 5), mm(30, 20), 0, -1)            # separate bar 1.3 mm away
    cv2.circle(lab, mm(10, 7), int(0.5 * P), TRANSPARENT, -1)     # a small counter
    hole_before = L._holes(lab != TRANSPARENT).sum()
    out, rep = L.apply(lab, L.Fixes(repair_dots=False, fill_corner_tips=True), crop=False)
    assert (out != TRANSPARENT).sum() > (lab != TRANSPARENT).sum()  # something was filled
    assert shapes(out) == 2
    assert L._holes(out != TRANSPARENT).sum() == hole_before


def test_tight_letter_is_widened_and_the_next_letter_moves():
    """An "n" whose inside is 0.6 mm wide is spread to 1 mm; the letter after it moves
    by the same amount so the gap between them is unchanged."""
    lab = canvas(50, 30)
    cv2.rectangle(lab, mm(10, 8), mm(12, 22), 0, -1)
    cv2.rectangle(lab, mm(12.6, 8), mm(14.6, 22), 0, -1)          # inside 0.6 mm
    cv2.rectangle(lab, mm(10, 8), mm(14.6, 10), 0, -1)
    cv2.rectangle(lab, mm(17, 8), mm(19, 22), 0, -1)              # next letter, 2.4 mm away
    rep = L.Report()
    out = L._spread_narrow(lab, 1.0, rep)
    assert out.shape[1] > lab.shape[1] and rep.notes
    row = np.flatnonzero(out[15 * P] != TRANSPARENT)
    runs = np.split(row, np.flatnonzero(np.diff(row) > 1) + 1)
    assert len(runs) == 3
    inside = runs[1][0] - runs[0][-1] - 1
    gap = runs[2][0] - runs[1][-1] - 1
    assert inside >= 0.95 * P and abs(gap - (17 - 14.6) * P) <= 2


def test_open_gaps_widens_narrow_gaps_but_not_wedges():
    """Two bars 0.5 mm apart get a 0.8 mm gap; the wedge of a V (two strokes meeting at a
    shallow angle) is a corner tip and is left alone."""
    lab = canvas(50, 30)
    cv2.rectangle(lab, mm(5, 5), mm(15, 25), 0, -1)
    cv2.rectangle(lab, mm(15.5, 5), mm(25, 25), 0, -1)            # 0.5 mm gap
    v = np.array([mm(30, 5), mm(33, 5), mm(37, 20), mm(41, 5), mm(44, 5), mm(38.5, 25), mm(35.5, 25)], np.int32)
    cv2.fillPoly(lab, [v], 0)
    before_v = (lab[:, mm(29)[0]:] != TRANSPARENT).sum()
    out, rep = L.apply(lab, L.Fixes(repair_dots=False, open_gaps=True), crop=False)
    row = np.flatnonzero(out[mm(15)[0], :mm(28)[0]] != TRANSPARENT)
    gap = np.diff(row).max() - 1
    assert gap >= 0.8 * P and shapes(out) == 3
    assert (out[:, mm(29)[0]:] != TRANSPARENT).sum() == before_v      # the V's crotch untouched
