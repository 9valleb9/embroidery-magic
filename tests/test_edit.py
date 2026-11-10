"""Tests for embroidery/edit.py and /api/edit: cutting touching letters apart, rebuilding
a cut edge, moving a letter with the rest of its word, and the on-ink check."""
import cv2
import numpy as np
import pytest

from embroidery import edit
from embroidery.palette import TRANSPARENT
from embroidery.raster import PX_PER_MM as P


def canvas(w=60, h=30):
    return np.full((h * P, w * P), TRANSPARENT, np.uint8)


def box(lab, x0, y0, x1, y1):
    cv2.rectangle(lab, (int(x0 * P), int(y0 * P)), (int(x1 * P), int(y1 * P)), 0, -1)


def shapes(lab):
    return cv2.connectedComponents((lab != TRANSPARENT).astype(np.uint8), connectivity=8)[0] - 1


def word():
    """Three letters 1.5 mm apart, the second touching the first through a bridge, and a
    fourth letter after a word gap."""
    lab = canvas()
    for x in (5, 10, 15):
        box(lab, x, 5, x + 3.5, 20)
    box(lab, 8, 12, 10.5, 13)                     # the "t" touching the "h"
    box(lab, 30, 5, 33.5, 20)                     # next word
    return lab


def test_cut_separates_touching_letters():
    lab = word()
    assert shapes(lab) == 3
    out, note, line = edit.apply_op(lab, {"op": "cut", "x0": 9.2, "y0": 10, "x1": 9.2, "y1": 15}, None)
    assert shapes(out) == 4 and "became" in note and line


def test_cut_off_the_design_is_an_error():
    with pytest.raises(edit.EditError):
        edit.cut(word(), (40, 25), (50, 25))


def test_move_takes_the_rest_of_the_word_but_not_the_next_word():
    lab, _, _ = edit.apply_op(word(), {"op": "cut", "x0": 9.2, "y0": 10, "x1": 9.2, "y1": 15}, None)
    out, k = edit.move(lab, (11, 8), 1.0, 0)
    assert k == 2                                 # the clicked letter and the one after it
    row = np.flatnonzero(out[8 * P] != TRANSPARENT)
    runs = np.split(row, np.flatnonzero(np.diff(row) > 1) + 1)
    starts = [r[0] / P for r in runs]
    # gaps: first letter -> moved letter grows by 1 mm, the moved letters keep theirs,
    # and the next word stays where it was
    assert abs((starts[1] - starts[0]) - 6.0) < 0.15 and abs((starts[2] - starts[1]) - 5.0) < 0.15
    assert abs((starts[3] - starts[0]) - 25.0) < 0.15


def test_click_must_land_on_a_letter():
    with pytest.raises(edit.EditError, match="isn't on a letter"):
        edit.move(word(), (12, 3), 1.0, 0)        # above the letters: background


def test_rebuild_mirrors_the_cut_edge():
    """A 2 mm bar cut lengthwise down the middle keeps one half; rebuilding that half
    mirrors it back to about full width."""
    lab = canvas()
    box(lab, 10, 10, 40, 12)
    half = lab.copy()
    half[int(11 * P):] = TRANSPARENT              # the lower half was "under" another letter
    out = edit.rebuild(half, (10, 11.2), (40, 11.2), edit.side_of((10, 11.2), (40, 11.2), (20, 10.5)))
    col = np.flatnonzero(out[:, 25 * P] != TRANSPARENT)
    assert (col.max() - col.min()) / P >= 1.9


def test_edit_api_with_undo(client=None):
    from fastapi.testclient import TestClient
    from app import app
    from tests.fixtures import flat_logo_png
    c = TestClient(app)
    s = c.post("/api/analyze", files={"file": ("logo.png", flat_logo_png(), "image/png")},
               data={"width_mm": "80", "height_mm": "80"}).json()
    r = c.post("/api/edit", json={"session_id": s["session_id"], "op": "move", "x": 1, "y": 1, "dx": 1})
    assert r.status_code == 400 and "isn't on a letter" in r.json()["detail"]
    r = c.post("/api/edit", json={"session_id": s["session_id"], "op": "undo"})
    assert r.status_code == 400
