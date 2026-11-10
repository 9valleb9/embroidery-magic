"""Tests for embroidery/photo.py: photo detection, clean-up of a photo of a one-color
logo, lost-detail restore and the ink color estimate."""
import io

import cv2
import numpy as np
from PIL import Image

from embroidery import colors as C
from embroidery import photo
from embroidery.palette import TRANSPARENT


def logo_photo(thin_crossbar=True, size=(360, 200)):
    """A fake phone photo: a dull blue "e" (ring plus thin crossbar) and a bar on a
    noisy, ribbed gray knit, slightly out of focus. Returns (PNG bytes, ink mask)."""
    w, h = size
    ink = np.zeros((h, w), np.uint8)
    cv2.circle(ink, (110, 100), 45, 1, 9)                         # the bowl of an "e"
    if thin_crossbar:
        cv2.line(ink, (68, 100), (152, 100), 1, 2)                # its thin crossbar
    cv2.rectangle(ink, (200, 50), (215, 150), 1, -1)              # a stem
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[:h, :w]
    knit = 18 * np.sin(xx / 2.2) * np.sin(yy / 3.1)
    bg = np.dstack([150 + knit, 152 + knit, 158 + knit])
    img = bg * (1 - ink[..., None]) + np.array([95, 110, 160]) * ink[..., None]
    img = cv2.GaussianBlur(img + rng.normal(0, 9, img.shape), (0, 0), 0.8)
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue(), ink


def test_looks_like_photo():
    data, _ = logo_photo()
    assert photo.looks_like_photo(data)
    flat = io.BytesIO()
    a = np.full((200, 300, 3), 255, np.uint8)
    cv2.circle(a, (150, 100), 50, (20, 60, 160), -1)
    Image.fromarray(a).save(flat, "PNG")
    assert not photo.looks_like_photo(flat.getvalue())


def test_clean_photo_to_one_flat_color():
    data, ink = logo_photo()
    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    r = photo.clean(rgb, np.full(rgb.shape[:2], 255, np.uint8))
    m = r.labels != TRANSPARENT
    assert set(np.unique(r.labels)) <= {0, TRANSPARENT}
    # the knit texture is gone: what's left is the logo (2 shapes: the "e" and the stem)
    n = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1
    assert n == 2
    assert (m & ink.astype(bool)).sum() >= 0.85 * ink.sum()
    assert m[98:103, 75:145].any(axis=0).all()                      # the 2 px crossbar survives: an "e", not a "c"
    # the color estimate: sampled near the real ink, boosted more saturated, same hue
    real = C.rgb_to_hex((95, 110, 160))
    assert C.delta_e(r.ink_hex, real) < 12
    lab, blab = C.hex_to_lab(r.ink_hex), C.hex_to_lab(r.boosted_hex)
    assert np.hypot(*blab[1:]) > np.hypot(*lab[1:])
    assert photo.nearest_thread(r.boosted_hex)["name"]


def test_restore_lost_detail_only_adds():
    data, _ = logo_photo()
    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    a = np.full(rgb.shape[:2], 255, np.uint8)
    plain, restored = photo.clean(rgb, a), photo.clean(rgb, a, restore=True)
    assert restored.lost_detail == 0 and restored.restored == plain.lost_detail
    assert (restored.labels != TRANSPARENT).sum() >= (plain.labels != TRANSPARENT).sum()
