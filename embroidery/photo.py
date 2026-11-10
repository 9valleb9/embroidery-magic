"""Clean up a photo of a one-color logo (vinyl on a beanie, a print on a shirt) into a
flat design, and estimate its real color.

Where it sits: ``pipeline.new_session`` calls :func:`clean` instead of the k-means
palette when the upload is a photo (``palette.analyze`` flags it) and photo clean-up is
on. The result is a label map with one color (id 0) and ``TRANSPARENT`` background, so
everything downstream works as for any flat logo.

Why: fabric texture, noise and soft focus give a photo ragged edges, and a plain
threshold also deletes the thinnest strokes (an "e" crossbar disappears and the "e"
reads as a "c"). Lighting and the camera wash out the color, so the sampled color is
duller than the real one.

Steps (:func:`clean`):

1. Denoise: 3 px median filter, then a light Gaussian blur.
2. Ink strength: each pixel's Lab distance from the background color (the median of
   the image border), so dark-on-light, light-on-dark and colored ink all work.
3. Threshold: Otsu on the ink strength, then lowered step by step while that joins
   broken strokes together without growing the ink area much, so thin strokes stay
   connected. A small close (``CLOSE_MM``) mends the last breaks.
4. Smooth outlines by Gaussian-smoothing the points along each contour
   (``SMOOTH_MM``), not by blurring the image, which would round corners and thin
   hairlines.
5. Remove specks and fill pinholes under ``SPECK_MM2``; real counters stay.
6. Lost detail: the skeleton of a looser mask (a lower threshold) is compared with the
   chosen mask. Stroke pieces at least ``LOST_MIN_MM`` long that only the looser mask
   has are "possible lost detail"; ``restore=True`` adds them back from the looser mask,
   in just those areas.

Color (:func:`ink_color`): the median of the confidently-inside pixels (the mask eroded),
plus a "boosted" version with the same hue and more chroma, and the nearest Brother
thread for each.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.filters import threshold_otsu

from . import colors as C
from .palette import TRANSPARENT
from .raster import PX_PER_MM

CLOSE_MM = 0.1        # close breaks this small after thresholding (about 2 px at photo scale)
SMOOTH_MM = 0.25      # Gaussian sigma along each outline
SPECK_MM2 = 0.3       # specks and pinholes smaller than this are removed / filled
LOST_MIN_MM = 0.5     # a stroke piece only the looser mask has, at least this long, is lost detail
LOOSE_FACTOR = 0.7    # the looser mask's threshold, relative to the chosen one
BOOST_CHROMA = 1.6    # chroma multiplier for the boosted color


@dataclass
class PhotoResult:
    """Clean-up result. ``labels`` uses id 0 for ink and ``TRANSPARENT`` for background."""
    labels: np.ndarray
    ink_hex: str
    boosted_hex: str
    background_hex: str
    lost_detail: int = 0                 # pieces of possible lost detail (0 after restoring)
    restored: int = 0                    # pieces restored
    notes: list[str] = field(default_factory=list)


def _border_color(lab: np.ndarray) -> np.ndarray:
    """Median Lab color of the image's outer 3% (the background around the logo)."""
    h, w = lab.shape[:2]
    b = max(2, round(0.03 * min(h, w)))
    edge = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                           lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    return np.median(edge, axis=0)


def _ink_strength(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Denoised image -> (ink strength 0..255 as float32, background Lab)."""
    den = cv2.medianBlur(rgb, 3)
    den = cv2.GaussianBlur(den, (0, 0), 0.6)
    lab = cv2.cvtColor(den, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    bg = _border_color(lab)
    d = np.linalg.norm(lab - bg, axis=2)
    return np.clip(d * 2.55, 0, 255).astype(np.float32), bg


def _components(mask: np.ndarray, min_px: float) -> int:
    """Number of 8-connected components of at least ``min_px`` pixels."""
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int((st[1:, cv2.CC_STAT_AREA] >= min_px).sum())


def _pick_threshold(ink: np.ndarray) -> float:
    """Otsu, lowered while that reconnects strokes without bloating the ink.

    Steps down in 5% of Otsu. A step is taken when it doesn't add components (pieces
    joining, or none appearing) and grows the ink area by under 8%; it stops at the
    first step that adds noise or bloats, or at 60% of Otsu.
    """
    t0 = float(threshold_otsu(ink))
    min_px = SPECK_MM2 * PX_PER_MM ** 2
    best = t0
    m = ink >= t0
    n, area = _components(m, min_px), m.sum()
    for f in np.arange(0.95, 0.59, -0.05):
        t = t0 * f
        m = ink >= t
        n2, a2 = _components(m, min_px), m.sum()
        if n2 > n or a2 > area * 1.08:
            break
        best, n, area = t, n2, a2
    return best


def _smooth_outlines(mask: np.ndarray) -> np.ndarray:
    """Redraw the mask from its contours, each smoothed along its length (holes kept)."""
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(mask.shape, np.uint8)
    if hier is None:
        return mask
    sigma = SMOOTH_MM * PX_PER_MM
    for i, c in enumerate(cnts):
        pts = c[:, 0, :].astype(np.float32)
        if len(pts) > 8:
            k = int(3 * sigma) * 2 + 1
            if len(pts) > k:
                g = cv2.getGaussianKernel(k, sigma).ravel()
                pad = np.concatenate([pts[-(k // 2):], pts, pts[:k // 2]])
                pts = np.stack([np.convolve(pad[:, a], g, mode="valid") for a in (0, 1)], axis=1)
        poly = np.round(pts * 8).astype(np.int32)
        hole = hier[0][i][3] >= 0
        cv2.fillPoly(out, [poly], 0 if hole else 1, lineType=cv2.LINE_8, shift=3)
    return out.astype(bool)


def _despeckle(mask: np.ndarray) -> np.ndarray:
    """Remove specks and fill pinholes smaller than ``SPECK_MM2``."""
    min_px = SPECK_MM2 * PX_PER_MM ** 2
    out = mask.copy()
    for val in (True, False):
        m = (out == val).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        small = np.flatnonzero(st[1:, cv2.CC_STAT_AREA] < min_px) + 1
        if len(small):
            out[np.isin(cc, small)] = not val
    return out


def _lost_detail(mask: np.ndarray, loose: np.ndarray) -> tuple[np.ndarray, int]:
    """Stroke pieces the looser mask has that the chosen mask lost.

    Returns:
        (mask of the looser mask's pixels around each lost piece, number of pieces).
    """
    from skimage.morphology import skeletonize

    grown = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    sk = skeletonize(loose) & ~grown
    n, cc, st, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    # a lost stroke piece is long enough and joins the logo at an end (a crossbar, a
    # shoulder); texture specks the looser threshold picks up float free
    touch = cv2.dilate(grown.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    keep = [i for i in range(1, n)
            if max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) >= LOST_MIN_MM * PX_PER_MM
            and touch[cc == i].any()]
    if not keep:
        return np.zeros_like(mask), 0
    pieces = np.isin(cc, keep).astype(np.uint8)
    # the looser mask's pixels near those skeleton pieces (their stroke)
    near = cv2.dilate(pieces, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(PX_PER_MM) | 1,) * 2)).astype(bool)
    # the restored stroke is the looser mask's pixels near the piece, opened so blobs of
    # texture hanging off it don't come along
    out = near & loose
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return out, len(keep)


def ink_color(rgb: np.ndarray, mask: np.ndarray) -> tuple[str, str]:
    """Sampled ink color (median of the mask eroded by 0.3 mm, so edge pixels blended
    with the background don't count) and a boosted version: same hue and lightness,
    chroma times ``BOOST_CHROMA``."""
    core = cv2.erode(mask.astype(np.uint8), np.ones((round(0.3 * PX_PER_MM) * 2 + 1,) * 2, np.uint8)).astype(bool)
    if core.sum() < 50:
        core = mask
    med = np.median(rgb[core], axis=0)
    hx = C.rgb_to_hex(med)
    lab = C.hex_to_lab(hx)
    boosted = lab.copy()
    boosted[1:] *= BOOST_CHROMA
    return hx, C.lab_to_hex(boosted)


def clean(rgb: np.ndarray, alpha: np.ndarray, restore: bool = False) -> PhotoResult:
    """Turn a photo of a one-color logo into a flat label map. See the module docs.

    Args:
        rgb: HxWx3 uint8 from ``raster.rasterize``.
        alpha: HxW uint8 alpha (transparent pixels are background).
        restore: Add back the possible lost detail from the looser mask.
    """
    ink, bg = _ink_strength(rgb)
    ink[alpha < 128] = 0
    t = _pick_threshold(ink)
    mask = ink >= t
    k = max(3, round(CLOSE_MM * 2 * PX_PER_MM) | 1)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    loose = ink >= t * LOOSE_FACTOR
    lost, n_lost = _lost_detail(mask, loose)
    restored = 0
    if restore and n_lost:
        mask |= lost
        restored, n_lost = n_lost, 0
    mask = _despeckle(_smooth_outlines(mask))
    ink_hex, boosted = ink_color(rgb, mask)
    bg_hex = C.lab_to_hex(bg)
    labels = np.full(mask.shape, TRANSPARENT, np.uint8)
    labels[mask] = 0
    notes = [f"Photo cleaned up: one ink color on a {C.color_name(bg_hex).lower()} background. "
             "Colors from a photo are estimates; lighting and the camera usually make them duller."]
    if restored:
        notes.append(f"Restored {restored} piece(s) of thin detail from a looser threshold.")
    return PhotoResult(labels, ink_hex, boosted, bg_hex, n_lost, restored, notes)


PHOTO_NOISE = 2.0     # median |Laplacian| (gray, at most 1000 px) above this: camera noise / fabric texture


def looks_like_photo(data: bytes) -> bool:
    """True when a raster upload has camera noise or fabric texture everywhere.

    Clean artwork has flat areas, so the median absolute Laplacian of its gray image is
    0; a phone photo (grain, knit texture, JPEG noise) is well above ``PHOTO_NOISE``.
    Measured at the upload's own resolution (at most 1000 px), before any resizing
    smooths the noise away. SVG and unreadable input return False.
    """
    import io

    from PIL import Image, ImageOps

    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L")
    except Exception:
        return False
    g = np.asarray(im, np.float32)
    f = 1000 / max(g.shape)
    if f < 1:
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return float(np.median(np.abs(cv2.Laplacian(g, cv2.CV_32F)))) > PHOTO_NOISE


_THREADS = None


def nearest_thread(hx: str) -> dict:
    """The Brother thread (pyembroidery's PEC chart) closest to ``hx`` by ΔE2000."""
    global _THREADS
    if _THREADS is None:
        import pyembroidery as pe
        _THREADS = [(t.hex_color(), t.description) for t in pe.EmbThreadPec.get_thread_set() if t is not None and t.description]
    best = min(_THREADS, key=lambda t: C.delta_e(hx, t[0]))
    return {"hex": best[0], "name": best[1], "de": round(C.delta_e(hx, best[0]), 1)}
