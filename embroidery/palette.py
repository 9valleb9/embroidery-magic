"""Steps 2–3 — pick the thread palette with k-means and clean up the label map."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

from .colors import rgb_to_hex
from .raster import PX_PER_MM

TRANSPARENT = 255          # label for pixels with no design (alpha < 50%)
_UNASSIGNED = 254          # scratch label during cleanup

FLAT_VARIANCE = 40.0       # max local variance for a pixel to count as "flat" (not anti-aliased)
OFF_COLOR_DIST = 10.0      # a flat pixel this far from every palette color is "unrepresented"
OFF_COLOR_SHARE = 0.005    # palette fits when fewer than this share of flat pixels are unrepresented
PHOTO_DISTINCT = 80        # distinct coarse colors that suggest a photo/gradient
PHOTO_ERR = 6.0            # residual Lab error at 8 colors that suggests a photo/gradient

MODE_KERNEL_PX = 9
MORPH_DISK_MM = 0.35
MIN_PART_MM2 = 1.0


@dataclass
class PaletteResult:
    labels: np.ndarray          # HxW uint8, palette index or TRANSPARENT (already cleaned)
    hexes: list[str]
    suggested_k: int
    warnings: list[dict]


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (...x3) -> float32 CIE Lab (L 0..100)."""
    flat = rgb.reshape(-1, 1, 3).astype(np.float32) / 255.0
    return cv2.cvtColor(flat, cv2.COLOR_RGB2Lab).reshape(rgb.shape[:-1] + (3,))


def _flat_mask(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Pixels with low local variance — i.e. not on an anti-aliased edge."""
    var = np.zeros(rgb.shape[:2], np.float32)
    for c in range(3):
        ch = rgb[..., c].astype(np.float32)
        m = cv2.blur(ch, (5, 5))
        var = np.maximum(var, cv2.blur(ch * ch, (5, 5)) - m * m)
    return (var < FLAT_VARIANCE) & (alpha >= 250)


def _fit(lab: np.ndarray, k: int) -> tuple[KMeans, float, float]:
    """Returns (model, mean error, share of pixels with no close palette color)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(lab)
    dist = km.transform(lab).min(axis=1)
    return km, float(dist.mean()), float((dist > OFF_COLOR_DIST).mean())


def _assign(lab_px: np.ndarray, centers: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(lab_px), np.uint8)
    for i in range(0, len(lab_px), chunk):
        d = ((lab_px[i:i + chunk, None, :] - centers[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out


def analyze(rgb: np.ndarray, alpha: np.ndarray, n_colors: int | None = None) -> PaletteResult:
    opaque = alpha >= 128
    if not opaque.any():
        raise ValueError("The image is fully transparent.")
    flat = _flat_mask(rgb, alpha) & opaque
    if flat.sum() < 500:
        flat = opaque

    rng = np.random.default_rng(0)
    flat_rgb = rgb[flat]
    sample = flat_rgb[rng.choice(len(flat_rgb), min(30_000, len(flat_rgb)), replace=False)]
    lab_sample = _to_lab(sample)
    n_unique = len(np.unique(sample, axis=0))

    warns: list[dict] = []
    max_k = min(8, n_unique)
    fits: dict[int, tuple[KMeans, float, float]] = {}
    for k in range(min(2, max_k), max_k + 1):
        fits[k] = _fit(lab_sample, k)
    # Fewest colors that represent every flat area (small accents included).
    suggested = next((k for k in sorted(fits) if fits[k][2] < OFF_COLOR_SHARE), None)
    if suggested is None:
        # Elbow: stop when one more color buys < 15% less error.
        ks = sorted(fits)
        suggested = ks[-1]
        for a, b in zip(ks, ks[1:]):
            if fits[a][1] - fits[b][1] < 0.15 * fits[a][1]:
                suggested = a
                break

    # Photo / gradient detection on coarse (3 bits per channel) colors.
    codes = (sample >> 5).astype(np.int32)
    codes = codes[:, 0] * 64 + codes[:, 1] * 8 + codes[:, 2]
    counts = np.bincount(codes, minlength=512)
    distinct = int((counts >= max(5, 0.001 * len(sample))).sum())
    err8 = fits[max(fits)][1]
    photo = distinct > PHOTO_DISTINCT or (max(fits) == 8 and err8 > PHOTO_ERR)
    if photo:
        warns.append({
            "level": "warn", "code": "photo",
            "message": f"This looks like a photo or gradient ({distinct} distinct color groups). "
                       "Embroidery needs flat colors — results will be posterized.",
        })
    elif max(fits) == 8 and fits[8][2] >= OFF_COLOR_SHARE:
        warns.append({
            "level": "warn", "code": "too_many_colors",
            "message": "The design seems to use more than 8 colors; some will be merged into their nearest thread.",
        })

    k = int(np.clip(n_colors, 2, 8)) if n_colors else suggested
    k = min(k, n_unique)
    km = fits[k][0] if k in fits else _fit(lab_sample, k)[0]
    centers = km.cluster_centers_.astype(np.float32)

    labels = np.full(alpha.shape, TRANSPARENT, np.uint8)
    labels[opaque] = _assign(_to_lab(rgb[opaque]), centers)

    # Thread color = mean of the flat pixels in each cluster (truer than the Lab centroid).
    flat_labels = labels[flat]
    hexes = []
    for i in range(k):
        px = flat_rgb[flat_labels == i]
        hexes.append(rgb_to_hex(px.mean(0)) if len(px) else rgb_to_hex(rgb[labels == i].mean(0)))

    labels = cleanup(labels, k)
    return PaletteResult(labels=labels, hexes=hexes, suggested_k=suggested, warnings=warns)


def _fill_unassigned(labels: np.ndarray) -> np.ndarray:
    """Give every unassigned pixel the label of its nearest assigned pixel."""
    holes = labels == _UNASSIGNED
    if not holes.any():
        return labels
    if holes.all():
        return np.full_like(labels, TRANSPARENT)
    _, (iy, ix) = ndi.distance_transform_edt(holes, return_indices=True)
    return labels[iy, ix]


def _disk(diameter_px: int) -> np.ndarray:
    d = max(3, diameter_px | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def cleanup(labels: np.ndarray, k: int) -> np.ndarray:
    ids = list(range(k)) + [TRANSPARENT]

    # Majority (mode) filter. This is the label-safe form of a 9 px median blur:
    # a median over label *ids* can invent a third color where two others meet.
    best = np.full(labels.shape, -1.0, np.float32)
    out = labels.copy()
    for i in ids:
        s = cv2.boxFilter((labels == i).astype(np.float32), -1, (MODE_KERNEL_PX, MODE_KERNEL_PX))
        upd = s > best
        out[upd] = i
        best[upd] = s[upd]
    labels = out

    # Morphological open then close per color, ~0.35 mm disk.
    disk = _disk(round(MORPH_DISK_MM * PX_PER_MM))
    cleaned = {}
    for i in ids:
        m = (labels == i).astype(np.uint8)
        if not m.any():
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk)
        cleaned[i] = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk).astype(bool)
    out = np.full(labels.shape, _UNASSIGNED, np.uint8)
    for i, m in cleaned.items():        # a pixel keeps its own color if it survived
        out[m & (labels == i)] = i
    for i, m in cleaned.items():        # else it joins a color that closed over it
        out[m & (out == _UNASSIGNED)] = i
    labels = _fill_unassigned(out)

    # Absorb specks smaller than 1 mm² into their surroundings, so dropping
    # them never leaves an unstitched gap.
    min_px = MIN_PART_MM2 * PX_PER_MM ** 2
    for _ in range(2):
        changed = False
        for i in ids:
            n, cc, stats, _ = cv2.connectedComponentsWithStats((labels == i).astype(np.uint8), connectivity=8)
            small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < min_px) + 1
            if len(small):
                labels[np.isin(cc, small)] = _UNASSIGNED
                changed = True
        if not changed:
            break
        labels = _fill_unassigned(labels)
    return labels
