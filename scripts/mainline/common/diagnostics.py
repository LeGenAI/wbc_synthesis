"""Generation-side diagnostics shared by stage 03 and stage 04."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_skimage


def ssim_pair(img_a: Image.Image, img_b: Image.Image, size: int = 256) -> float:
    a = np.array(img_a.convert("RGB").resize((size, size), Image.LANCZOS)) / 255.0
    b = np.array(img_b.convert("RGB").resize((size, size), Image.LANCZOS)) / 255.0
    return float(ssim_skimage(a, b, data_range=1.0, channel_axis=2))


def fallback_center_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    ellipse = (((xx - cx) / (0.22 * w)) ** 2 + ((yy - cy) / (0.18 * h)) ** 2) <= 1.0
    return ellipse.astype(np.uint8) * 255


def extract_cell_mask(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    val = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)

    sat_thr = max(20, int(np.percentile(sat, 60)))
    val_thr = int(np.percentile(val, 85))
    mask = ((sat > sat_thr) & (val < val_thr)).astype(np.uint8) * 255

    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    center_weight = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.22 * w) ** 2)))
    mask = np.where(center_weight > 0.15, mask, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return fallback_center_mask(mask.shape)

    best_idx = None
    best_score = -1.0
    for idx in range(1, num_labels):
        x, y, ww, hh, area = stats[idx]
        comp_cx = x + ww / 2.0
        comp_cy = y + hh / 2.0
        dist = np.hypot(comp_cx - cx, comp_cy - cy)
        score = area - 2.5 * dist
        if score > best_score:
            best_score = score
            best_idx = idx

    mask = (labels == best_idx).astype(np.uint8) * 255
    mask = cv2.dilate(mask, kernel, iterations=2)
    if mask.sum() < 0.02 * h * w * 255:
        return fallback_center_mask(mask.shape)
    return mask


def masked_similarity(img_a: Image.Image, img_b: Image.Image, mask: np.ndarray, size: int = 224) -> float:
    a = np.array(img_a.resize((size, size), Image.LANCZOS).convert("RGB")).astype(np.float32) / 255.0
    b = np.array(img_b.resize((size, size), Image.LANCZOS).convert("RGB")).astype(np.float32) / 255.0
    m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 0
    if int(m.sum()) < 32:
        return 0.0
    mse = ((a - b) ** 2)[m].mean()
    return float(max(0.0, 1.0 - mse * 3.0))


def compute_reference_diagnostics(
    ref_img: Image.Image,
    synth_img: Image.Image,
) -> dict[str, float]:
    cell_mask = extract_cell_mask(ref_img)
    background_mask = np.where(cell_mask > 0, 0, 255).astype(np.uint8)
    ssim = ssim_pair(ref_img, synth_img)
    cell_ssim = masked_similarity(ref_img, synth_img, cell_mask)
    background_ssim = masked_similarity(ref_img, synth_img, background_mask)
    return {
        "ssim": round(float(ssim), 4),
        "cell_ssim": round(float(cell_ssim), 4),
        "background_ssim": round(float(background_ssim), 4),
        "region_gap": round(float(cell_ssim - background_ssim), 4),
    }
