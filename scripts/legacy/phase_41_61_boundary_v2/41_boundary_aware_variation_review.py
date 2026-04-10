"""
Script 41: Boundary-aware variation review for retuned cross-domain generations.

Goal:
  Re-score generated images from the perspective the current pipeline is missing:
  1. keep the center cell relatively stable
  2. diversify the background/domain style
  3. move samples closer to the classifier boundary instead of maximizing easy correctness

Outputs:
  results/boundary_aware_variation/{class_name}/report.json
  results/boundary_aware_variation/{class_name}/summary.md

Example:
  python3 scripts/legacy/phase_41_61_boundary_v2/41_boundary_aware_variation_review.py --class_name monocyte
  python3 scripts/legacy/phase_41_61_boundary_v2/41_boundary_aware_variation_review.py --class_name eosinophil
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).parent.parent
REPORT_ROOT = ROOT / "results" / "diverse_generation"
OUT_ROOT = ROOT / "results" / "boundary_aware_variation"
CNN_CKPT = ROOT / "models" / "multidomain_cnn.pt"

CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cnn(device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    ckpt = torch.load(CNN_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(device)


@torch.no_grad()
def cnn_probs(model: nn.Module, img: Image.Image, device: torch.device) -> np.ndarray:
    x = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    return probs


def extract_cell_mask(img: Image.Image) -> np.ndarray:
    """
    Heuristic center-focused WBC mask.
    We intentionally bias toward preserving the central cell while allowing the
    background region to absorb most of the variation score.
    """
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    sat_blur = cv2.GaussianBlur(sat, (5, 5), 0)
    val_blur = cv2.GaussianBlur(val, (5, 5), 0)

    sat_thr = max(20, int(np.percentile(sat_blur, 60)))
    val_thr = int(np.percentile(val_blur, 85))
    mask = ((sat_blur > sat_thr) & (val_blur < val_thr)).astype(np.uint8) * 255

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

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = None
    best_score = -1.0
    for i, area in enumerate(areas, start=1):
        x, y, ww, hh, _ = stats[i]
        comp_cx = x + ww / 2.0
        comp_cy = y + hh / 2.0
        dist = np.hypot(comp_cx - cx, comp_cy - cy)
        score = area - 2.5 * dist
        if score > best_score:
            best_score = score
            best_idx = i
    mask = (labels == best_idx).astype(np.uint8) * 255
    mask = cv2.dilate(mask, kernel, iterations=2)
    if mask.sum() < 0.02 * h * w * 255:
        return fallback_center_mask(mask.shape)
    return mask


def fallback_center_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    ellipse = (((xx - cx) / (0.22 * w)) ** 2 + ((yy - cy) / (0.18 * h)) ** 2) <= 1.0
    return ellipse.astype(np.uint8) * 255


def masked_ssim(img_a: Image.Image, img_b: Image.Image, mask: np.ndarray, size: int = 224) -> float:
    a = np.array(img_a.resize((size, size)).convert("RGB")).astype(np.float32) / 255.0
    b = np.array(img_b.resize((size, size)).convert("RGB")).astype(np.float32) / 255.0
    m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 0
    if m.sum() < 32:
        return float("nan")
    diff = (a - b) ** 2
    mse = diff[m].mean()
    return float(max(0.0, 1.0 - mse * 3.0))


def entropy_and_margin(probs: np.ndarray, target_idx: int) -> tuple[float, float, float]:
    eps = 1e-8
    entropy = float(-(probs * np.log(probs + eps)).sum())
    target_p = float(probs[target_idx])
    max_other = float(np.max(np.delete(probs, target_idx)))
    margin = target_p - max_other
    return entropy, margin, target_p


def review_class(class_name: str):
    report_path = REPORT_ROOT / class_name / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    out_dir = OUT_ROOT / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = load_cnn(device)
    target_idx = CLASSES.index(class_name)
    data = json.loads(report_path.read_text())

    reviewed = []
    for item in data["per_image"]:
        gen_path = ROOT / item["file"]
        gen_img = Image.open(gen_path).convert("RGB")

        ref_domain = item["ref_domain"]
        ref_class_dir = ROOT / "data" / "processed_multidomain" / ref_domain / class_name
        ref_name = Path(item["file"]).name
        ref_idx = item["inp_idx"]
        candidates = sorted(ref_class_dir.glob("*"))
        if ref_idx >= len(candidates):
            continue
        ref_img = Image.open(candidates[ref_idx]).convert("RGB")

        cell_mask = extract_cell_mask(ref_img)
        bg_mask = np.where(cell_mask > 0, 0, 255).astype(np.uint8)

        cell_ssim = masked_ssim(ref_img, gen_img, cell_mask)
        bg_ssim = masked_ssim(ref_img, gen_img, bg_mask)

        probs = cnn_probs(model, gen_img, device)
        entropy, margin, target_prob = entropy_and_margin(probs, target_idx)
        near_boundary = bool(item["cnn_pred"] == class_name and margin < 0.20)
        variation_score = float((1.0 - np.nan_to_num(bg_ssim, nan=1.0)) + 0.5 * entropy - 0.25 * max(margin, 0.0))
        preservation_score = float(np.nan_to_num(cell_ssim, nan=0.0))

        reviewed.append({
            **item,
            "cell_ssim": round(cell_ssim, 4) if not np.isnan(cell_ssim) else None,
            "background_ssim": round(bg_ssim, 4) if not np.isnan(bg_ssim) else None,
            "cnn_entropy": round(entropy, 4),
            "target_margin": round(margin, 4),
            "target_prob": round(target_prob, 4),
            "near_boundary": near_boundary,
            "variation_score": round(variation_score, 4),
            "preservation_score": round(preservation_score, 4),
        })

    by_variation = sorted(reviewed, key=lambda r: (-r["variation_score"], -(r["preservation_score"] or 0.0)))
    boundary_hits = [r for r in reviewed if r["near_boundary"]]
    payload = {
        "class_name": class_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_images": len(reviewed),
        "n_near_boundary": len(boundary_hits),
        "mean_cell_ssim": round(float(np.nanmean([r["cell_ssim"] for r in reviewed])), 4),
        "mean_background_ssim": round(float(np.nanmean([r["background_ssim"] for r in reviewed])), 4),
        "mean_entropy": round(float(np.mean([r["cnn_entropy"] for r in reviewed])), 4),
        "mean_margin": round(float(np.mean([r["target_margin"] for r in reviewed])), 4),
        "top_variation_examples": by_variation[:12],
        "top_boundary_examples": sorted(boundary_hits, key=lambda r: r["target_margin"])[:12],
        "items": reviewed,
    }
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Boundary-Aware Variation Review: {class_name}",
        "",
        f"- Source report: `{report_path}`",
        f"- Images reviewed: `{payload['n_images']}`",
        f"- Near-boundary images: `{payload['n_near_boundary']}`",
        f"- Mean cell SSIM: `{payload['mean_cell_ssim']}`",
        f"- Mean background SSIM: `{payload['mean_background_ssim']}`",
        f"- Mean entropy: `{payload['mean_entropy']}`",
        f"- Mean margin: `{payload['mean_margin']}`",
        "",
        "## Top Variation Examples",
        "",
        "| file | ref -> prompt | ds | cell_ssim | bg_ssim | entropy | margin | score |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["top_variation_examples"]:
        lines.append(
            f"| `{item['file']}` | {item['ref_domain_short']} -> {item['prompt_domain_short']} | "
            f"{item['denoise']} | {item['cell_ssim']} | {item['background_ssim']} | "
            f"{item['cnn_entropy']} | {item['target_margin']} | {item['variation_score']} |"
        )
    lines.extend([
        "",
        "## Top Near-Boundary Examples",
        "",
        "| file | ref -> prompt | ds | target_prob | margin | entropy |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for item in payload["top_boundary_examples"]:
        lines.append(
            f"| `{item['file']}` | {item['ref_domain_short']} -> {item['prompt_domain_short']} | "
            f"{item['denoise']} | {item['target_prob']} | {item['target_margin']} | {item['cnn_entropy']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {class_name} -> {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Boundary-aware variation review")
    parser.add_argument("--class_name", choices=CLASSES, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    review_class(args.class_name)


if __name__ == "__main__":
    main()
