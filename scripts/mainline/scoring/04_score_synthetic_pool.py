#!/usr/bin/env python3
"""
Stage 04: score and filter a synthetic pool before benchmark training.

Two-stage quality gate (consistent with legacy pipeline):
  1. Classifier confidence: run a pre-trained CNN on each synthetic image,
     keep only those where argmax == target class AND max(softmax) >= threshold.
  2. Sharpness floor: compute Laplacian variance for each image, drop those
     below a percentile floor derived from real training images.

Inputs:
  - synthetic manifest from stage 03 (or merged manifest)
  - pre-trained classifier checkpoint (e.g. from a real_only benchmark run)
  - real training manifest (for sharpness reference distribution)

Outputs:
  - scored manifest: all items with confidence and sharpness scores attached
  - filtered manifest: items passing both gates

Usage:
    python -m scripts.mainline.scoring.04_score_synthetic_pool \
        --config configs/mainline/scoring/score_v1.yaml

    python -m scripts.mainline.scoring.04_score_synthetic_pool \
        --synthetic-manifest results/mainline/generation/runs/combined.json \
        --classifier-ckpt results/mainline/benchmark/.../best_model.pt \
        --real-manifest results/mainline/data/heldout_domain_b_raabin/train_manifest.json \
        --output-root results/mainline/scoring/run_01
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mainline.common.config import apply_overrides, dump_yaml_config, load_yaml_config
from scripts.mainline.common.constants import CLASS_TO_IDX, CLASSES
from scripts.mainline.common.manifests import (
    load_manifest_items,
    write_manifest_payload,
)
from scripts.mainline.common.reporting import markdown_table, write_json, write_text
from scripts.mainline.common.runtime import (
    build_backbone,
    ensure_dir,
    get_device,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score and filter a synthetic pool."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to scoring YAML config.",
    )
    parser.add_argument("--synthetic-manifest", type=str, default=None)
    parser.add_argument("--classifier-ckpt", type=str, default=None)
    parser.add_argument("--real-manifest", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--sharp-floor-pctile", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    if args.config:
        config = load_yaml_config(resolve_project_path(PROJECT_ROOT, args.config))
    else:
        config = {}

    config = apply_overrides(config, {
        "synthetic_manifest": args.synthetic_manifest,
        "classifier_ckpt": args.classifier_ckpt,
        "real_manifest": args.real_manifest,
        "output_root": args.output_root,
        "conf_threshold": args.conf_threshold,
        "sharp_floor_pctile": args.sharp_floor_pctile,
        "backbone": args.backbone,
    })

    # Defaults
    config.setdefault("conf_threshold", 0.7)
    config.setdefault("sharp_floor_pctile", 20)
    config.setdefault("backbone", "efficientnet_b0")
    config.setdefault("image_size", 224)

    required = ["synthetic_manifest", "classifier_ckpt", "real_manifest", "output_root"]
    missing = [k for k in required if not config.get(k)]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    return {
        **config,
        "conf_threshold": float(config["conf_threshold"]),
        "sharp_floor_pctile": float(config["sharp_floor_pctile"]),
        "image_size": int(config["image_size"]),
    }


def compute_laplacian_variance(image_path: str) -> float:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_real_sharpness_distribution(items: list[dict]) -> np.ndarray:
    values = []
    for item in tqdm(items, desc="sharpness/real", leave=False):
        values.append(compute_laplacian_variance(item["file_abs"]))
    return np.array(values)


def score_synthetic_items(
    items: list[dict],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
) -> list[dict]:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    scored = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(items, desc="scoring"):
            path = item["file_abs"]
            if not Path(path).exists():
                continue

            # Classifier confidence
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            predicted_idx = int(probs.argmax().item())
            confidence = float(probs.max().item())
            target_idx = CLASS_TO_IDX.get(item["class_name"], -1)
            class_correct = predicted_idx == target_idx

            # Sharpness
            sharpness = compute_laplacian_variance(path)

            scored_item = copy.deepcopy(item)
            scored_item["score_confidence"] = round(confidence, 4)
            scored_item["score_predicted_class"] = CLASSES[predicted_idx] if predicted_idx < len(CLASSES) else "unknown"
            scored_item["score_class_correct"] = class_correct
            scored_item["score_sharpness"] = round(sharpness, 2)
            scored.append(scored_item)

    return scored


def apply_quality_gate(
    scored_items: list[dict],
    conf_threshold: float,
    sharpness_floor: float,
) -> tuple[list[dict], dict]:
    passed = []
    rejected_conf = 0
    rejected_sharp = 0
    rejected_both = 0

    for item in scored_items:
        fail_conf = not item["score_class_correct"] or item["score_confidence"] < conf_threshold
        fail_sharp = item["score_sharpness"] < sharpness_floor
        if fail_conf and fail_sharp:
            rejected_both += 1
        elif fail_conf:
            rejected_conf += 1
        elif fail_sharp:
            rejected_sharp += 1
        else:
            passed.append(item)

    stats = {
        "total_scored": len(scored_items),
        "passed": len(passed),
        "rejected_confidence": rejected_conf,
        "rejected_sharpness": rejected_sharp,
        "rejected_both": rejected_both,
        "conf_threshold": conf_threshold,
        "sharpness_floor": round(float(sharpness_floor), 2),
        "pass_rate": round(len(passed) / max(len(scored_items), 1), 4),
    }
    return passed, stats


def render_report(config: dict, gate_stats: dict, scored_items: list[dict], filtered_items: list[dict]) -> str:
    scored_counter = Counter((item["class_name"], item.get("domain", "?")) for item in scored_items)
    filtered_counter = Counter((item["class_name"], item.get("domain", "?")) for item in filtered_items)

    scored_rows = [[cls, dom, cnt] for (cls, dom), cnt in sorted(scored_counter.items())]
    filtered_rows = [[cls, dom, cnt] for (cls, dom), cnt in sorted(filtered_counter.items())]

    lines = [
        "# Synthetic Pool Scoring Report",
        "",
        "## Gate Settings",
        "",
        f"- Classifier checkpoint: `{config['classifier_ckpt']}`",
        f"- Backbone: `{config['backbone']}`",
        f"- Confidence threshold: `{config['conf_threshold']}`",
        f"- Sharpness floor percentile: `{config['sharp_floor_pctile']}`",
        f"- Computed sharpness floor: `{gate_stats['sharpness_floor']}`",
        "",
        "## Gate Results",
        "",
        f"- Total scored: `{gate_stats['total_scored']}`",
        f"- Passed: `{gate_stats['passed']}`",
        f"- Rejected (confidence): `{gate_stats['rejected_confidence']}`",
        f"- Rejected (sharpness): `{gate_stats['rejected_sharpness']}`",
        f"- Rejected (both): `{gate_stats['rejected_both']}`",
        f"- Pass rate: `{gate_stats['pass_rate']}`",
        "",
        "## Scored Items by Class/Domain",
        "",
        markdown_table(["Class", "Domain", "Count"], scored_rows),
        "",
        "## Filtered Items by Class/Domain",
        "",
        markdown_table(["Class", "Domain", "Count"], filtered_rows),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = build_config(args)

    synth_manifest_path = resolve_project_path(PROJECT_ROOT, config["synthetic_manifest"])
    ckpt_path = resolve_project_path(PROJECT_ROOT, config["classifier_ckpt"])
    real_manifest_path = resolve_project_path(PROJECT_ROOT, config["real_manifest"])
    output_root = ensure_dir(resolve_project_path(PROJECT_ROOT, config["output_root"]))

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")
    if not real_manifest_path.exists():
        raise FileNotFoundError(f"Real manifest not found: {real_manifest_path}")

    # Load classifier
    device = get_device()
    model = build_backbone(config["backbone"], num_classes=len(CLASSES))
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Compute sharpness reference from real images
    real_items = load_manifest_items(real_manifest_path)
    real_sharpness = compute_real_sharpness_distribution(real_items)
    sharpness_floor = float(np.percentile(real_sharpness, config["sharp_floor_pctile"]))

    # Score synthetic items
    synth_items = load_manifest_items(synth_manifest_path)
    scored_items = score_synthetic_items(synth_items, model, device, config["image_size"])

    # Apply quality gate
    filtered_items, gate_stats = apply_quality_gate(
        scored_items,
        conf_threshold=config["conf_threshold"],
        sharpness_floor=sharpness_floor,
    )

    # Write outputs
    write_json(
        output_root / "scored_manifest.json",
        write_manifest_payload("scored_synthetic_pool", scored_items, {"gate_stats": gate_stats}),
    )
    write_json(
        output_root / "filtered_manifest.json",
        write_manifest_payload("filtered_synthetic_pool", filtered_items, {"gate_stats": gate_stats}),
    )
    write_json(output_root / "gate_stats.json", gate_stats)
    write_text(output_root / "report.md", render_report(config, gate_stats, scored_items, filtered_items))
    if args.config:
        dump_yaml_config(output_root / "resolved_config.yaml", config)
    print(f"Scored {len(scored_items)} items, {len(filtered_items)} passed quality gate ({gate_stats['pass_rate']:.1%})")
    print(f"Wrote outputs to: {output_root}")


if __name__ == "__main__":
    main()
