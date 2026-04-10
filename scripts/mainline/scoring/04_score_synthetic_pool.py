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
from scripts.mainline.common.constants import CLASS_TO_IDX, CLASSES, HARD_CLASSES
from scripts.mainline.common.diagnostics import compute_reference_diagnostics
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
    parser.add_argument("--disable-reference-diagnostics", action="store_true")
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
        "enable_reference_diagnostics": False if args.disable_reference_diagnostics else None,
    })

    # Defaults
    config.setdefault("conf_threshold", 0.7)
    config.setdefault("sharp_floor_pctile", 20)
    config.setdefault("backbone", "efficientnet_b0")
    config.setdefault("image_size", 224)
    config.setdefault("enable_reference_diagnostics", True)

    required = ["synthetic_manifest", "classifier_ckpt", "real_manifest", "output_root"]
    missing = [k for k in required if not config.get(k)]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    return {
        **config,
        "conf_threshold": float(config["conf_threshold"]),
        "sharp_floor_pctile": float(config["sharp_floor_pctile"]),
        "image_size": int(config["image_size"]),
        "enable_reference_diagnostics": bool(config["enable_reference_diagnostics"]),
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
    enable_reference_diagnostics: bool,
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

            if enable_reference_diagnostics:
                ref_path = item.get("ref_file_abs")
                if ref_path and Path(ref_path).exists():
                    ref_image = Image.open(ref_path).convert("RGB")
                    diagnostics = compute_reference_diagnostics(ref_image, image)
                    scored_item["score_ssim"] = diagnostics["ssim"]
                    scored_item["score_cell_ssim"] = diagnostics["cell_ssim"]
                    scored_item["score_background_ssim"] = diagnostics["background_ssim"]
                    scored_item["score_region_gap"] = diagnostics["region_gap"]
            scored.append(scored_item)

    return scored


def summarize_reference_diagnostics(scored_items: list[dict]) -> dict:
    diagnostic_items = [item for item in scored_items if "score_ssim" in item]
    if not diagnostic_items:
        return {
            "n_with_reference_diagnostics": 0,
            "hard_classes": HARD_CLASSES,
            "overall": None,
            "per_class": {},
        }

    def mean_metric(items: list[dict], key: str) -> float:
        return round(float(np.mean([item[key] for item in items])), 4)

    per_class = {}
    for class_name in CLASSES:
        class_items = [item for item in diagnostic_items if item["class_name"] == class_name]
        if not class_items:
            continue
        per_class[class_name] = {
            "n": len(class_items),
            "ssim_mean": mean_metric(class_items, "score_ssim"),
            "cell_ssim_mean": mean_metric(class_items, "score_cell_ssim"),
            "background_ssim_mean": mean_metric(class_items, "score_background_ssim"),
            "region_gap_mean": mean_metric(class_items, "score_region_gap"),
        }

    return {
        "n_with_reference_diagnostics": len(diagnostic_items),
        "hard_classes": HARD_CLASSES,
        "overall": {
            "ssim_mean": mean_metric(diagnostic_items, "score_ssim"),
            "cell_ssim_mean": mean_metric(diagnostic_items, "score_cell_ssim"),
            "background_ssim_mean": mean_metric(diagnostic_items, "score_background_ssim"),
            "region_gap_mean": mean_metric(diagnostic_items, "score_region_gap"),
        },
        "per_class": per_class,
    }


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


def render_report(
    config: dict,
    gate_stats: dict,
    diagnostic_summary: dict,
    scored_items: list[dict],
    filtered_items: list[dict],
) -> str:
    scored_counter = Counter((item["class_name"], item.get("domain", "?")) for item in scored_items)
    filtered_counter = Counter((item["class_name"], item.get("domain", "?")) for item in filtered_items)

    scored_rows = [[cls, dom, cnt] for (cls, dom), cnt in sorted(scored_counter.items())]
    filtered_rows = [[cls, dom, cnt] for (cls, dom), cnt in sorted(filtered_counter.items())]
    diagnostic_rows = []
    for class_name, values in diagnostic_summary["per_class"].items():
        diagnostic_rows.append(
            [
                class_name,
                values["n"],
                values["ssim_mean"],
                values["cell_ssim_mean"],
                values["background_ssim_mean"],
                values["region_gap_mean"],
                "yes" if class_name in HARD_CLASSES else "no",
            ]
        )

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
        f"- Reference diagnostics enabled: `{config['enable_reference_diagnostics']}`",
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
        "## Generation Diagnostics",
        "",
    ]
    if diagnostic_summary["overall"] is None:
        lines.extend(
            [
                "- No reference-linked diagnostics were available in this manifest.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- N with diagnostics: `{diagnostic_summary['n_with_reference_diagnostics']}`",
                f"- Hard classes: `{diagnostic_summary['hard_classes']}`",
                f"- Overall SSIM mean: `{diagnostic_summary['overall']['ssim_mean']}`",
                f"- Overall cell SSIM mean: `{diagnostic_summary['overall']['cell_ssim_mean']}`",
                f"- Overall background SSIM mean: `{diagnostic_summary['overall']['background_ssim_mean']}`",
                f"- Overall region gap mean: `{diagnostic_summary['overall']['region_gap_mean']}`",
                "",
                markdown_table(
                    ["Class", "N", "SSIM", "Cell SSIM", "Background SSIM", "Region Gap", "Hard Class"],
                    diagnostic_rows,
                ),
                "",
            ]
        )
    lines.extend(
        [
        "## Scored Items by Class/Domain",
        "",
        markdown_table(["Class", "Domain", "Count"], scored_rows),
        "",
        "## Filtered Items by Class/Domain",
        "",
        markdown_table(["Class", "Domain", "Count"], filtered_rows),
        "",
        ]
    )
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
    scored_items = score_synthetic_items(
        synth_items,
        model,
        device,
        config["image_size"],
        config["enable_reference_diagnostics"],
    )
    diagnostic_summary = summarize_reference_diagnostics(scored_items)

    # Apply quality gate
    filtered_items, gate_stats = apply_quality_gate(
        scored_items,
        conf_threshold=config["conf_threshold"],
        sharpness_floor=sharpness_floor,
    )

    # Write outputs
    write_json(
        output_root / "scored_manifest.json",
        write_manifest_payload(
            "scored_synthetic_pool",
            scored_items,
            {"gate_stats": gate_stats, "diagnostic_summary": diagnostic_summary},
        ),
    )
    write_json(
        output_root / "filtered_manifest.json",
        write_manifest_payload(
            "filtered_synthetic_pool",
            filtered_items,
            {"gate_stats": gate_stats, "diagnostic_summary": diagnostic_summary},
        ),
    )
    write_json(output_root / "gate_stats.json", gate_stats)
    write_json(output_root / "diagnostic_summary.json", diagnostic_summary)
    write_text(
        output_root / "report.md",
        render_report(config, gate_stats, diagnostic_summary, scored_items, filtered_items),
    )
    if args.config:
        dump_yaml_config(output_root / "resolved_config.yaml", config)
    print(f"Scored {len(scored_items)} items, {len(filtered_items)} passed quality gate ({gate_stats['pass_rate']:.1%})")
    print(f"Wrote outputs to: {output_root}")


if __name__ == "__main__":
    main()
