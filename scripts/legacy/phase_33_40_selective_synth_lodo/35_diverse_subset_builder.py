"""
Script 35: Build selective synthetic subset manifests from Script 33 outputs.

This script does not copy images. It reads per-class `report.json` files under
`results/diverse_generation/` and writes manifest files that define reusable
subset selections for downstream training.

Subset definitions:
  S1 all_synth
  S2 cnn_correct
  S3 high_conf_correct
  S4 ds025_only
  S5 ds035_only
  S6 ds045_only
  S7 hard_classes_only
  S8 weakest_domain_only
  S9 curriculum_all
  S10 monocyte_only
  S11 eosinophil_only
  S12 basophil_only
  S13 lymphocyte_only
  S14 neutrophil_only

Outputs:
  results/selective_synth/subsets/*.json
  results/selective_synth/summary.json
  results/selective_synth/summary.md

Examples:
  python3 scripts/legacy/phase_33_40_selective_synth_lodo/35_diverse_subset_builder.py
  python3 scripts/legacy/phase_33_40_selective_synth_lodo/35_diverse_subset_builder.py --high_conf_threshold 0.92
  python3 scripts/legacy/phase_33_40_selective_synth_lodo/35_diverse_subset_builder.py --weakest_domain domain_e_amc
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent.parent
REPORT_ROOT = ROOT / "results" / "diverse_generation"
OUT_DIR = ROOT / "results" / "selective_synth"
SUBSET_DIR = OUT_DIR / "subsets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUBSET_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_LABELS = {
    "domain_a_pbc": "PBC",
    "domain_b_raabin": "Raabin",
    "domain_c_mll23": "MLL23",
    "domain_e_amc": "AMC",
}
MULTI_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
HARD_CLASSES_DEFAULT = ["eosinophil", "monocyte"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build selective synthetic subset manifests from diverse generation reports"
    )
    parser.add_argument("--high_conf_threshold", type=float, default=0.90)
    parser.add_argument(
        "--weakest_domain",
        default="auto",
        help="auto or one of: domain_a_pbc, domain_b_raabin, domain_c_mll23, domain_e_amc",
    )
    parser.add_argument(
        "--hard_classes",
        default=",".join(HARD_CLASSES_DEFAULT),
        help="comma-separated classes for the hard-class subset",
    )
    return parser.parse_args()


def load_records() -> list[dict]:
    records = []
    for report_path in sorted(REPORT_ROOT.glob("*/report.json")):
        data = json.loads(report_path.read_text())
        class_name = data["class"]
        for item in data["per_image"]:
            denoise = float(item["denoise"])
            denoise_tag = f"ds{int(round(denoise * 100)):03d}"
            rel_path = item["file"]
            abs_path = str((ROOT / rel_path).resolve())
            stage = {0.25: 1, 0.35: 2, 0.45: 3}.get(round(denoise, 2), 0)
            records.append({
                "class_name": class_name,
                "file_rel": rel_path,
                "file_abs": abs_path,
                "domain": item["domain"],
                "domain_short": item["domain_short"],
                "denoise": denoise,
                "denoise_tag": denoise_tag,
                "template_idx": item["template_idx"],
                "template_name": item["template_name"],
                "inp_idx": item["inp_idx"],
                "seed": item["seed"],
                "sharpness": item["sharpness"],
                "ssim_vs_ref": item["ssim_vs_ref"],
                "cnn_pred": item["cnn_pred"],
                "cnn_conf": item["cnn_conf"],
                "cnn_correct": bool(item["cnn_correct"]),
                "curriculum_stage": stage,
            })
    return records


def infer_weakest_domain(records: list[dict]) -> str:
    domain_scores: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        domain_scores[rec["domain"]].append(1.0 if rec["cnn_correct"] else 0.0)
    summary = {
        dom: float(np.mean(vals))
        for dom, vals in domain_scores.items()
        if vals
    }
    return min(summary.items(), key=lambda kv: kv[1])[0]


def summarize_records(records: list[dict]) -> dict:
    class_counts = Counter(rec["class_name"] for rec in records)
    domain_counts = Counter(rec["domain"] for rec in records)
    denoise_counts = Counter(rec["denoise_tag"] for rec in records)
    template_counts = Counter(rec["template_name"] for rec in records)
    correct_rate = float(np.mean([1.0 if rec["cnn_correct"] else 0.0 for rec in records])) if records else 0.0
    conf_mean = float(np.mean([rec["cnn_conf"] for rec in records])) if records else 0.0
    return {
        "n_images": len(records),
        "correct_rate": round(correct_rate, 4),
        "cnn_conf_mean": round(conf_mean, 4),
        "by_class": dict(sorted(class_counts.items())),
        "by_domain": dict(sorted(domain_counts.items())),
        "by_denoise": dict(sorted(denoise_counts.items())),
        "by_template": dict(sorted(template_counts.items())),
    }


def write_manifest(
    subset_id: str,
    name: str,
    description: str,
    records: list[dict],
    extra: dict | None = None,
):
    payload = {
        "subset_id": subset_id,
        "name": name,
        "description": description,
        "summary": summarize_records(records),
        "items": records,
    }
    if extra:
        payload["extra"] = extra
    out_path = SUBSET_DIR / f"{subset_id}_{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload, out_path


def make_summary_md(summary: dict) -> str:
    lines = [
        "# Selective Synthetic Subset Summary",
        "",
        f"- Source: `{REPORT_ROOT}`",
        f"- Total source images: `{summary['total_source_images']}`",
        f"- Weakest domain rule: `{summary['weakest_domain']}` ({DOMAIN_LABELS[summary['weakest_domain']]})",
        f"- High-confidence threshold: `{summary['high_conf_threshold']}`",
        f"- Hard classes: `{', '.join(summary['hard_classes'])}`",
        "",
        "## Subsets",
        "",
        "| ID | Name | N | Correct Rate | Mean Conf |",
        "|---|---|---:|---:|---:|",
    ]
    for item in summary["subsets"]:
        s = item["summary"]
        lines.append(
            f"| {item['subset_id']} | {item['name']} | {s['n_images']} | "
            f"{s['correct_rate']:.4f} | {s['cnn_conf_mean']:.4f} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- `S3` requires both `cnn_correct == True` and `cnn_conf >= threshold`.",
        "- `S8` uses the weakest domain inferred from generated-image correctness unless overridden.",
        "- `S9` keeps all images and attaches `curriculum_stage` (`1=ds025`, `2=ds035`, `3=ds045`).",
        "- `S10-S14` are class-targeted manifests for follow-up rescue experiments.",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    hard_classes = sorted([x.strip().lower() for x in args.hard_classes.split(",") if x.strip()])
    for class_name in hard_classes:
        if class_name not in MULTI_CLASSES:
            raise ValueError(f"unknown hard class: {class_name}")

    records = load_records()
    if not records:
        raise RuntimeError("no records found under results/diverse_generation/*/report.json")

    weakest_domain = infer_weakest_domain(records) if args.weakest_domain == "auto" else args.weakest_domain
    if weakest_domain not in DOMAINS:
        raise ValueError(f"invalid weakest_domain: {weakest_domain}")

    subsets = []
    subset_specs = [
        ("S1", "all_synth", "All diverse synthetic images.", records, None),
        ("S2", "cnn_correct", "Synthetic images with correct CNN class prediction.",
         [r for r in records if r["cnn_correct"]], None),
        ("S3", "high_conf_correct",
         "Synthetic images with correct CNN class prediction and high confidence.",
         [r for r in records if r["cnn_correct"] and r["cnn_conf"] >= args.high_conf_threshold],
         {"high_conf_threshold": args.high_conf_threshold}),
        ("S4", "ds025_only", "Only denoise strength 0.25 images.",
         [r for r in records if r["denoise_tag"] == "ds025"], None),
        ("S5", "ds035_only", "Only denoise strength 0.35 images.",
         [r for r in records if r["denoise_tag"] == "ds035"], None),
        ("S6", "ds045_only", "Only denoise strength 0.45 images.",
         [r for r in records if r["denoise_tag"] == "ds045"], None),
        ("S7", "hard_classes_only",
         "Only hard classes selected for rescue experiments.",
         [r for r in records if r["class_name"] in hard_classes],
         {"hard_classes": hard_classes}),
        ("S8", "weakest_domain_only",
         "Only synthetic images aligned with the weakest target domain.",
         [r for r in records if r["domain"] == weakest_domain],
         {"weakest_domain": weakest_domain, "weakest_domain_label": DOMAIN_LABELS[weakest_domain]}),
        ("S9", "curriculum_all",
         "All synthetic images with curriculum stages by denoise level.",
         sorted(records, key=lambda r: (r["curriculum_stage"], r["class_name"], r["domain"], r["inp_idx"], r["seed"])),
         {"curriculum": {"1": "ds025", "2": "ds035", "3": "ds045"}}),
        ("S10", "monocyte_only",
         "Only monocyte synthetic images for class-targeted rescue experiments.",
         [r for r in records if r["class_name"] == "monocyte"],
         {"target_class": "monocyte"}),
        ("S11", "eosinophil_only",
         "Only eosinophil synthetic images for class-targeted rescue experiments.",
         [r for r in records if r["class_name"] == "eosinophil"],
         {"target_class": "eosinophil"}),
        ("S12", "basophil_only",
         "Only basophil synthetic images for class-targeted rescue experiments.",
         [r for r in records if r["class_name"] == "basophil"],
         {"target_class": "basophil"}),
        ("S13", "lymphocyte_only",
         "Only lymphocyte synthetic images for class-targeted rescue experiments.",
         [r for r in records if r["class_name"] == "lymphocyte"],
         {"target_class": "lymphocyte"}),
        ("S14", "neutrophil_only",
         "Only neutrophil synthetic images for class-targeted rescue experiments.",
         [r for r in records if r["class_name"] == "neutrophil"],
         {"target_class": "neutrophil"}),
    ]

    for subset_id, name, description, items, extra in subset_specs:
        manifest, out_path = write_manifest(subset_id, name, description, items, extra=extra)
        manifest["path"] = str(out_path)
        subsets.append(manifest)
        print(
            f"{subset_id} {name:18s} | n={manifest['summary']['n_images']:4d} | "
            f"correct={manifest['summary']['correct_rate']:.4f} | {out_path}"
        )

    summary = {
        "total_source_images": len(records),
        "weakest_domain": weakest_domain,
        "high_conf_threshold": args.high_conf_threshold,
        "hard_classes": hard_classes,
        "subsets": [
            {
                "subset_id": item["subset_id"],
                "name": item["name"],
                "path": item["path"],
                "summary": item["summary"],
            }
            for item in subsets
        ],
    }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    (OUT_DIR / "summary.md").write_text(make_summary_md(summary), encoding="utf-8")
    print(f"\nsummary saved: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
