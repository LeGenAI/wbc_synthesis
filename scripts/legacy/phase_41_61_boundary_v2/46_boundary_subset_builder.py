"""
Script 46: Build boundary-aware subset manifests from boundary V2 generation reports.

Outputs:
  results/boundary_selective_synth/subsets/*.json
  results/boundary_selective_synth/analysis/*.json
  results/boundary_selective_synth/summary.json
  results/boundary_selective_synth/summary.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent.parent
REPORT_ROOT = ROOT / "results" / "boundary_v2_generation"
OUT_ROOT = ROOT / "results" / "boundary_selective_synth"
SUBSET_DIR = OUT_ROOT / "subsets"
ANALYSIS_DIR = OUT_ROOT / "analysis"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
SUBSET_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build boundary-aware manifests from V2 generation reports")
    parser.add_argument(
        "--report_paths",
        nargs="+",
        help="Optional explicit list of report.json files to merge. Defaults to results/boundary_v2_generation/*/report.json.",
    )
    parser.add_argument("--margin_min", type=float, default=0.02)
    parser.add_argument("--margin_max", type=float, default=0.20)
    parser.add_argument("--cell_ssim_min", type=float, default=0.80)
    parser.add_argument("--background_ssim_max", type=float, default=0.85)
    parser.add_argument("--top_quantile", type=float, default=0.30)
    return parser.parse_args()


def resolve_report_paths(report_paths: list[str] | None) -> list[Path]:
    if not report_paths:
        return sorted(REPORT_ROOT.glob("*/report.json"))
    return [Path(p).expanduser().resolve() for p in report_paths]


def dedupe_key(row: dict) -> tuple:
    return (
        row["class_name"],
        row.get("input_file"),
        row["ref_domain"],
        row["domain"],
        row["background_strength"],
        row["refine_strength"],
        row.get("seed"),
    )


def load_records(report_paths: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for report_path in report_paths:
        data = json.loads(report_path.read_text())
        for item in data["per_image"]:
            row = {
                "class_name": item["class_name"],
                "file_rel": item["file"],
                "file_abs": str((ROOT / item["file"]).resolve()),
                "input_file": item.get("input_file"),
                "domain": item["domain"],
                "domain_short": item["domain_short"],
                "ref_domain": item["ref_domain"],
                "ref_domain_short": item["ref_domain_short"],
                "background_strength": item["background_strength"],
                "refine_strength": item["refine_strength"],
                "seed": item.get("seed"),
                "cnn_pred": item["cnn_pred"],
                "cnn_correct": bool(item["cnn_correct"]),
                "cnn_probs": item["cnn_probs"],
                "cnn_entropy": item["cnn_entropy"],
                "target_margin": item["target_margin"],
                "target_prob": item["target_prob"],
                "cell_ssim": item["cell_ssim"],
                "background_ssim": item["background_ssim"],
                "near_boundary": bool(item["near_boundary"]),
                "variation_score": item["variation_score"],
            }
            key = dedupe_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def summarize(records: list[dict]) -> dict:
    return {
        "n_images": len(records),
        "correct_rate": round(float(np.mean([1.0 if r["cnn_correct"] else 0.0 for r in records])) if records else 0.0, 4),
        "mean_margin": round(float(np.mean([r["target_margin"] for r in records])) if records else 0.0, 4),
        "mean_cell_ssim": round(float(np.mean([r["cell_ssim"] for r in records])) if records else 0.0, 4),
        "mean_background_ssim": round(float(np.mean([r["background_ssim"] for r in records])) if records else 0.0, 4),
        "by_class": dict(sorted(Counter(r["class_name"] for r in records).items())),
        "by_domain": dict(sorted(Counter(r["domain"] for r in records).items())),
    }


def write_manifest(subset_id: str, name: str, description: str, records: list[dict], extra: dict | None = None):
    payload = {
        "subset_id": subset_id,
        "name": name,
        "description": description,
        "summary": summarize(records),
        "items": records,
    }
    if extra:
        payload["extra"] = extra
    out_path = SUBSET_DIR / f"{subset_id}_{name}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload, out_path


def write_analysis(name: str, description: str, records: list[dict], extra: dict | None = None):
    payload = {
        "name": name,
        "description": description,
        "summary": summarize(records),
        "items": records,
    }
    if extra:
        payload["extra"] = extra
    out_path = ANALYSIS_DIR / f"{name}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload, out_path


def main():
    args = parse_args()
    report_paths = resolve_report_paths(args.report_paths)
    records = load_records(report_paths)
    if not records:
        raise RuntimeError("no V2 report.json files found under results/boundary_v2_generation")

    eligible = [
        r for r in records
        if r["cnn_correct"]
        and args.margin_min <= r["target_margin"] <= args.margin_max
        and r["cell_ssim"] >= args.cell_ssim_min
        and r["background_ssim"] <= args.background_ssim_max
    ]
    incorrect_near_miss = [
        r for r in records
        if (not r["cnn_correct"])
        and (-args.margin_max <= r["target_margin"] <= -args.margin_min)
        and r["cell_ssim"] >= args.cell_ssim_min
        and r["background_ssim"] <= args.background_ssim_max
    ]

    by_combo: dict[tuple[str, str], list[dict]] = {}
    for rec in eligible:
        by_combo.setdefault((rec["class_name"], rec["domain"]), []).append(rec)

    ranked = []
    for key, items in by_combo.items():
        items = sorted(items, key=lambda r: r["variation_score"], reverse=True)
        n_keep = max(1, int(round(len(items) * args.top_quantile)))
        ranked.extend(items[:n_keep])

    subsets = []
    subset_specs = [
        ("B1", "boundary_eligible", "All correct low-margin samples that satisfy preservation/diversity rules.", eligible),
        ("B2", "boundary_ranked", "Top-ranked low-margin samples per class/domain bucket.", ranked),
    ]
    for class_name in CLASSES:
        subset_specs.append((
            f"B_{class_name[:3].upper()}",
            f"{class_name}_boundary",
            f"Boundary-aware subset for class {class_name}.",
            [r for r in ranked if r["class_name"] == class_name],
        ))
    for domain in DOMAINS:
        subset_specs.append((
            f"B_{domain.split('_')[-1].upper()}",
            f"{domain}_boundary",
            f"Boundary-aware subset for target domain {domain}.",
            [r for r in ranked if r["domain"] == domain],
        ))

    for subset_id, name, description, subset_records in subset_specs:
        manifest, out_path = write_manifest(
            subset_id,
            name,
            description,
            subset_records,
            extra={
                "margin_min": args.margin_min,
                "margin_max": args.margin_max,
                "cell_ssim_min": args.cell_ssim_min,
                "background_ssim_max": args.background_ssim_max,
                "top_quantile": args.top_quantile,
            },
        )
        print(
            f"{subset_id:8s} {name:24s} | n={manifest['summary']['n_images']:4d} "
            f"| mean_margin={manifest['summary']['mean_margin']:.4f} | {out_path}"
        )
        subsets.append(manifest)

    summary = {
        "source_reports": [str(p) for p in report_paths],
        "thresholds": {
            "margin_min": args.margin_min,
            "margin_max": args.margin_max,
            "cell_ssim_min": args.cell_ssim_min,
            "background_ssim_max": args.background_ssim_max,
            "top_quantile": args.top_quantile,
        },
        "total_source_images": len(records),
        "n_eligible": len(eligible),
        "n_ranked": len(ranked),
        "n_incorrect_near_miss": len(incorrect_near_miss),
        "subsets": [
            {
                "subset_id": item["subset_id"],
                "name": item["name"],
                "summary": item["summary"],
            }
            for item in subsets
        ],
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    analysis_payload, analysis_path = write_analysis(
        "incorrect_near_miss",
        "Incorrect low-margin images that satisfy preservation/diversity rules; analysis only.",
        incorrect_near_miss,
        extra={
            "margin_window": [-args.margin_max, -args.margin_min],
            "cell_ssim_min": args.cell_ssim_min,
            "background_ssim_max": args.background_ssim_max,
        },
    )
    print(
        f"ANALYSIS incorrect_near_miss      | n={analysis_payload['summary']['n_images']:4d} "
        f"| mean_margin={analysis_payload['summary']['mean_margin']:.4f} | {analysis_path}"
    )

    lines = [
        "# Boundary-Aware Selective Synth Summary",
        "",
        f"- Source reports: `{len(report_paths)}`",
        f"- Total source images: `{len(records)}`",
        f"- Eligible low-margin images: `{len(eligible)}`",
        f"- Ranked images: `{len(ranked)}`",
        f"- Incorrect near-miss (analysis only): `{len(incorrect_near_miss)}`",
        "",
        "| ID | Name | N | Mean Margin | Cell SSIM | Background SSIM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in subsets:
        s = item["summary"]
        lines.append(
            f"| {item['subset_id']} | {item['name']} | {s['n_images']} | "
            f"{s['mean_margin']:.4f} | {s['mean_cell_ssim']:.4f} | {s['mean_background_ssim']:.4f} |"
        )
    lines.extend([
        "",
        "## Analysis Only",
        "",
        f"- incorrect_near_miss: `{analysis_payload['summary']['n_images']}` images",
        f"- file: `{analysis_path}`",
    ])
    (OUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
