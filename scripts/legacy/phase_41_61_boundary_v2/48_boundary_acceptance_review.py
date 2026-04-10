"""
Script 48: Review boundary-aware V2 generation and downstream acceptance.

Outputs:
  results/boundary_v2_acceptance/review.json
  results/boundary_v2_acceptance/review.md
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BASE_GEN_ROOT = ROOT / "results" / "diverse_generation"
V2_GEN_ROOT = ROOT / "results" / "boundary_v2_generation"
BASELINE_ROOT = ROOT / "results" / "lodo_baseline" / "efficientnet_b0"
S7_ROOT = ROOT / "results" / "selective_synth" / "efficientnet_b0"
V2_LODO_ROOT = ROOT / "results" / "boundary_v2_lodo" / "efficientnet_b0"
OUT_ROOT = ROOT / "results" / "boundary_v2_acceptance"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_SHORT = {
    "domain_a_pbc": "pbc",
    "domain_b_raabin": "raabin",
    "domain_c_mll23": "mll23",
    "domain_e_amc": "amc",
}
DOMAIN_LABELS = {
    "domain_a_pbc": "PBC (Spain)",
    "domain_b_raabin": "Raabin (Iran)",
    "domain_c_mll23": "MLL23 (Germany)",
    "domain_e_amc": "AMC (Korea)",
}
HARD_CLASSES = ["monocyte", "eosinophil"]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def safe_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_float(value) -> str:
    value = safe_float(value)
    return "-" if value is None else f"{value:.4f}"


def baseline_generation_metrics(class_name: str) -> dict:
    path = BASE_GEN_ROOT / class_name / "report.json"
    data = load_json(path)
    if not data:
        return {}
    margin_key = "margin_mean"
    near_key = "near_boundary_rate"
    aggregate = data.get("aggregate", {})
    return {
        "path": str(path),
        "n": aggregate.get("n", data.get("n", 0)),
        "cnn_accuracy": aggregate.get("cnn_accuracy"),
        "cell_ssim_mean": aggregate.get("cell_ssim_mean"),
        "background_ssim_mean": aggregate.get("background_ssim_mean"),
        "margin_mean": aggregate.get(margin_key),
        "near_boundary_rate": aggregate.get(near_key),
    }


def v2_generation_metrics(class_name: str) -> dict:
    path = V2_GEN_ROOT / class_name / "report.json"
    data = load_json(path)
    if not data:
        return {}
    aggregate = data.get("aggregate", {})
    return {
        "path": str(path),
        "n": data.get("n_generated", 0),
        "cnn_accuracy": aggregate.get("cnn_accuracy"),
        "cell_ssim_mean": aggregate.get("cell_ssim_mean"),
        "background_ssim_mean": aggregate.get("background_ssim_mean"),
        "margin_mean": aggregate.get("margin_mean"),
        "near_boundary_rate": aggregate.get("near_boundary_rate"),
        "region_gap": round(aggregate.get("cell_ssim_mean", 0.0) - aggregate.get("background_ssim_mean", 0.0), 4),
    }


def load_domain_result(root: Path, heldout: str, subset_id: str | None = None) -> dict | None:
    domain_dir = root / f"heldout_{DOMAIN_SHORT[heldout]}"
    path = domain_dir / "report.json" if subset_id is None else domain_dir / subset_id / "report.json"
    return load_json(path)


def find_v2_runs() -> list[dict]:
    summary = load_json(V2_LODO_ROOT / "summary.json")
    if not summary:
        return []
    return summary.get("runs", [])


def main():
    hard_class_review = {}
    for class_name in HARD_CLASSES:
        baseline = baseline_generation_metrics(class_name)
        v2 = v2_generation_metrics(class_name)
        base_near = safe_float(baseline.get("near_boundary_rate")) if baseline else None
        v2_near = safe_float(v2.get("near_boundary_rate")) if v2 else None
        v2_gap = safe_float(v2.get("region_gap"), 0.0) if v2 else None
        hard_class_review[class_name] = {
            "baseline": baseline,
            "v2": v2,
            "near_boundary_x": round(v2_near / max(1e-8, base_near), 2)
            if (base_near is not None and v2_near is not None) else None,
            "region_gap_target_met": bool(v2) and (v2_gap is not None) and v2_gap >= 0.08,
        }

    domain_review = {}
    v2_runs = find_v2_runs()
    best_v2_by_domain = {}
    for run in v2_runs:
        heldout = run["heldout_domain"]
        cur = best_v2_by_domain.get(heldout)
        if cur is None or run.get("macro_f1", -1) > cur.get("macro_f1", -1):
            best_v2_by_domain[heldout] = run

    for heldout in DOMAINS:
        baseline = load_domain_result(BASELINE_ROOT, heldout)
        s7 = load_domain_result(S7_ROOT, heldout, "S7")
        v2_best = best_v2_by_domain.get(heldout)
        if v2_best:
            v2_report = load_json(Path(v2_best["report_path"]))
        else:
            v2_report = None
        domain_review[heldout] = {
            "baseline_macro_f1": baseline["test"]["macro_f1"] if baseline else None,
            "s7_macro_f1": s7["test"]["macro_f1"] if s7 else None,
            "v2_subset": v2_best["subset_id"] if v2_best else None,
            "v2_macro_f1": v2_report["test"]["macro_f1"] if v2_report else None,
            "v2_vs_s7_delta": round(v2_report["test"]["macro_f1"] - s7["test"]["macro_f1"], 4) if (v2_report and s7) else None,
            "v2_vs_baseline_delta": round(v2_report["test"]["macro_f1"] - baseline["test"]["macro_f1"], 4) if (v2_report and baseline) else None,
        }

    acceptance = {
        "hard_classes": hard_class_review,
        "domains": domain_review,
        "targets": {
            "near_boundary_share_multiplier": ">= 5x on hard classes",
            "region_gap": "cell_ssim - background_ssim >= 0.08",
            "utility_guardrail": "macro-F1 no worse than refreshed S7 by more than 0.02 on hard domains",
        },
    }
    (OUT_ROOT / "review.json").write_text(json.dumps(acceptance, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Boundary V2 Acceptance Review",
        "",
        "## Hard-Class Generation",
        "",
        "| Class | Baseline near-boundary | V2 near-boundary | Multiplier | Cell SSIM | Background SSIM | Region Gap | Gap >= 0.08 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for class_name in HARD_CLASSES:
        row = hard_class_review[class_name]
        base = row["baseline"]
        v2 = row["v2"]
        if not base or not v2:
            lines.append(f"| {class_name} | - | - | - | - | - | - | - |")
            continue
        near_x = "-" if row["near_boundary_x"] is None else f"{row['near_boundary_x']:.2f}x"
        lines.append(
            f"| {class_name} | {fmt_float(base.get('near_boundary_rate'))} | {fmt_float(v2.get('near_boundary_rate'))} | "
            f"{near_x} | {fmt_float(v2.get('cell_ssim_mean'))} | "
            f"{fmt_float(v2.get('background_ssim_mean'))} | {fmt_float(v2.get('region_gap'))} | "
            f"{'yes' if row['region_gap_target_met'] else 'no'} |"
        )

    lines.extend([
        "",
        "## LODO Utility",
        "",
        "| Held-out | Baseline | Refreshed S7 | Best V2 | Subset | V2-Baseline | V2-S7 |",
        "|---|---:|---:|---:|---|---:|---:|",
    ])
    for heldout in DOMAINS:
        row = domain_review[heldout]
        def fmt(v):
            return "-" if v is None else f"{v:.4f}"
        lines.append(
            f"| {DOMAIN_LABELS[heldout]} | {fmt(row['baseline_macro_f1'])} | {fmt(row['s7_macro_f1'])} | "
            f"{fmt(row['v2_macro_f1'])} | {row['v2_subset'] or '-'} | {fmt(row['v2_vs_baseline_delta'])} | {fmt(row['v2_vs_s7_delta'])} |"
        )

    lines.extend([
        "",
        "## Acceptance Rules",
        "",
        "- Hard classes should increase near-boundary rate by at least `5x`.",
        "- Region separation target is `cell_ssim - background_ssim >= 0.08`.",
        "- Hard-domain downstream utility should not fall behind refreshed `S7` by more than `0.02` macro-F1.",
    ])
    (OUT_ROOT / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] review -> {OUT_ROOT / 'review.md'}")


if __name__ == "__main__":
    main()
