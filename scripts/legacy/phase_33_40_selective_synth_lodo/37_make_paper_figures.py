"""
Script 37: Generate paper-ready quantitative figures from LODO result JSON files.

Outputs:
  results/paper_figures/fig02_lodo_baseline_macro_f1.png
  results/paper_figures/fig03_subset_ablation_macro_f1.png
  results/paper_figures/fig04_best_setting_by_domain.png
  results/paper_figures/fig05_raabin_per_class_rescue.png
  results/paper_figures/fig06_amc_pbc_best_per_class.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent.parent
BASE_DIR = ROOT / "results" / "lodo_baseline" / "efficientnet_b0"
SEL_DIR = ROOT / "results" / "selective_synth" / "efficientnet_b0"
OUT_DIR = ROOT / "results" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["amc", "pbc", "raabin", "mll23"]
DOMAIN_LABELS = {
    "amc": "AMC",
    "pbc": "PBC",
    "raabin": "Raabin",
    "mll23": "MLL23",
}
SUBSET_LABELS = {
    "baseline": "Baseline",
    "S2": "S2",
    "S3": "S3",
    "S7": "S7",
    "S10": "S10",
}
CLASS_ORDER = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results():
    baseline = {}
    selective = {}

    for domain in DOMAINS:
        baseline_path = BASE_DIR / f"heldout_{domain}" / "report.json"
        if baseline_path.exists():
            baseline[domain] = load_json(baseline_path)

        selective[domain] = {}
        for subset in ["S2", "S3", "S7", "S10"]:
            subset_path = SEL_DIR / f"heldout_{domain}" / subset / "report.json"
            if subset_path.exists():
                selective[domain][subset] = load_json(subset_path)

    return baseline, selective


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 240,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })


def save(fig: plt.Figure, name: str):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def fig02_baseline_macro_f1(baseline: dict):
    labels = [DOMAIN_LABELS[d] for d in DOMAINS]
    vals = [baseline[d]["test"]["macro_f1"] for d in DOMAINS]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(labels, vals, color=["#D55E00", "#E69F00", "#009E73", "#0072B2"])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Macro-F1")
    ax.set_title("LODO Baseline Performance by Held-out Domain")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center")
    save(fig, "fig02_lodo_baseline_macro_f1.png")


def fig03_subset_ablation(baseline: dict, selective: dict):
    series = ["baseline", "S2", "S3", "S7", "S10"]
    colors = {
        "baseline": "#4C566A",
        "S2": "#5E81AC",
        "S3": "#88C0D0",
        "S7": "#A3BE8C",
        "S10": "#EBCB8B",
    }
    x = np.arange(len(DOMAINS))
    width = 0.16

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i, series_name in enumerate(series):
        vals = []
        for dom in DOMAINS:
            if series_name == "baseline":
                vals.append(baseline[dom]["test"]["macro_f1"])
            else:
                vals.append(selective.get(dom, {}).get(series_name, {}).get("test", {}).get("macro_f1", np.nan))
        ax.bar(x + (i - 2) * width, vals, width=width, label=SUBSET_LABELS[series_name], color=colors[series_name])

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Subset Ablation Across Held-out Domains")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    save(fig, "fig03_subset_ablation_macro_f1.png")


def best_setting_for_domain(domain: str, baseline: dict, selective: dict):
    best_name = "baseline"
    best_data = baseline[domain]
    best_f1 = baseline[domain]["test"]["macro_f1"]
    for subset, data in selective.get(domain, {}).items():
        score = data["test"]["macro_f1"]
        if score > best_f1:
            best_name = subset
            best_data = data
            best_f1 = score
    return best_name, best_data


def fig04_best_setting_by_domain(baseline: dict, selective: dict):
    labels = [DOMAIN_LABELS[d] for d in DOMAINS]
    base_vals = [baseline[d]["test"]["macro_f1"] for d in DOMAINS]
    best_vals = []
    best_labels = []
    for d in DOMAINS:
        best_name, best_data = best_setting_for_domain(d, baseline, selective)
        best_vals.append(best_data["test"]["macro_f1"])
        best_labels.append(best_name)

    x = np.arange(len(DOMAINS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bars1 = ax.bar(x - width / 2, base_vals, width=width, label="Baseline", color="#6B7280")
    bars2 = ax.bar(x + width / 2, best_vals, width=width, label="Domain-best", color="#2A9D8F")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Best Domain-specific Policy vs Baseline")
    ax.legend()

    for i, (bar, tag) in enumerate(zip(bars2, best_labels)):
        ax.text(bar.get_x() + bar.get_width() / 2, best_vals[i] + 0.015, tag, ha="center", fontsize=9)

    save(fig, "fig04_best_setting_by_domain.png")


def per_class_f1(result: dict) -> list[float]:
    return [result["test"]["per_class"][cls]["f1"] for cls in CLASS_ORDER]


def fig05_raabin_per_class(baseline: dict, selective: dict):
    domain = "raabin"
    series = {
        "Baseline": baseline[domain],
        "S2": selective[domain]["S2"],
        "S7": selective[domain]["S7"],
        "S10": selective[domain]["S10"],
    }
    x = np.arange(len(CLASS_ORDER))
    width = 0.19
    colors = ["#4C566A", "#5E81AC", "#A3BE8C", "#EBCB8B"]

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    for idx, (name, data) in enumerate(series.items()):
        ax.bar(x + (idx - 1.5) * width, per_class_f1(data), width=width, label=name, color=colors[idx])

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASS_ORDER])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Raabin Per-class Rescue: Baseline vs S2 / S7 / S10")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    save(fig, "fig05_raabin_per_class_rescue.png")


def fig06_amc_pbc_best_per_class(baseline: dict, selective: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    panels = [
        ("amc", "S7", "AMC: Baseline vs S7"),
        ("pbc", "S3", "PBC: Baseline vs S3"),
    ]
    colors = ["#6B7280", "#2A9D8F"]

    for ax, (domain, subset, title) in zip(axes, panels):
        x = np.arange(len(CLASS_ORDER))
        width = 0.34
        base_vals = per_class_f1(baseline[domain])
        sel_vals = per_class_f1(selective[domain][subset])
        ax.bar(x - width / 2, base_vals, width=width, label="Baseline", color=colors[0])
        ax.bar(x + width / 2, sel_vals, width=width, label=subset, color=colors[1])
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in CLASS_ORDER], rotation=15)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("F1")

    axes[0].legend()
    save(fig, "fig06_amc_pbc_best_per_class.png")


def main():
    setup_style()
    baseline, selective = load_results()
    fig02_baseline_macro_f1(baseline)
    fig03_subset_ablation(baseline, selective)
    fig04_best_setting_by_domain(baseline, selective)
    fig05_raabin_per_class(baseline, selective)
    fig06_amc_pbc_best_per_class(baseline, selective)
    print(f"saved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
