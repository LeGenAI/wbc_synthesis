#!/usr/bin/env python3
"""
Stage 06: assemble paper tables, figures, and supplementary artifacts
from benchmark results.

Reads report.json files from results/mainline/benchmark/ and produces:
  - LaTeX tables for the experiment grid (main paper Table 1)
  - Per-class F1 breakdown tables
  - Hard-class rescue comparison
  - Low-data utility comparison
  - Dataset statistics table
  - Summary markdown for quick inspection

Usage:
    python -m scripts.mainline.reporting.06_make_submission_package \
        --benchmark-root results/mainline/benchmark \
        --data-summary results/mainline/data/dataset_summary.json \
        --output-root results/mainline/reporting
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mainline.common.constants import CLASSES, DOMAINS, DOMAIN_LABELS, DOMAIN_SHORT, HARD_CLASSES
from scripts.mainline.common.reporting import markdown_table, write_json, write_text
from scripts.mainline.common.runtime import ensure_dir, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble paper submission artifacts from benchmark results.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=str,
        default="results/mainline/benchmark",
    )
    parser.add_argument(
        "--data-summary",
        type=str,
        default="results/mainline/data/dataset_summary.json",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/mainline/reporting",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
    )
    parser.add_argument(
        "--fill-tex",
        type=str,
        default=None,
        help="Path to main.tex with %%PLACEHOLDER%% tokens to fill with real numbers.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def collect_reports(benchmark_root: Path, backbone: str) -> list[dict]:
    """Recursively collect all report.json files under benchmark_root/backbone/."""
    reports = []
    backbone_root = benchmark_root / backbone
    if not backbone_root.exists():
        return reports
    for report_path in sorted(backbone_root.rglob("report.json")):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        report["_report_path"] = str(report_path)
        reports.append(report)
    return reports


def group_by_heldout(reports: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in reports:
        groups[r["heldout_domain"]].append(r)
    return dict(groups)


def group_by_config_key(reports: list[dict]) -> dict[str, list[dict]]:
    """Group reports by a canonical config key (mode + augment + tta + fractions)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in reports:
        cfg = r.get("config", {})
        key_parts = [
            cfg.get("mode", "?"),
            f"tf{cfg.get('train_fraction', 1.0)}",
            f"aug_{cfg.get('train_augment_mode', 'standard')}",
            f"tta_{cfg.get('eval_tta_mode', 'none')}",
        ]
        cf = cfg.get("class_fractions", {})
        if cf:
            cf_tag = "_".join(f"{k[:4]}{v}" for k, v in sorted(cf.items()))
            key_parts.append(f"cf_{cf_tag}")
        key = "__".join(key_parts)
        groups[key].append(r)
    return dict(groups)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


# ---------------------------------------------------------------------------
# LaTeX generators
# ---------------------------------------------------------------------------

def latex_table_experiment_grid(
    reports: list[dict],
) -> str:
    """Generate LaTeX table: rows = config key, columns = heldout domains."""
    by_config = group_by_config_key(reports)
    domain_shorts = [DOMAIN_SHORT[d] for d in DOMAINS]

    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{LODO macro-F1 by held-out domain (mean $\\pm$ std over 3 seeds).}\n"
        "\\label{tab:main-results}\n"
        "\\begin{tabular}{l" + "c" * len(DOMAINS) + "c}\n"
        "\\toprule\n"
        "Configuration & " + " & ".join(domain_shorts) + " & Avg \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for config_key in sorted(by_config.keys()):
        config_reports = by_config[config_key]
        by_heldout = group_by_heldout(config_reports)
        cells = []
        domain_means = []
        for domain in DOMAINS:
            domain_reports = by_heldout.get(domain, [])
            f1s = [r["test"]["macro_f1"] for r in domain_reports]
            m, s = mean_std(f1s)
            domain_means.append(m)
            if len(f1s) >= 2:
                cells.append(f"{m:.3f}$\\pm${s:.3f}")
            elif len(f1s) == 1:
                cells.append(f"{m:.3f}")
            else:
                cells.append("--")
        avg_m = float(np.mean(domain_means)) if domain_means else 0.0
        cells.append(f"{avg_m:.3f}")

        label = config_key.replace("__", " / ").replace("_", "\\_")
        rows.append(f"  {label} & " + " & ".join(cells) + " \\\\")

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )

    return header + "\n".join(rows) + "\n" + footer


def latex_table_per_class(
    reports: list[dict],
    heldout_domain: str,
) -> str:
    """Generate per-class F1 table for a specific held-out domain."""
    by_config = group_by_config_key(reports)
    domain_reports_map: dict[str, list[dict]] = {}
    for key, reps in by_config.items():
        domain_reps = [r for r in reps if r["heldout_domain"] == heldout_domain]
        if domain_reps:
            domain_reports_map[key] = domain_reps

    if not domain_reports_map:
        return f"% No reports for heldout={heldout_domain}\n"

    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{Per-class test F1 (heldout={DOMAIN_SHORT[heldout_domain]}).}}\n"
        f"\\label{{tab:perclass-{DOMAIN_SHORT[heldout_domain]}}}\n"
        "\\begin{tabular}{l" + "c" * len(CLASSES) + "c}\n"
        "\\toprule\n"
        "Config & " + " & ".join(c[:4] for c in CLASSES) + " & Macro \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for config_key in sorted(domain_reports_map.keys()):
        reps = domain_reports_map[config_key]
        cells = []
        for cls in CLASSES:
            f1s = [r["test"]["per_class"][cls]["f1"] for r in reps]
            m, _ = mean_std(f1s)
            cells.append(f"{m:.3f}")
        macro_f1s = [r["test"]["macro_f1"] for r in reps]
        m, _ = mean_std(macro_f1s)
        cells.append(f"{m:.3f}")
        label = config_key.replace("__", " / ").replace("_", "\\_")
        rows.append(f"  {label} & " + " & ".join(cells) + " \\\\")

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return header + "\n".join(rows) + "\n" + footer


def latex_table_dataset_stats(data_summary: dict) -> str:
    """Generate dataset statistics table from data_summary.json."""
    rows = []
    for domain in DOMAINS:
        class_counts = data_summary["inventory_by_domain_class"].get(domain, {})
        total = sum(class_counts.values())
        cells = [DOMAIN_LABELS.get(domain, domain)]
        for cls in CLASSES:
            cells.append(str(class_counts.get(cls, 0)))
        cells.append(str(total))
        rows.append("  " + " & ".join(cells) + " \\\\")

    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Dataset statistics: images per domain and class.}\n"
        "\\label{tab:dataset-stats}\n"
        "\\begin{tabular}{l" + "c" * len(CLASSES) + "c}\n"
        "\\toprule\n"
        "Domain & " + " & ".join(c[:4] for c in CLASSES) + " & Total \\\\\n"
        "\\midrule\n"
    )
    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return header + "\n".join(rows) + "\n" + footer


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def build_markdown_summary(reports: list[dict], data_summary: dict | None) -> str:
    lines = ["# Submission Package Summary", ""]

    if data_summary:
        lines.append(f"- Total images: `{data_summary['inventory_count']}`")
        lines.append(f"- Domains: `{data_summary.get('domains', DOMAINS)}`")
        lines.append(f"- Classes: `{CLASSES}`")
        lines.append("")

    lines.append(f"- Total benchmark reports collected: `{len(reports)}`")
    by_heldout = group_by_heldout(reports)
    for domain in DOMAINS:
        domain_reps = by_heldout.get(domain, [])
        lines.append(f"- {DOMAIN_SHORT[domain]}: `{len(domain_reps)}` runs")
    lines.append("")

    # Best per domain
    lines.append("## Best Macro-F1 by Held-out Domain")
    lines.append("")
    best_rows = []
    for domain in DOMAINS:
        domain_reps = by_heldout.get(domain, [])
        if not domain_reps:
            best_rows.append([DOMAIN_SHORT[domain], "--", "--", "--"])
            continue
        best = max(domain_reps, key=lambda r: r["test"]["macro_f1"])
        best_rows.append([
            DOMAIN_SHORT[domain],
            best["run_name"],
            f"{best['test']['macro_f1']:.4f}",
            f"{best['test']['accuracy']:.4f}",
        ])
    lines.append(markdown_table(["Heldout", "Best Run", "Macro-F1", "Accuracy"], best_rows))
    lines.append("")

    # Hard-class summary
    lines.append("## Hard-class F1 Summary")
    lines.append("")
    for domain in DOMAINS:
        domain_reps = by_heldout.get(domain, [])
        if not domain_reps:
            continue
        best = max(domain_reps, key=lambda r: r["test"]["macro_f1"])
        hc_rows = []
        for cls in HARD_CLASSES:
            pc = best["test"]["per_class"].get(cls, {})
            hc_rows.append([cls, f"{pc.get('f1', 0):.4f}", f"{pc.get('recall', 0):.4f}", pc.get("support", 0)])
        lines.append(f"### {DOMAIN_SHORT[domain]} (best: {best['run_name']})")
        lines.append("")
        lines.append(markdown_table(["Class", "F1", "Recall", "Support"], hc_rows))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confusion matrix collector
# ---------------------------------------------------------------------------

def collect_confusion_matrices(reports: list[dict], output_dir: Path) -> list[str]:
    """Copy confusion matrix PNGs into the output directory."""
    collected = []
    for r in reports:
        cm_path = r.get("artifacts", {}).get("confusion_matrix_png")
        if cm_path and Path(cm_path).exists():
            dest_name = f"cm__{r['run_name']}.png"
            dest = output_dir / dest_name
            shutil.copy2(cm_path, dest)
            collected.append(dest_name)
    return collected


# ---------------------------------------------------------------------------
# Delta comparison table
# ---------------------------------------------------------------------------

def latex_table_delta(reports: list[dict]) -> str:
    """Generate a delta table comparing each config against real_only baseline."""
    by_heldout = group_by_heldout(reports)
    by_config = group_by_config_key(reports)

    # Find baseline key (real_only, tf=1.0, standard, no TTA, no class_fractions)
    baseline_key = None
    for key in by_config:
        if ("real_only" in key and "tf1.0" in key and "aug_standard" in key
                and "tta_none" in key and "cf_" not in key):
            baseline_key = key
            break

    if baseline_key is None:
        return "% No baseline found for delta table\n"

    # Baseline F1 per domain
    baseline_reps = by_config[baseline_key]
    baseline_f1: dict[str, float] = {}
    for domain in DOMAINS:
        domain_reps = [r for r in baseline_reps if r["heldout_domain"] == domain]
        if domain_reps:
            baseline_f1[domain] = float(np.mean([r["test"]["macro_f1"] for r in domain_reps]))

    domain_shorts = [DOMAIN_SHORT[d] for d in DOMAINS]
    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Macro-F1 delta vs.\\ real-only baseline (positive = improvement).}\n"
        "\\label{tab:delta-results}\n"
        "\\begin{tabular}{l" + "c" * len(DOMAINS) + "c}\n"
        "\\toprule\n"
        "Configuration & " + " & ".join(f"$\\Delta${s}" for s in domain_shorts) + " & $\\Delta$Avg \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for config_key in sorted(by_config.keys()):
        if config_key == baseline_key:
            continue
        config_reports = by_config[config_key]
        config_by_heldout = group_by_heldout(config_reports)
        cells = []
        deltas = []
        for domain in DOMAINS:
            domain_reps = config_by_heldout.get(domain, [])
            base = baseline_f1.get(domain)
            if domain_reps and base is not None:
                m = float(np.mean([r["test"]["macro_f1"] for r in domain_reps]))
                delta = m - base
                deltas.append(delta)
                sign = "+" if delta >= 0 else ""
                cells.append(f"{sign}{delta:.3f}")
            else:
                cells.append("--")
        avg_delta = float(np.mean(deltas)) if deltas else 0.0
        sign = "+" if avg_delta >= 0 else ""
        cells.append(f"{sign}{avg_delta:.3f}")

        label = config_key.replace("__", " / ").replace("_", "\\_")
        rows.append(f"  {label} & " + " & ".join(cells) + " \\\\")

    # Baseline row
    base_cells = []
    for domain in DOMAINS:
        val = baseline_f1.get(domain)
        base_cells.append(f"{val:.3f}" if val is not None else "--")
    avg_base = float(np.mean(list(baseline_f1.values()))) if baseline_f1 else 0.0
    base_cells.append(f"{avg_base:.3f}")
    baseline_row = "  \\textit{baseline (real\\_only)} & " + " & ".join(base_cells) + " \\\\"

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )

    return header + baseline_row + "\n\\midrule\n" + "\n".join(rows) + "\n" + footer


# ---------------------------------------------------------------------------
# Placeholder fill for main.tex
# ---------------------------------------------------------------------------

def build_placeholder_map(reports: list[dict]) -> dict[str, str]:
    """Build a map of %%PLACEHOLDER%% -> value from collected reports.

    Supported placeholders (all case-insensitive):
      %%BASELINE_MACRO_F1_AVG%%        mean across domains for real_only tf=1.0
      %%BASELINE_MACRO_F1_<DOMAIN>%%   per-domain real_only baseline
      %%BEST_MACRO_F1_<DOMAIN>%%       best macro-F1 for each heldout domain
      %%BEST_RUN_<DOMAIN>%%            run name of the best for each domain
      %%HARDCLASS_<CLASS>_F1_<DOMAIN>%%  per hard-class F1 at best run
      %%N_TOTAL_IMAGES%%               total images in generation pool
    """
    placeholders: dict[str, str] = {}
    by_heldout = group_by_heldout(reports)

    # Find baseline (real_only, tf=1.0, standard augment, no TTA)
    baseline_f1s = {}
    for domain in DOMAINS:
        domain_reps = by_heldout.get(domain, [])
        baseline_reps = [
            r for r in domain_reps
            if r.get("config", {}).get("mode") == "real_only"
            and abs(float(r.get("config", {}).get("train_fraction", 0)) - 1.0) < 0.01
            and r.get("config", {}).get("train_augment_mode", "standard") == "standard"
            and r.get("config", {}).get("eval_tta_mode", "none") == "none"
            and not r.get("config", {}).get("class_fractions")
        ]
        if baseline_reps:
            f1s = [r["test"]["macro_f1"] for r in baseline_reps]
            m = float(np.mean(f1s))
            baseline_f1s[domain] = m
            short = DOMAIN_SHORT[domain].upper()
            placeholders[f"%%BASELINE_MACRO_F1_{short}%%"] = f"{m:.4f}"

    if baseline_f1s:
        avg = float(np.mean(list(baseline_f1s.values())))
        placeholders["%%BASELINE_MACRO_F1_AVG%%"] = f"{avg:.4f}"

    # Best per domain
    for domain in DOMAINS:
        domain_reps = by_heldout.get(domain, [])
        if not domain_reps:
            continue
        best = max(domain_reps, key=lambda r: r["test"]["macro_f1"])
        short = DOMAIN_SHORT[domain].upper()
        placeholders[f"%%BEST_MACRO_F1_{short}%%"] = f"{best['test']['macro_f1']:.4f}"
        placeholders[f"%%BEST_ACC_{short}%%"] = f"{best['test']['accuracy']:.4f}"
        placeholders[f"%%BEST_RUN_{short}%%"] = best["run_name"]

        # Hard-class F1 at best run
        for cls in HARD_CLASSES:
            pc = best["test"]["per_class"].get(cls, {})
            cls_upper = cls.upper()
            placeholders[f"%%HARDCLASS_{cls_upper}_F1_{short}%%"] = f"{pc.get('f1', 0):.4f}"
            placeholders[f"%%HARDCLASS_{cls_upper}_RECALL_{short}%%"] = f"{pc.get('recall', 0):.4f}"

    return placeholders


def fill_tex_placeholders(tex_path: Path, placeholders: dict[str, str], output_path: Path) -> int:
    """Replace %%PLACEHOLDER%% tokens in a .tex file. Returns count of replacements."""
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    count = 0
    for token, value in placeholders.items():
        if token in content:
            content = content.replace(token, value)
            count += 1

    # Find unfilled placeholders
    import re
    unfilled = re.findall(r"%%[A-Z_]+%%", content)
    if unfilled:
        print(f"  WARNING: {len(unfilled)} unfilled placeholders remain: {unfilled[:5]}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    benchmark_root = resolve_project_path(PROJECT_ROOT, args.benchmark_root)
    output_root = ensure_dir(resolve_project_path(PROJECT_ROOT, args.output_root))

    # Load data summary
    data_summary = None
    data_summary_path = resolve_project_path(PROJECT_ROOT, args.data_summary)
    if data_summary_path.exists():
        with open(data_summary_path, "r", encoding="utf-8") as f:
            data_summary = json.load(f)

    # Collect reports
    reports = collect_reports(benchmark_root, args.backbone)
    if not reports:
        print(f"No reports found under {benchmark_root / args.backbone}")
        return

    print(f"Collected {len(reports)} reports")

    # LaTeX tables
    tables_dir = ensure_dir(output_root / "tables")

    write_text(
        tables_dir / "tab_main_results.tex",
        latex_table_experiment_grid(reports),
    )

    write_text(
        tables_dir / "tab_delta_results.tex",
        latex_table_delta(reports),
    )

    for domain in DOMAINS:
        write_text(
            tables_dir / f"tab_perclass_{DOMAIN_SHORT[domain]}.tex",
            latex_table_per_class(reports, domain),
        )

    if data_summary:
        write_text(
            tables_dir / "tab_dataset_stats.tex",
            latex_table_dataset_stats(data_summary),
        )

    # Confusion matrices
    figures_dir = ensure_dir(output_root / "figures")
    cm_files = collect_confusion_matrices(reports, figures_dir)
    print(f"Collected {len(cm_files)} confusion matrices")

    # Markdown summary
    summary_md = build_markdown_summary(reports, data_summary)
    write_text(output_root / "submission_summary.md", summary_md)

    # Fill tex placeholders
    placeholders = build_placeholder_map(reports)
    write_json(output_root / "placeholders.json", placeholders)
    print(f"Built {len(placeholders)} placeholder values")

    if args.fill_tex:
        tex_path = resolve_project_path(PROJECT_ROOT, args.fill_tex)
        if tex_path.exists():
            filled_path = output_root / "main_filled.tex"
            n_filled = fill_tex_placeholders(tex_path, placeholders, filled_path)
            print(f"Filled {n_filled} placeholders -> {filled_path}")
        else:
            print(f"WARNING: --fill-tex path not found: {tex_path}")

    # Metadata
    write_json(output_root / "package_metadata.json", {
        "n_reports": len(reports),
        "backbone": args.backbone,
        "domains": DOMAINS,
        "classes": CLASSES,
        "hard_classes": HARD_CLASSES,
        "tables": sorted(str(p.name) for p in tables_dir.iterdir()),
        "figures": sorted(cm_files),
        "n_placeholders": len(placeholders),
    })

    print(f"Wrote submission package to: {output_root}")


if __name__ == "__main__":
    main()
