#!/usr/bin/env python3
"""
Orchestrator: run the full benchmark experiment grid.

Discovers all *_production.yaml configs, runs each across all heldout
domains and seeds, skipping runs that already have a report.json.

Usage:
    # Dry-run: show the plan without executing
    python -m scripts.mainline.benchmark.run_experiment_grid --dry-run

    # Execute all production configs across all domains
    python -m scripts.mainline.benchmark.run_experiment_grid \
        --seeds 42 123 456

    # Only real_only configs
    python -m scripts.mainline.benchmark.run_experiment_grid \
        --pattern "real_only*_production.yaml" --seeds 42 123 456

    # Resume after interruption (skips existing runs automatically)
    python -m scripts.mainline.benchmark.run_experiment_grid \
        --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mainline.common.config import load_yaml_config
from scripts.mainline.common.constants import DOMAINS, DOMAIN_SHORT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full benchmark experiment grid.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/mainline/benchmark",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_production.yaml",
        help="Glob pattern for config files within config-dir.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
    )
    parser.add_argument(
        "--heldout-domains",
        type=str,
        nargs="+",
        default=None,
        help="Heldout domains to sweep. Default: all 4 domains.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running anything.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/mainline/benchmark",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
    )
    return parser.parse_args()


def run_already_exists(
    output_root: Path,
    backbone: str,
    heldout_domain: str,
    run_name: str,
) -> bool:
    report_path = output_root / backbone / f"heldout_{heldout_domain}" / run_name / "report.json"
    return report_path.exists()


def build_run_name_from_config(config: dict, seed: int, heldout_domain: str) -> str:
    """Approximate the run name that stage 05 would produce."""
    import re
    cfg = {**config, "seed": seed, "heldout_domain": heldout_domain}
    tf_tag = str(cfg.get("train_fraction", 1.0)).replace(".", "p")
    domain_short = DOMAIN_SHORT[heldout_domain]
    parts = [
        cfg.get("mode", "real_only"),
        cfg.get("backbone", "efficientnet_b0"),
        f"heldout_{domain_short}",
        f"tf{tf_tag}",
        f"seed{seed}",
    ]
    if cfg.get("mode") == "real_plus_synth" and cfg.get("synthetic_manifest"):
        stem = Path(str(cfg["synthetic_manifest"])).stem
        stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem)
        parts.append(f"synth_{stem}")
    cf = cfg.get("class_fractions", {})
    if cf:
        cf_tag = "_".join(f"{k[:4]}{str(v).replace('.', 'p')}" for k, v in sorted(cf.items()))
        parts.append(f"cf_{cf_tag}")
    if cfg.get("train_augment_mode", "standard") != "standard":
        parts.append(f"aug_{cfg['train_augment_mode']}")
    if cfg.get("eval_tta_mode", "none") != "none":
        parts.append(f"tta_{cfg['eval_tta_mode']}")
    return "__".join(parts)


def main() -> None:
    args = parse_args()
    config_dir = PROJECT_ROOT / args.config_dir
    output_root = PROJECT_ROOT / args.output_root
    heldout_domains = args.heldout_domains or list(DOMAINS)

    # Discover configs
    config_paths = sorted(glob.glob(str(config_dir / args.pattern)))
    if not config_paths:
        print(f"No configs found matching {config_dir / args.pattern}")
        return

    # Build execution plan
    plan: list[dict] = []
    skip_count = 0
    for config_path in config_paths:
        config = load_yaml_config(Path(config_path))
        config_name = Path(config_path).stem

        # Skip real_plus_synth configs that have no manifest set
        if config.get("mode") == "real_plus_synth" and not config.get("synthetic_manifest"):
            print(f"  SKIP {config_name}: synthetic_manifest is null (set it before running)")
            skip_count += 1
            continue

        for heldout_domain in heldout_domains:
            for seed in args.seeds:
                run_name = build_run_name_from_config(config, seed, heldout_domain)
                exists = run_already_exists(output_root, args.backbone, heldout_domain, run_name)
                if exists:
                    skip_count += 1
                    continue
                plan.append({
                    "config_path": config_path,
                    "config_name": config_name,
                    "heldout_domain": heldout_domain,
                    "seed": seed,
                    "run_name": run_name,
                })

    # Print plan
    total = len(plan)
    print(f"\n{'='*70}")
    print(f"Experiment Grid Plan")
    print(f"{'='*70}")
    print(f"  Configs found:    {len(config_paths)}")
    print(f"  Heldout domains:  {heldout_domains}")
    print(f"  Seeds:            {args.seeds}")
    print(f"  Runs to execute:  {total}")
    print(f"  Runs to skip:     {skip_count} (already exist or no manifest)")
    print(f"{'='*70}\n")

    if args.dry_run:
        for i, entry in enumerate(plan, 1):
            print(f"  [{i:3d}/{total}] {entry['config_name']}  heldout={DOMAIN_SHORT[entry['heldout_domain']]}  seed={entry['seed']}")
        print(f"\nDry-run complete. Use without --dry-run to execute.")
        return

    if total == 0:
        print("Nothing to run. All experiments already have results.")
        return

    # Execute
    start_time = time.time()
    completed = 0
    failed = 0

    for i, entry in enumerate(plan, 1):
        elapsed = time.time() - start_time
        if completed > 0:
            avg_time = elapsed / completed
            remaining = avg_time * (total - completed)
            eta_min = int(remaining / 60)
            print(f"\n[{i}/{total}] ETA: ~{eta_min}min remaining")
        else:
            print(f"\n[{i}/{total}]")

        print(f"  Config:  {entry['config_name']}")
        print(f"  Heldout: {entry['heldout_domain']}")
        print(f"  Seed:    {entry['seed']}")

        cmd = [
            sys.executable, "-m",
            "scripts.mainline.benchmark.05_train_lodo_utility_benchmark",
            "--config", entry["config_path"],
            "--heldout-domain", entry["heldout_domain"],
            "--seeds", str(entry["seed"]),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode == 0:
                completed += 1
                print(f"  -> OK ({completed}/{total} done)")
            else:
                failed += 1
                print(f"  -> FAILED (exit code {result.returncode})")
                print(f"     stderr: {result.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"  -> TIMEOUT (1h limit)")
        except KeyboardInterrupt:
            print(f"\n\nInterrupted after {completed} completed, {failed} failed.")
            print(f"Re-run to resume (completed runs will be skipped).")
            sys.exit(1)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Grid complete: {completed} OK, {failed} failed, {total_time/60:.1f} min total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
