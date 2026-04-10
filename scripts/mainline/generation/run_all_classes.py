#!/usr/bin/env python3
"""
Orchestrator: run generation stages 02 + 03 + merge for all 5 classes.

Produces per-class policy artifacts and synthetic pools, then merges
them into a single combined manifest ready for scoring (stage 04).

Usage:
    # Dry-run: show the plan
    python -m scripts.mainline.generation.run_all_classes --dry-run

    # Full run for heldout=raabin
    python -m scripts.mainline.generation.run_all_classes \
        --heldout-domain domain_b_raabin

    # Custom classes only
    python -m scripts.mainline.generation.run_all_classes \
        --heldout-domain domain_b_raabin \
        --classes monocyte eosinophil
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mainline.common.constants import CLASSES, DOMAINS, DOMAIN_SHORT, LORA_DIR_PATTERN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generation pipeline for all classes and merge manifests.",
    )
    parser.add_argument(
        "--heldout-domain",
        type=str,
        required=True,
        help="Held-out domain for leakage-safe generation.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Classes to generate. Default: all 5.",
    )
    parser.add_argument(
        "--config-policy",
        type=str,
        default="configs/mainline/generation/policy_v1.yaml",
    )
    parser.add_argument(
        "--config-generate",
        type=str,
        default="configs/mainline/generation/generate_v1_production.yaml",
    )
    parser.add_argument(
        "--policy-prefix",
        type=str,
        default="policy_v1",
        help="Prefix for policy_id naming.",
    )
    parser.add_argument(
        "--merge-output",
        type=str,
        default=None,
        help="Path for merged manifest. Default: auto-named under runs/.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], label: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return True

    print(f"  Running: {label}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if result.returncode == 0:
            # Print last line of stdout for status
            last_line = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""
            print(f"    -> OK: {last_line}")
            return True
        else:
            print(f"    -> FAILED (exit {result.returncode})")
            stderr_tail = result.stderr.strip().split("\n")[-3:] if result.stderr.strip() else []
            for line in stderr_tail:
                print(f"       {line}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    -> TIMEOUT")
        return False


def main() -> None:
    args = parse_args()
    heldout = args.heldout_domain
    if heldout not in DOMAINS:
        raise ValueError(f"Unknown heldout domain: {heldout}")

    classes = args.classes or list(CLASSES)
    heldout_short = DOMAIN_SHORT[heldout]

    print(f"\n{'='*70}")
    print(f"Generation Pipeline: all classes")
    print(f"{'='*70}")
    print(f"  Heldout domain: {heldout}")
    print(f"  Classes:        {classes}")
    print(f"  Policy config:  {args.config_policy}")
    print(f"  Generate config: {args.config_generate}")
    print(f"  Dry-run:        {args.dry_run}")
    print(f"{'='*70}\n")

    start_time = time.time()
    policy_dirs: dict[str, str] = {}
    manifest_paths: list[str] = []

    # Stage 02 + 03 per class
    for cls in classes:
        policy_id = f"{args.policy_prefix}_{cls}_{heldout_short}"
        lora_dir = LORA_DIR_PATTERN.replace("{class_name}", cls)
        lora_path = PROJECT_ROOT / lora_dir
        if not (lora_path / "pytorch_lora_weights.safetensors").exists():
            print(f"  SKIP {cls}: LoRA weights not found at {lora_dir}")
            continue

        policy_dir = f"results/mainline/generation/policies/{policy_id}"

        print(f"\n--- Stage 02: {cls} ---")
        cmd_02 = [
            sys.executable, "-m",
            "scripts.mainline.generation.02_train_generation_policy",
            "--config", args.config_policy,
            "--class-name", cls,
            "--lora-dir", lora_dir,
            "--policy-id", policy_id,
        ]
        ok = run_cmd(cmd_02, f"policy for {cls}", args.dry_run)
        if not ok:
            print(f"  Aborting {cls} generation due to policy failure.")
            continue

        policy_dirs[cls] = policy_dir

        print(f"\n--- Stage 03: {cls} ---")
        cmd_03 = [
            sys.executable, "-m",
            "scripts.mainline.generation.03_generate_synthetic_pool",
            "--config", args.config_generate,
            "--policy-dir", policy_dir,
            "--heldout-domain", heldout,
        ]
        if args.force:
            cmd_03.append("--force")

        ok = run_cmd(cmd_03, f"generate for {cls}", args.dry_run)
        if not ok:
            print(f"  Generation failed for {cls}, continuing with remaining classes.")
            continue

        # Find the synthetic manifest
        runs_dir = PROJECT_ROOT / "results/mainline/generation/runs"
        matching = sorted(runs_dir.glob(f"{policy_id}__*/synthetic_manifest.json"))
        if matching:
            manifest_paths.append(str(matching[-1]))
            print(f"    manifest: {matching[-1].relative_to(PROJECT_ROOT)}")

    # Merge
    if len(manifest_paths) >= 1:
        merge_output = args.merge_output or f"results/mainline/generation/runs/combined_heldout_{heldout_short}.json"
        print(f"\n--- Merge: {len(manifest_paths)} manifests ---")
        cmd_merge = [
            sys.executable, "-m",
            "scripts.mainline.generation.merge_synthetic_manifests",
            "--manifests", *manifest_paths,
            "--output", merge_output,
            "--heldout-domain", heldout,
        ]
        run_cmd(cmd_merge, "merge manifests", args.dry_run)
    else:
        print("\nNo manifests to merge.")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Complete: {len(policy_dirs)} policies, {len(manifest_paths)} manifests, {elapsed/60:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
