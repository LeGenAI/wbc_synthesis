#!/usr/bin/env python3
"""Merge per-class synthetic manifests into one combined manifest.

Typical usage after running 03_generate_synthetic_pool.py for each class:

    python -m scripts.mainline.generation.merge_synthetic_manifests \
        --manifests \
            results/mainline/generation/runs/run_basophil/synthetic_manifest.json \
            results/mainline/generation/runs/run_eosinophil/synthetic_manifest.json \
            results/mainline/generation/runs/run_lymphocyte/synthetic_manifest.json \
            results/mainline/generation/runs/run_monocyte/synthetic_manifest.json \
            results/mainline/generation/runs/run_neutrophil/synthetic_manifest.json \
        --output results/mainline/generation/runs/combined_synthetic_manifest.json \
        --heldout-domain domain_b_raabin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mainline.common.constants import CLASSES, DOMAINS
from scripts.mainline.common.manifests import (
    load_manifest_items,
    merge_synthetic_manifests,
    validate_no_leakage,
    write_manifest_payload,
)
from scripts.mainline.common.reporting import write_json, write_text
from scripts.mainline.common.runtime import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-class synthetic manifests into a combined manifest."
    )
    parser.add_argument(
        "--manifests",
        nargs="+",
        required=True,
        help="Paths to per-class synthetic_manifest.json files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the combined manifest.",
    )
    parser.add_argument(
        "--heldout-domain",
        type=str,
        default=None,
        help="If set, run leakage validation against this domain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_paths = [
        resolve_project_path(PROJECT_ROOT, p) for p in args.manifests
    ]
    for path in manifest_paths:
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

    merged_items = merge_synthetic_manifests(manifest_paths)

    # Validate class coverage
    classes_present = sorted(set(item["class_name"] for item in merged_items))
    missing_classes = sorted(set(CLASSES) - set(classes_present))
    if missing_classes:
        import warnings
        warnings.warn(
            f"Merged manifest is missing classes: {missing_classes}. "
            f"Present classes: {classes_present}",
            stacklevel=2,
        )

    # Leakage validation
    if args.heldout_domain:
        if args.heldout_domain not in DOMAINS:
            raise ValueError(f"Unknown heldout domain: {args.heldout_domain}")
        leakage_warnings = validate_no_leakage(merged_items, args.heldout_domain)
        for warning in leakage_warnings:
            print(f"WARNING: {warning}")
        if leakage_warnings:
            raise RuntimeError(
                "Leakage detected in merged manifest. "
                "Fix the per-class generation configs before proceeding."
            )

    output_path = resolve_project_path(PROJECT_ROOT, args.output)
    payload = write_manifest_payload(
        "merged_synthetic_pool",
        merged_items,
        {
            "source_manifests": [str(p) for p in manifest_paths],
            "classes_present": classes_present,
            "total_items": len(merged_items),
            "heldout_domain": args.heldout_domain,
        },
    )
    write_json(output_path, payload)
    print(f"Merged {len(merged_items)} items from {len(manifest_paths)} manifests")
    print(f"Classes: {classes_present}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
