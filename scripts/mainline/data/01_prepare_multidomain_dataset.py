#!/usr/bin/env python3
"""
Stage 01: canonical multidomain manifest builder.

Purpose:
- validate the existing 5-class / 4-domain processed dataset
- freeze deterministic held-out-domain manifests
- write canonical inventory and split summaries for later stages
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mainline.common.config import apply_overrides, dump_yaml_config, load_yaml_config
from scripts.mainline.common.constants import CLASSES, DOMAINS
from scripts.mainline.common.manifests import (
    collect_inventory,
    count_by_domain_class,
    write_manifest_payload,
)
from scripts.mainline.common.reporting import markdown_table, write_json, write_text
from scripts.mainline.common.runtime import ensure_dir, resolve_project_path
from scripts.mainline.common.split import stratified_train_val_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical multidomain manifests for the mainline pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mainline/data/base.yaml",
        help="Path to the mainline data YAML config.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional override for the manifest output root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for split_seed.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Optional override for val_ratio.",
    )
    return parser.parse_args()


def validate_config(config: dict) -> dict:
    domains = config.get("domains", DOMAINS)
    classes = config.get("classes", CLASSES)
    unknown_domains = sorted(set(domains) - set(DOMAINS))
    unknown_classes = sorted(set(classes) - set(CLASSES))
    if unknown_domains:
        raise ValueError(f"Unknown domains in config: {unknown_domains}")
    if unknown_classes:
        raise ValueError(f"Unknown classes in config: {unknown_classes}")
    val_ratio = float(config["val_ratio"])
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    return {
        **config,
        "domains": list(domains),
        "classes": list(classes),
        "val_ratio": val_ratio,
        "split_seed": int(config["split_seed"]),
    }


def assign_split(items: list[dict], split_name: str) -> list[dict]:
    updated = []
    for item in items:
        clone = dict(item)
        clone["split"] = split_name
        updated.append(clone)
    return updated


def render_summary_markdown(summary: dict) -> str:
    inventory_rows = []
    for domain, class_counts in summary["inventory_by_domain_class"].items():
        for class_name, count in class_counts.items():
            inventory_rows.append([domain, class_name, count])

    lines = [
        "# Mainline Dataset Summary",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Output root: `{summary['output_root']}`",
        f"- Split seed: `{summary['split_seed']}`",
        f"- Validation ratio: `{summary['val_ratio']}`",
        f"- Total inventory images: `{summary['inventory_count']}`",
        "",
        "## Inventory",
        "",
        markdown_table(["Domain", "Class", "Count"], inventory_rows),
        "",
        "## Held-out Splits",
        "",
    ]

    for heldout_domain, split_summary in summary["heldout_summaries"].items():
        rows = []
        for split_name, counts in split_summary["by_split_domain_class"].items():
            for domain, class_counts in counts.items():
                for class_name, count in class_counts.items():
                    rows.append([split_name, domain, class_name, count])
        lines.extend(
            [
                f"### {heldout_domain}",
                "",
                f"- train: `{split_summary['train_count']}`",
                f"- val: `{split_summary['val_count']}`",
                f"- test: `{split_summary['test_count']}`",
                "",
                markdown_table(["Split", "Domain", "Class", "Count"], rows),
                "",
            ]
        )

    return "\n".join(lines)


def nested_counts(items: list[dict]) -> dict[str, dict[str, dict[str, int]]]:
    payload: dict[str, dict[str, dict[str, int]]] = {}
    for item in items:
        payload.setdefault(item["split"], {}).setdefault(item["domain"], {})
        class_counts = payload[item["split"]][item["domain"]]
        class_counts[item["class_name"]] = class_counts.get(item["class_name"], 0) + 1
    return payload


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = load_yaml_config(config_path)
    config = apply_overrides(
        config,
        {
            "output_root": args.output_root,
            "split_seed": args.seed,
            "val_ratio": args.val_ratio,
        },
    )
    config = validate_config(config)

    dataset_root = resolve_project_path(PROJECT_ROOT, config["dataset_root"])
    output_root = ensure_dir(resolve_project_path(PROJECT_ROOT, config["output_root"]))

    inventory_items = collect_inventory(
        dataset_root=dataset_root,
        project_root=PROJECT_ROOT,
        domains=config["domains"],
        classes=config["classes"],
    )

    write_json(
        output_root / "inventory_manifest.json",
        write_manifest_payload(
            "inventory",
            inventory_items,
            {
                "dataset_root": str(dataset_root),
                "domains": config["domains"],
                "classes": config["classes"],
            },
        ),
    )

    heldout_summaries: dict[str, dict] = {}
    for heldout_domain in config["domains"]:
        heldout_dir = ensure_dir(output_root / f"heldout_{heldout_domain}")
        source_items = [item for item in inventory_items if item["domain"] != heldout_domain]
        test_items = [item for item in inventory_items if item["domain"] == heldout_domain]
        train_items, val_items = stratified_train_val_split(
            source_items,
            val_ratio=config["val_ratio"],
            seed=config["split_seed"],
        )
        train_items = assign_split(train_items, "train")
        val_items = assign_split(val_items, "val")
        test_items = assign_split(test_items, "test")

        write_json(
            heldout_dir / "train_manifest.json",
            write_manifest_payload("split", train_items, {"heldout_domain": heldout_domain, "split": "train"}),
        )
        write_json(
            heldout_dir / "val_manifest.json",
            write_manifest_payload("split", val_items, {"heldout_domain": heldout_domain, "split": "val"}),
        )
        write_json(
            heldout_dir / "test_manifest.json",
            write_manifest_payload("split", test_items, {"heldout_domain": heldout_domain, "split": "test"}),
        )

        split_summary = {
            "heldout_domain": heldout_domain,
            "train_count": len(train_items),
            "val_count": len(val_items),
            "test_count": len(test_items),
            "by_split_domain_class": nested_counts(train_items + val_items + test_items),
        }
        write_json(heldout_dir / "split_summary.json", split_summary)
        heldout_summaries[heldout_domain] = split_summary

    dataset_summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "split_seed": config["split_seed"],
        "val_ratio": config["val_ratio"],
        "inventory_count": len(inventory_items),
        "inventory_by_domain_class": count_by_domain_class(inventory_items),
        "heldout_summaries": heldout_summaries,
    }
    write_json(output_root / "dataset_summary.json", dataset_summary)
    write_text(output_root / "dataset_summary.md", render_summary_markdown(dataset_summary))
    dump_yaml_config(output_root / "resolved_config.yaml", config)
    print(f"Wrote canonical manifests to: {output_root}")


if __name__ == "__main__":
    main()
