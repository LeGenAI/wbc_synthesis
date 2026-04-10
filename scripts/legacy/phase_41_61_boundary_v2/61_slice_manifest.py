"""
Script 61: Slice a manifest into a smaller manifest.

Useful for quick ablations such as:
  - top-N lowest-margin boundary samples
  - first K items from a ranked manifest
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Slice a manifest into a smaller manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subset_id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top_n", type=int, required=True)
    parser.add_argument("--sort_by", default="target_margin")
    parser.add_argument("--descending", action="store_true")
    return parser.parse_args()


def summarize(records: list[dict]) -> dict:
    summary = {
        "n_images": len(records),
        "by_class": dict(sorted(Counter(r["class_name"] for r in records).items())),
        "by_domain": dict(sorted(Counter(r["domain"] for r in records).items())),
    }
    if records and "target_margin" in records[0]:
        summary["mean_margin"] = round(float(np.mean([r["target_margin"] for r in records])), 4)
    if records and "background_ssim" in records[0]:
        summary["mean_background_ssim"] = round(float(np.mean([r["background_ssim"] for r in records])), 4)
    return summary


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    payload = json.loads(manifest_path.read_text())
    items = payload["items"]
    items = sorted(items, key=lambda x: x.get(args.sort_by, 0.0), reverse=args.descending)
    sliced = items[:args.top_n]

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "subset_id": args.subset_id,
        "name": args.name,
        "description": args.description,
        "summary": summarize(sliced),
        "items": sliced,
        "source_manifest": str(manifest_path),
        "slice": {
            "top_n": args.top_n,
            "sort_by": args.sort_by,
            "descending": args.descending,
        },
    }
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "summary": out_payload["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
