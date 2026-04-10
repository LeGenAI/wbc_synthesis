"""
Script 60: Merge selective-synth manifests into a hybrid manifest.

Primary use:
  - combine a strong baseline subset such as S7 with a small boundary-aware
    subset such as B2
  - preserve original metadata while deduplicating by absolute file path

Outputs:
  user-chosen manifest path, typically under results/selective_synth/subsets/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Merge manifest files into one hybrid manifest")
    parser.add_argument("--manifest", action="append", required=True, help="Input manifest path; may be repeated")
    parser.add_argument("--subset_id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--out", required=True, help="Output manifest path")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize(records: list[dict]) -> dict:
    class_counts = Counter(rec["class_name"] for rec in records)
    domain_counts = Counter(rec["domain"] for rec in records)
    summary = {
        "n_images": len(records),
        "by_class": dict(sorted(class_counts.items())),
        "by_domain": dict(sorted(domain_counts.items())),
    }
    if records and "cnn_correct" in records[0]:
        summary["correct_rate"] = round(float(np.mean([1.0 if rec.get("cnn_correct") else 0.0 for rec in records])), 4)
    if records and "cnn_conf" in records[0]:
        confs = [rec["cnn_conf"] for rec in records if "cnn_conf" in rec]
        if confs:
            summary["cnn_conf_mean"] = round(float(np.mean(confs)), 4)
    if records and "target_margin" in records[0]:
        margins = [rec["target_margin"] for rec in records if "target_margin" in rec]
        if margins:
            summary["mean_margin"] = round(float(np.mean(margins)), 4)
    return summary


def main():
    args = parse_args()
    manifest_paths = [Path(p).expanduser().resolve() for p in args.manifest]
    merged = []
    seen = set()
    sources = []

    for path in manifest_paths:
        payload = load_manifest(path)
        items = payload["items"]
        sources.append({
            "path": str(path),
            "subset_id": payload.get("subset_id"),
            "name": payload.get("name"),
            "n_items": len(items),
        })
        for item in items:
            key = item.get("file_abs") or item.get("file_rel")
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subset_id": args.subset_id,
        "name": args.name,
        "description": args.description,
        "summary": summarize(merged),
        "items": merged,
        "sources": sources,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "out": str(out_path),
        "n_items": len(merged),
        "summary": payload["summary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
