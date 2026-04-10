"""
Script 43: Build heuristic cell masks for contextual V2 dataset.

Outputs:
  data/processed_contextual_masks/{domain}/{class}/*.png
  data/processed_contextual_masks/{domain}/{class}/mask_metadata.jsonl
  results/contextual_masks/summary.json
  results/contextual_masks/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from boundary_aware_utils import (
    ROOT,
    CLASSES,
    DOMAINS,
    DOMAIN_SHORT,
    ensure_jsonl,
    extract_cell_mask,
    fallback_center_mask,
    mask_qc,
)


DATA_ROOT = ROOT / "data" / "processed_contextual_multidomain"
MASK_ROOT = ROOT / "data" / "processed_contextual_masks"
SUMMARY_ROOT = ROOT / "results" / "contextual_masks"
SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)


def process_class(domain: str, class_name: str, dry_run: bool) -> dict:
    src_dir = DATA_ROOT / domain / class_name
    out_dir = MASK_ROOT / domain / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in src_dir.glob("*.jpg"))
    rows = []
    n_fallback = 0
    for img_path in tqdm(images, desc=f"  {DOMAIN_SHORT[domain]}/{class_name}", leave=False):
        img = Image.open(img_path).convert("RGB")
        mask = extract_cell_mask(img)
        qc = mask_qc(mask)
        fallback = qc["area_ratio"] < 0.02 or qc["area_ratio"] > 0.45 or qc["center_overlap"] < 0.40
        if fallback:
            mask = fallback_center_mask(mask.shape)
            qc = mask_qc(mask)
            qc["fallback_required"] = True
            n_fallback += 1

        mask_name = img_path.with_suffix(".png").name
        if not dry_run:
            save_mask(mask, out_dir / mask_name)
        rows.append({
            "file_name": img_path.name,
            "mask_name": mask_name,
            "domain": domain,
            "class_name": class_name,
            **qc,
        })
    if not dry_run:
        ensure_jsonl(out_dir / "mask_metadata.jsonl", rows)
    return {
        "domain": domain,
        "class_name": class_name,
        "n_images": len(rows),
        "n_fallback": n_fallback,
        "mean_area_ratio": round(float(np.mean([r["area_ratio"] for r in rows])) if rows else 0.0, 4),
        "mean_center_overlap": round(float(np.mean([r["center_overlap"] for r in rows])) if rows else 0.0, 4),
    }


def write_summary(records: list[dict], label: str = "all") -> None:
    payload = {"records": records}
    (SUMMARY_ROOT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (SUMMARY_ROOT / f"summary_{label}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Contextual Mask Summary",
        "",
        f"- scope: `{label}`",
        "",
        "| Domain | Class | N | Fallbacks | Mean Area | Mean Center Overlap |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {DOMAIN_SHORT[row['domain']]} | {row['class_name']} | {row['n_images']} | "
            f"{row['n_fallback']} | {row['mean_area_ratio']:.4f} | {row['mean_center_overlap']:.4f} |"
        )
    content = "\n".join(lines) + "\n"
    (SUMMARY_ROOT / "summary.md").write_text(content, encoding="utf-8")
    (SUMMARY_ROOT / f"summary_{label}.md").write_text(content, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build heuristic masks for contextual dataset")
    parser.add_argument("--class_name", choices=CLASSES, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    class_names = [args.class_name] if args.class_name else list(CLASSES)
    if args.check:
        records = []
        for domain in DOMAINS:
            for class_name in class_names:
                mask_dir = MASK_ROOT / domain / class_name
                meta_path = mask_dir / "mask_metadata.jsonl"
                rows = []
                if meta_path.exists():
                    with open(meta_path, encoding="utf-8") as f:
                        rows = [json.loads(line) for line in f if line.strip()]
                records.append({
                    "domain": domain,
                    "class_name": class_name,
                    "n_images": len(rows),
                    "n_fallback": sum(1 for r in rows if r.get("fallback_required")),
                    "mean_area_ratio": round(float(np.mean([r["area_ratio"] for r in rows])) if rows else 0.0, 4),
                    "mean_center_overlap": round(float(np.mean([r["center_overlap"] for r in rows])) if rows else 0.0, 4),
                })
        write_summary(records, label=f"check_{args.class_name or 'all'}")
        print(f"[check] summary -> {SUMMARY_ROOT / 'summary.md'}")
        return

    records = []
    for domain in DOMAINS:
        print(f"\n[{DOMAIN_SHORT[domain]}]")
        for class_name in class_names:
            records.append(process_class(domain, class_name, args.dry_run))
    if not args.dry_run:
        write_summary(records, label=args.class_name or "all")
        print(f"[done] summary -> {SUMMARY_ROOT / 'summary.md'}")
    else:
        print("[dry_run] mask plan complete")


if __name__ == "__main__":
    main()
