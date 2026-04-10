"""
Script 42: Context-preserving multi-domain preprocessing for boundary-aware V2.

Outputs:
  data/processed_contextual_multidomain/{domain}/{class}/*.jpg
  data/processed_contextual_multidomain/{domain}/{class}/metadata.jsonl
  results/contextual_preprocess/summary.json
  results/contextual_preprocess/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from boundary_aware_utils import (
    ROOT,
    CLASSES,
    DOMAIN_SHORT,
    DOMAINS,
    build_contextual_prompt,
    bounded_center_jitter_crop,
    ensure_jsonl,
    resize_with_padding,
)


RAW_MULTI = ROOT / "data" / "raw" / "multi_domain"
RAW_PBC = ROOT / "data" / "raw" / "PBC_dataset_normal_DIB_224" / "PBC_dataset_normal_DIB_224"
OUT_ROOT = ROOT / "data" / "processed_contextual_multidomain"
SUMMARY_ROOT = ROOT / "results" / "contextual_preprocess"
SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

RAABIN_MAP = {c: c for c in CLASSES}
MLL23_MAP = {c: c for c in CLASSES}
AMC_MAP = {
    "basophil": "basophil",
    "eosinophil": "eosinophil",
    "lymphocyte": "lymphocyte",
    "monocyte": "monocyte",
    "seg_neutrophil": "neutrophil",
}
PBC_MAP = {c: c for c in CLASSES}


def save_contextual_image(src: Path, dst: Path, class_name: str, domain: str, policy_key: str) -> dict:
    img = Image.open(src).convert("RGB")
    if policy_key == "raabin":
        out, meta = bounded_center_jitter_crop(img, crop_size=448, output_size=384, key=str(src))
    else:
        out, meta = resize_with_padding(img, canvas=384)
        meta["policy"] = f"{policy_key}_{meta['policy']}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, "JPEG", quality=95)
    meta.update({
        "file_name": dst.name,
        "source_path": str(src.resolve()),
        "domain": domain,
        "class_name": class_name,
        "prompt": build_contextual_prompt(class_name, domain),
    })
    return meta


def process_class_images(domain: str, class_name: str, src_paths: list[Path], policy_key: str, dry_run: bool) -> dict:
    out_dir = OUT_ROOT / domain / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, src in enumerate(tqdm(src_paths, desc=f"  {DOMAIN_SHORT[domain]}/{class_name}", leave=False)):
        dst = out_dir / f"{domain}_{class_name}_{i:06d}.jpg"
        meta = save_contextual_image(src, dst, class_name, domain, policy_key) if not dry_run else {
            "file_name": dst.name,
            "source_path": str(src.resolve()),
            "domain": domain,
            "class_name": class_name,
            "prompt": build_contextual_prompt(class_name, domain),
            "policy": policy_key,
        }
        rows.append(meta)
    if not dry_run:
        ensure_jsonl(out_dir / "metadata.jsonl", rows)
    return {
        "domain": domain,
        "class_name": class_name,
        "n_images": len(rows),
        "policy": policy_key,
    }


def gather_pbc() -> dict[str, list[Path]]:
    out = {c: [] for c in CLASSES}
    for cls_name, unified in PBC_MAP.items():
        src_dir = RAW_PBC / cls_name
        if src_dir.exists():
            out[unified] = sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")))
    return out


def gather_raabin() -> dict[str, list[Path]]:
    out = {c: [] for c in CLASSES}
    base = RAW_MULTI / "raabin"
    for split_dir in base.iterdir():
        if not split_dir.is_dir():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            unified = RAABIN_MAP.get(cls_dir.name.lower())
            if unified is None:
                continue
            out[unified].extend(sorted(p for p in cls_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}))
    return out


def gather_mll23() -> dict[str, list[Path]]:
    out = {c: [] for c in CLASSES}
    base = RAW_MULTI / "mll23"
    for orig_cls, unified in MLL23_MAP.items():
        found = []
        for d in base.iterdir():
            if d.is_dir() and orig_cls in d.name.lower():
                found.extend(sorted(p for p in d.rglob("*") if p.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}))
        out[unified] = found
    return out


def gather_amc() -> dict[str, list[Path]]:
    out = {c: [] for c in CLASSES}
    base = RAW_MULTI / "amc_korea" / "multi-focus-wbc-dataset"
    label_csv = base / "labels.csv"
    img_label: dict[str, str] = {}
    with open(label_csv) as f:
        for row in csv.DictReader(f):
            img_label[row["img_num"]] = row["label"]
    for img_num, label in img_label.items():
        unified = AMC_MAP.get(label)
        if unified is None:
            continue
        plane = base / f"{img_num}_5.jpg"
        if plane.exists():
            out[unified].append(plane)
    for key in out:
        out[key] = sorted(out[key])
    return out


def write_summary(records: list[dict], label: str = "all") -> None:
    payload = {
        "total_images": sum(r["n_images"] for r in records),
        "records": records,
    }
    (SUMMARY_ROOT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (SUMMARY_ROOT / f"summary_{label}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Contextual Multi-Domain Preprocess Summary",
        "",
        f"- scope: `{label}`",
        f"- Total images: `{payload['total_images']}`",
        "",
        "| Domain | Class | N | Policy |",
        "|---|---|---:|---|",
    ]
    for row in records:
        lines.append(f"| {DOMAIN_SHORT[row['domain']]} | {row['class_name']} | {row['n_images']} | {row['policy']} |")
    content = "\n".join(lines) + "\n"
    (SUMMARY_ROOT / "summary.md").write_text(content, encoding="utf-8")
    (SUMMARY_ROOT / f"summary_{label}.md").write_text(content, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Context-preserving multi-domain preprocessing")
    parser.add_argument("--domain", choices=["all", "pbc", "raabin", "mll23", "amc"], default="all")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.check:
        counts = []
        for domain in DOMAINS:
            for cls in CLASSES:
                cls_dir = OUT_ROOT / domain / cls
                n = len([p for p in cls_dir.glob("*.jpg")]) if cls_dir.exists() else 0
                counts.append({"domain": domain, "class_name": cls, "n_images": n, "policy": "existing"})
        write_summary(counts, label="check")
        print(f"[check] summary -> {SUMMARY_ROOT / 'summary.md'}")
        return

    gatherers = []
    if args.domain in {"all", "pbc"}:
        gatherers.append(("domain_a_pbc", gather_pbc(), "pbc"))
    if args.domain in {"all", "raabin"}:
        gatherers.append(("domain_b_raabin", gather_raabin(), "raabin"))
    if args.domain in {"all", "mll23"}:
        gatherers.append(("domain_c_mll23", gather_mll23(), "mll23"))
    if args.domain in {"all", "amc"}:
        gatherers.append(("domain_e_amc", gather_amc(), "amc"))

    records = []
    for domain, class_map, policy_key in gatherers:
        print(f"\n[{DOMAIN_SHORT[domain]}] policy={policy_key}")
        for class_name in CLASSES:
            srcs = class_map.get(class_name, [])
            if not srcs:
                print(f"  [skip] {class_name}: no source images")
                records.append({"domain": domain, "class_name": class_name, "n_images": 0, "policy": policy_key})
                continue
            print(f"  {class_name}: {len(srcs)} images")
            records.append(process_class_images(domain, class_name, srcs, policy_key, args.dry_run))

    if not args.dry_run:
        write_summary(records, label=args.domain)
        print(f"[done] summary -> {SUMMARY_ROOT / 'summary.md'}")
    else:
        print("[dry_run] preprocessing plan complete")


if __name__ == "__main__":
    main()
