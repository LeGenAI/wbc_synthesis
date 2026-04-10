"""
Multi-Domain WBC Preprocessing & Normalization
================================================
4개 도메인 원본 데이터를 통일된 포맷으로 변환:

  Domain A — PBC Barcelona   (원본 그대로, 참조용)
  Domain B — Raabin-WBC      (Train+TestA+TestB → 클래스 폴더 통합)
  Domain C — MLL23           (TIF 288px → JPG 224px)
  Domain E — AMC Korea       (z=5 plane 선택, CSV 파싱 → 클래스 폴더)

출력 구조:
  data/processed_multidomain/
  ├── domain_a_pbc/
  │   ├── basophil/    neutrophil/    ...
  ├── domain_b_raabin/
  │   ├── basophil/    neutrophil/    ...
  ├── domain_c_mll23/
  │   ├── basophil/    neutrophil/    ...
  └── domain_e_amc/
      ├── basophil/    neutrophil/    ...

각 이미지:
  - 224 × 224 px, center-crop → resize (Lanczos)
  - JPG, quality=95
  - 파일명: {domain}_{class}_{index:06d}.jpg

Usage:
    python scripts/legacy/shared_support/preprocess_multidomain.py           # 전체
    python scripts/legacy/shared_support/preprocess_multidomain.py --domain raabin
    python scripts/legacy/shared_support/preprocess_multidomain.py --domain mll23
    python scripts/legacy/shared_support/preprocess_multidomain.py --domain amc
    python scripts/legacy/shared_support/preprocess_multidomain.py --domain pbc
    python scripts/legacy/shared_support/preprocess_multidomain.py --check   # 현황만 출력
"""

import argparse
import csv
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── 경로 설정 ────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
RAW_MULTI = ROOT / "data" / "raw" / "multi_domain"
RAW_PBC   = ROOT / "data" / "raw" / "PBC_dataset_normal_DIB_224" / "PBC_dataset_normal_DIB_224"
OUT_ROOT  = ROOT / "data" / "processed_multidomain"

# ── 5개 대상 클래스 (소문자 통일 키) ───────────────────────────────
TARGET_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]

# ── 각 도메인별 클래스명 매핑 (원본명 → 통일명) ─────────────────────
RAABIN_MAP = {
    "basophil":   "basophil",
    "eosinophil": "eosinophil",
    "lymphocyte": "lymphocyte",
    "monocyte":   "monocyte",
    "neutrophil": "neutrophil",
}
MLL23_MAP = {
    "basophil":           "basophil",
    "eosinophil":         "eosinophil",
    "lymphocyte":         "lymphocyte",
    "monocyte":           "monocyte",
    "neutrophil":         "neutrophil",   # 폴더명: neutrophil (segmented)
}
AMC_MAP = {
    "basophil":        "basophil",
    "eosinophil":      "eosinophil",
    "lymphocyte":      "lymphocyte",
    "monocyte":        "monocyte",
    "seg_neutrophil":  "neutrophil",
}
PBC_MAP = {
    "basophil":   "basophil",
    "eosinophil": "eosinophil",
    "lymphocyte": "lymphocyte",
    "monocyte":   "monocyte",
    "neutrophil": "neutrophil",
}

SIZE = 224  # 출력 해상도

# ── 이미지 변환 유틸 ─────────────────────────────────────────────────

def center_crop_resize(img: Image.Image, size: int = SIZE) -> Image.Image:
    """정방형 center-crop 후 size×size 리사이즈."""
    img = img.convert("RGB")
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top  = (h - m) // 2
    img  = img.crop((left, top, left + m, top + m))
    return img.resize((size, size), Image.LANCZOS)


def save_jpg(img: Image.Image, path: Path, quality: int = 95):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=quality)


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*")
               if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


# ── Domain A: PBC Barcelona ──────────────────────────────────────────

def preprocess_pbc():
    print("\n" + "="*60)
    print("  Domain A — PBC Barcelona")
    print("="*60)
    out_domain = OUT_ROOT / "domain_a_pbc"

    for cls_name, unified in PBC_MAP.items():
        src_dir = RAW_PBC / cls_name
        if not src_dir.exists():
            print(f"  [SKIP] {cls_name}: 소스 없음 ({src_dir})")
            continue

        out_cls = out_domain / unified
        existing = count_images(out_cls)

        src_imgs = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
        if existing >= len(src_imgs):
            print(f"  [SKIP] {unified}: 이미 {existing}장")
            continue

        print(f"  {unified}: {len(src_imgs)}장 처리 중...")
        for i, p in enumerate(tqdm(src_imgs, desc=f"    pbc/{unified}", leave=False)):
            try:
                img = center_crop_resize(Image.open(p))
                save_jpg(img, out_cls / f"pbc_{unified}_{i:06d}.jpg")
            except Exception as e:
                print(f"    [WARN] {p.name}: {e}")

        print(f"  ✅ {unified}: {count_images(out_cls)}장 → {out_cls}")


# ── Domain B: Raabin-WBC ─────────────────────────────────────────────

def preprocess_raabin():
    print("\n" + "="*60)
    print("  Domain B — Raabin-WBC (Iran)")
    print("="*60)
    raabin_raw = RAW_MULTI / "raabin"
    out_domain = OUT_ROOT / "domain_b_raabin"

    # 통합 클래스 딕셔너리: unified_name → [이미지 경로 리스트]
    cls_paths: dict[str, list[Path]] = {c: [] for c in TARGET_CLASSES}

    # Train / TestA / TestB / Test-B 폴더 모두 탐색
    for split_dir in raabin_raw.iterdir():
        if not split_dir.is_dir():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            orig = cls_dir.name.lower()
            unified = RAABIN_MAP.get(orig)
            if unified is None:
                continue
            imgs = [p for p in cls_dir.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            cls_paths[unified].extend(imgs)

    for unified, paths in cls_paths.items():
        out_cls = out_domain / unified
        existing = count_images(out_cls)
        if existing >= len(paths):
            print(f"  [SKIP] {unified}: 이미 {existing}장")
            continue
        print(f"  {unified}: {len(paths)}장 처리 중...")
        for i, p in enumerate(tqdm(paths, desc=f"    raabin/{unified}", leave=False)):
            try:
                img = center_crop_resize(Image.open(p))
                save_jpg(img, out_cls / f"raabin_{unified}_{i:06d}.jpg")
            except Exception as e:
                print(f"    [WARN] {p.name}: {e}")
        print(f"  ✅ {unified}: {count_images(out_cls)}장 → {out_cls}")


# ── Domain C: MLL23 ──────────────────────────────────────────────────

def preprocess_mll23():
    print("\n" + "="*60)
    print("  Domain C — MLL23 (Germany)")
    print("="*60)
    mll23_raw  = RAW_MULTI / "mll23"
    out_domain = OUT_ROOT / "domain_c_mll23"

    for orig_cls, unified in MLL23_MAP.items():
        # MLL23 폴더명이 다를 수 있으므로 유연하게 매칭
        src_dir = None
        for d in mll23_raw.iterdir():
            if d.is_dir() and orig_cls in d.name.lower():
                src_dir = d
                break
        if src_dir is None:
            print(f"  [SKIP] {orig_cls}: 폴더 없음")
            continue

        out_cls = out_domain / unified
        src_imgs = [p for p in src_dir.rglob("*")
                    if p.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}]
        existing = count_images(out_cls)
        if existing >= len(src_imgs):
            print(f"  [SKIP] {unified}: 이미 {existing}장")
            continue

        print(f"  {unified}: {len(src_imgs)}장 처리 중... (TIF→JPG 변환)")
        for i, p in enumerate(tqdm(src_imgs, desc=f"    mll23/{unified}", leave=False)):
            try:
                img = center_crop_resize(Image.open(p))
                save_jpg(img, out_cls / f"mll23_{unified}_{i:06d}.jpg")
            except Exception as e:
                print(f"    [WARN] {p.name}: {e}")
        print(f"  ✅ {unified}: {count_images(out_cls)}장 → {out_cls}")


# ── Domain E: AMC Korea ──────────────────────────────────────────────

def preprocess_amc():
    print("\n" + "="*60)
    print("  Domain E — AMC Multi-Focus (South Korea)")
    print("="*60)
    amc_raw    = RAW_MULTI / "amc_korea" / "multi-focus-wbc-dataset"
    out_domain = OUT_ROOT / "domain_e_amc"

    # labels.csv 파싱: img_num → label
    label_csv = amc_raw / "labels.csv"
    if not label_csv.exists():
        print(f"  [ERROR] labels.csv 없음: {label_csv}")
        return

    img_label: dict[str, str] = {}
    with open(label_csv) as f:
        for row in csv.DictReader(f):
            img_label[row["img_num"]] = row["label"]

    # 5클래스만 필터링 (z=5 plane 선택)
    cls_paths: dict[str, list[Path]] = {c: [] for c in TARGET_CLASSES}
    for img_num, label in img_label.items():
        unified = AMC_MAP.get(label)
        if unified is None:
            continue
        z5_file = amc_raw / f"{img_num}_5.jpg"
        if z5_file.exists():
            cls_paths[unified].append(z5_file)

    for unified, paths in cls_paths.items():
        out_cls = out_domain / unified
        existing = count_images(out_cls)
        if existing >= len(paths):
            print(f"  [SKIP] {unified}: 이미 {existing}장")
            continue
        print(f"  {unified}: {len(paths)}장 처리 중... (z=5 plane)")
        for i, p in enumerate(tqdm(paths, desc=f"    amc/{unified}", leave=False)):
            try:
                img = center_crop_resize(Image.open(p))
                save_jpg(img, out_cls / f"amc_{unified}_{i:06d}.jpg")
            except Exception as e:
                print(f"    [WARN] {p.name}: {e}")
        print(f"  ✅ {unified}: {count_images(out_cls)}장 → {out_cls}")


# ── 현황 출력 ────────────────────────────────────────────────────────

def check_status():
    print("\n=== 전처리 현황 ===\n")
    domains = {
        "domain_a_pbc":    "PBC Barcelona (Spain)",
        "domain_b_raabin": "Raabin-WBC    (Iran)",
        "domain_c_mll23":  "MLL23         (Germany)",
        "domain_e_amc":    "AMC Korea     (Korea)",
    }

    grand_total = 0
    print(f"  {'도메인':<28} {'basophil':>9} {'eosinophil':>11} {'lymphocyte':>11} {'monocyte':>9} {'neutrophil':>11} {'합계':>8}")
    print("  " + "-"*90)

    for domain_dir, label in domains.items():
        base = OUT_ROOT / domain_dir
        counts = {}
        for cls in TARGET_CLASSES:
            d = base / cls
            counts[cls] = count_images(d) if d.exists() else 0
        total = sum(counts.values())
        grand_total += total
        print(f"  {label:<28} "
              f"{counts['basophil']:>9,} "
              f"{counts['eosinophil']:>11,} "
              f"{counts['lymphocyte']:>11,} "
              f"{counts['monocyte']:>9,} "
              f"{counts['neutrophil']:>11,} "
              f"{total:>8,}")

    print("  " + "-"*90)
    print(f"  {'TOTAL':<28} {'':>9} {'':>11} {'':>11} {'':>9} {'':>11} {grand_total:>8,}")
    print()

    # 디스크
    if OUT_ROOT.exists():
        sz = sum(f.stat().st_size for f in OUT_ROOT.rglob("*") if f.is_file())
        print(f"  processed_multidomain 총 용량: {sz/1e9:.2f} GB")
    stat = shutil.disk_usage(ROOT)
    print(f"  디스크 여유: {stat.free/1e9:.1f} GB")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["pbc", "raabin", "mll23", "amc", "all"],
                        default="all")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_status()
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    d = args.domain
    if d in ("pbc",    "all"): preprocess_pbc()
    if d in ("raabin", "all"): preprocess_raabin()
    if d in ("mll23",  "all"): preprocess_mll23()
    if d in ("amc",    "all"): preprocess_amc()

    print()
    check_status()


if __name__ == "__main__":
    main()
