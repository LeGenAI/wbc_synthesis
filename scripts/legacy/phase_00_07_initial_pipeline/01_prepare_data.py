"""
Step 1: Data preparation pipeline.
- Discovers class folders under data/raw/
- Stratified Train/Val/Test split (70/15/15)
- Center-crop 360×360 → resize 512×512
- Saves processed images to data/processed/{train,val,test}/{class}/
- Applies Laplacian-variance sharpness filter on Train split
  and saves "sharp" subset to data/processed/train_sharp/{class}/
  (used only for LoRA fine-tuning)
"""
import os
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────
RAW_DIR       = Path(__file__).parent.parent / "data" / "raw" / "PBC_dataset_normal_DIB_224" / "PBC_dataset_normal_DIB_224"
PROC_DIR      = Path(__file__).parent.parent / "data" / "processed"
SHARP_DIR     = PROC_DIR / "train_sharp"
TARGET_SIZE   = 512            # output resolution (square)
SHARP_PCTILE  = 70             # keep top-N% by Laplacian variance
SEEDS         = [0, 1, 2]      # repeat experiments with these seeds
DEFAULT_SEED  = 0
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15


def laplacian_variance(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def center_crop_and_resize(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = img[y0:y0+s, x0:x0+s]
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_LANCZOS4)


def find_class_dirs(root: Path):
    """Return {class_name: [image_path, ...]} from a flat class-folder layout."""
    classes = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        imgs = [
            p for p in d.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ]
        if imgs:
            classes[d.name] = imgs
    return classes


def process_split(paths, class_name, split_dir, target_size):
    out_dir = split_dir / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in tqdm(paths, desc=f"  {split_dir.name}/{class_name}", leave=False):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = center_crop_and_resize(img, target_size)
        dst = out_dir / p.name
        cv2.imwrite(str(dst), img)
    return out_dir


def sharp_filter(split_dir, class_name, sharp_dir, pctile):
    src_dir = split_dir / class_name
    imgs = list(src_dir.glob("*"))
    scores = [(p, laplacian_variance(cv2.imread(str(p)))) for p in imgs
              if cv2.imread(str(p)) is not None]
    if not scores:
        return
    threshold = np.percentile([s for _, s in scores], 100 - pctile)
    out_dir = sharp_dir / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for p, s in scores:
        if s >= threshold:
            shutil.copy2(p, out_dir / p.name)
            kept += 1
    print(f"    [{class_name}] sharp filter: {kept}/{len(scores)} kept "
          f"(threshold Lap.var ≥ {threshold:.1f})")


def main(seed: int, val_ratio: float, test_ratio: float,
         target_size: int, sharp_pctile: int):

    print(f"\n=== Data preparation  seed={seed} ===")
    class_dirs = find_class_dirs(RAW_DIR)
    if not class_dirs:
        raise RuntimeError(
            f"No class folders found under {RAW_DIR}. "
            "Run 00_download_data.py first."
        )

    print(f"Found {len(class_dirs)} classes: {list(class_dirs.keys())}")

    train_dir = PROC_DIR / "train"
    val_dir   = PROC_DIR / "val"
    test_dir  = PROC_DIR / "test"

    stats = {}
    for cls, paths in class_dirs.items():
        print(f"\n[{cls}] total={len(paths)}")
        # Stratified split
        rel_test  = test_ratio
        rel_val   = val_ratio / (1.0 - rel_test)

        train_val, test_paths = train_test_split(
            paths, test_size=rel_test, random_state=seed
        )
        train_paths, val_paths = train_test_split(
            train_val, test_size=rel_val, random_state=seed
        )
        print(f"  split → train:{len(train_paths)}  val:{len(val_paths)}  test:{len(test_paths)}")
        stats[cls] = dict(train=len(train_paths), val=len(val_paths), test=len(test_paths))

        # Process each split
        process_split(train_paths, cls, train_dir, target_size)
        process_split(val_paths,   cls, val_dir,   target_size)
        process_split(test_paths,  cls, test_dir,  target_size)

        # Sharpness filter on train only (for LoRA)
        print(f"  Applying sharp filter (top {sharp_pctile}%) for LoRA data...")
        sharp_filter(train_dir, cls, SHARP_DIR, sharp_pctile)

    # Summary
    print("\n=== Split summary ===")
    for cls, s in stats.items():
        print(f"  {cls:20s}  train={s['train']:5d}  val={s['val']:5d}  test={s['test']:5d}")

    # Save stats
    import json
    out_json = PROC_DIR / f"split_stats_seed{seed}.json"
    with open(out_json, "w") as f:
        json.dump({"seed": seed, "classes": stats}, f, indent=2)
    print(f"\nStats saved → {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int,   default=DEFAULT_SEED)
    parser.add_argument("--val_ratio",    type=float, default=VAL_RATIO)
    parser.add_argument("--test_ratio",   type=float, default=TEST_RATIO)
    parser.add_argument("--target_size",  type=int,   default=TARGET_SIZE)
    parser.add_argument("--sharp_pctile", type=int,   default=SHARP_PCTILE)
    args = parser.parse_args()
    main(args.seed, args.val_ratio, args.test_ratio,
         args.target_size, args.sharp_pctile)
