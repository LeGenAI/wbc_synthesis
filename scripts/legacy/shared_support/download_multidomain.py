"""
Multi-Domain WBC Dataset Downloader
====================================
Downloads 3 additional WBC datasets for cross-domain research:

  Domain B — Raabin-WBC    (Iran, 4 hospitals, Giemsa, Smartphone)
  Domain C — MLL23          (Germany, Pappenheim, Metafer scanner)
  Domain E — AMC Multi-Focus (South Korea, Romanowsky, miLab/Noul)

Only downloads the 5 target classes:
  neutrophil, lymphocyte, eosinophil, monocyte, basophil

Usage:
    python scripts/legacy/shared_support/download_multidomain.py                  # all datasets
    python scripts/legacy/shared_support/download_multidomain.py --dataset mll23
    python scripts/legacy/shared_support/download_multidomain.py --dataset raabin
    python scripts/legacy/shared_support/download_multidomain.py --dataset amc
    python scripts/legacy/shared_support/download_multidomain.py --check           # check sizes only

Storage estimate:
    MLL23   ~ 3.1 GB  (5-class ZIPs only)
    Raabin  ~ 0.5 GB  (Cropped double-labeled split)
    AMC     ~ 2.3 GB  (single ZIP)
    Total   ~ 6.0 GB
"""

import argparse
import os
import subprocess
import sys
import zipfile
import shutil
from pathlib import Path

ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw" / "multi_domain"

# ── Target classes ──────────────────────────────────────────────────
TARGET_5 = {"neutrophil", "lymphocyte", "eosinophil", "monocyte", "basophil"}

# ── MLL23: per-class ZIPs on Zenodo ────────────────────────────────
MLL23_DIR = RAW_DIR / "mll23"
MLL23_BASE = "https://zenodo.org/api/records/14277609/files"

# MLL23 class name → ZIP filename mapping (Zenodo record 14277609)
MLL23_FILES = {
    "neutrophil":  "neutrophil_segmented.zip",   # 1.63 GB
    "lymphocyte":  "lymphocyte.zip",              # 1.22 GB
    "eosinophil":  "eosinophil.zip",              # 534  MB
    "monocyte":    "monocyte.zip",                # 587  MB
    "basophil":    "basophil.zip",                # 140  MB
}

# ── Raabin-WBC: Cropped double-labeled ─────────────────────────────
RAABIN_DIR  = RAW_DIR / "raabin"
RAABIN_BASE = "http://dl.raabindata.com/WBC/Cropped_double_labeled"
RAABIN_FILES = {
    "Train.rar":  f"{RAABIN_BASE}/Train.rar",    # ~279 MB
    "TestA.rar":  f"{RAABIN_BASE}/TestA.rar",    # ~115 MB
    "TestB.zip":  f"{RAABIN_BASE}/TestB.zip",    # ~104 MB
}

# ── AMC Multi-Focus: single ZIP on Figshare ────────────────────────
AMC_DIR  = RAW_DIR / "amc_korea"
AMC_URL  = "https://ndownloader.figshare.com/files/48650791"
AMC_FILE = "multi-focus-wbc-dataset.zip"         # 2.33 GB

# ── Helpers ────────────────────────────────────────────────────────

def run(cmd: list[str], desc: str = "") -> int:
    if desc:
        print(f"  → {desc}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def wget_download(url: str, dest_dir: Path, filename: str) -> bool:
    """Download with wget: resume-capable, progress bar, skip if done."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / filename
    if out_path.exists():
        print(f"  [SKIP] already exists: {out_path.name}")
        return True
    ret = run([
        "wget", "-c",               # resume
        "--progress=bar:force",
        "-O", str(out_path),
        url,
    ], f"Downloading {filename}")
    if ret != 0:
        print(f"  [ERROR] wget failed (exit {ret}) for {url}")
        return False
    return True


def extract(archive: Path, dest_dir: Path) -> bool:
    """Extract zip/rar/anything using unar (macOS) then fallback to Python zipfile."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()
    print(f"  → Extracting {archive.name} → {dest_dir.name}/")
    if suffix == ".zip":
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest_dir)
            return True
        except Exception as e:
            print(f"  [WARN] zipfile failed ({e}), trying unar...")
    # fallback to unar (handles rar, zip, tar, etc.)
    ret = run(["unar", "-o", str(dest_dir), "-D", str(archive)])
    return ret == 0


def human_size(path: Path) -> str:
    if not path.exists():
        return "not found"
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def count_images(directory: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for f in directory.rglob("*") if f.suffix.lower() in exts)


# ── MLL23 ──────────────────────────────────────────────────────────

def download_mll23():
    print("\n" + "="*60)
    print("  DOMAIN C — MLL23 (Germany, Zenodo 14277609)")
    print("="*60)
    MLL23_DIR.mkdir(parents=True, exist_ok=True)

    for cls, fname in MLL23_FILES.items():
        print(f"\n  Class: {cls}")
        url = f"{MLL23_BASE}/{fname}/content"
        zip_path = MLL23_DIR / fname
        cls_dir  = MLL23_DIR / cls

        # Download
        if not wget_download(url, MLL23_DIR, fname):
            continue

        # Extract
        if cls_dir.exists() and count_images(cls_dir) > 0:
            print(f"  [SKIP] already extracted: {cls_dir.name}/ ({count_images(cls_dir)} images)")
            continue
        if not extract(zip_path, MLL23_DIR):
            print(f"  [ERROR] extraction failed for {fname}")
            continue

        # Rename extracted folder to class name if needed
        # Zenodo ZIPs typically extract to a subfolder matching the ZIP name stem
        extracted_stem = MLL23_DIR / fname.replace(".zip", "")
        if extracted_stem.exists() and not cls_dir.exists():
            extracted_stem.rename(cls_dir)

        n = count_images(cls_dir)
        print(f"  ✓ {cls}: {n} images → {cls_dir}")

    print(f"\n  MLL23 total size: {human_size(MLL23_DIR)}")
    _print_class_summary(MLL23_DIR)


# ── Raabin ─────────────────────────────────────────────────────────

def download_raabin():
    print("\n" + "="*60)
    print("  DOMAIN B — Raabin-WBC (Iran, dl.raabindata.com)")
    print("="*60)
    RAABIN_DIR.mkdir(parents=True, exist_ok=True)

    for fname, url in RAABIN_FILES.items():
        print(f"\n  File: {fname}")
        archive = RAABIN_DIR / fname
        if not wget_download(url, RAABIN_DIR, fname):
            continue
        # Extract
        extract_dir = RAABIN_DIR / fname.replace(".rar", "").replace(".zip", "")
        if extract_dir.exists() and count_images(extract_dir) > 0:
            print(f"  [SKIP] already extracted: {extract_dir.name}/")
            continue
        extract_dir.mkdir(exist_ok=True)
        extract(archive, extract_dir)

    print(f"\n  Raabin total size: {human_size(RAABIN_DIR)}")
    # Raabin folder structure: Train/Basophil, Train/Neutrophil, etc.
    _print_class_summary(RAABIN_DIR)


# ── AMC Korea ──────────────────────────────────────────────────────

def download_amc():
    print("\n" + "="*60)
    print("  DOMAIN E — AMC Multi-Focus (South Korea, Figshare)")
    print("="*60)
    AMC_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = AMC_DIR / AMC_FILE
    if not wget_download(AMC_URL, AMC_DIR, AMC_FILE):
        return

    if count_images(AMC_DIR) > 0:
        print(f"  [SKIP] already extracted ({count_images(AMC_DIR)} images)")
    else:
        extract(zip_path, AMC_DIR)

    print(f"\n  AMC total size: {human_size(AMC_DIR)}")
    _print_class_summary(AMC_DIR)


# ── Summary helpers ─────────────────────────────────────────────────

def _print_class_summary(base_dir: Path):
    """Walk directory and count images per class-like subfolder."""
    print(f"\n  {'Class':<20} {'Images':>8}")
    print(f"  {'-'*20} {'-'*8}")
    totals = {}
    for d in sorted(base_dir.rglob("*")):
        if d.is_dir():
            cls_name = d.name.lower()
            if any(t in cls_name for t in TARGET_5):
                n = count_images(d)
                if n > 0:
                    totals[d.name] = n
    if totals:
        for name, n in sorted(totals.items()):
            print(f"  {name:<20} {n:>8,}")
        print(f"  {'TOTAL':<20} {sum(totals.values()):>8,}")
    else:
        print("  (no class folders found yet — check after extraction)")


def check_only():
    """Print current status of all download directories."""
    print("\n=== Multi-Domain Dataset Status ===\n")
    for name, d in [("MLL23 (Germany)", MLL23_DIR),
                    ("Raabin-WBC (Iran)", RAABIN_DIR),
                    ("AMC Korea", AMC_DIR)]:
        n_files  = sum(1 for f in d.rglob("*") if f.is_file()) if d.exists() else 0
        n_images = count_images(d) if d.exists() else 0
        size_str = human_size(d) if d.exists() else "—"
        print(f"  {name:<25}  files={n_files:>5}  images={n_images:>7,}  size={size_str}")
    print()
    # Available disk
    stat = shutil.disk_usage(ROOT)
    print(f"  Disk available: {stat.free / 1e9:.1f} GB / {stat.total / 1e9:.1f} GB")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download multi-domain WBC datasets")
    parser.add_argument("--dataset", choices=["mll23", "raabin", "amc", "all"],
                        default="all", help="Which dataset to download")
    parser.add_argument("--check", action="store_true",
                        help="Only check current status, no download")
    args = parser.parse_args()

    if args.check:
        check_only()
        return

    ds = args.dataset
    if ds in ("mll23", "all"):
        download_mll23()
    if ds in ("raabin", "all"):
        download_raabin()
    if ds in ("amc", "all"):
        download_amc()

    print("\n" + "="*60)
    print("  Download complete — final status:")
    check_only()


if __name__ == "__main__":
    main()
