"""
Step 7: Ablation study runner.

Runs the 4 ablation experiments:
  A1) Class-wise LoRA (8 LoRAs) vs Single LoRA + class token
  A2) text2img vs img2img (denoise 0.35)
  A3) Generated-image filter ON vs OFF
  A4) Generation multiplier: 1× vs 2× vs 5×

Each ablation calls 05_train_cnn.py or relevant scripts via subprocess,
logging results to results/ablation/.

Usage:
    python 07_ablation.py --ablation A4           # run single ablation
    python 07_ablation.py --all                   # run all 4
    python 07_ablation.py --all --dry_run         # print commands only
    python 07_ablation.py --summarize             # print table from saved JSONs
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT       = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
SCRIPTS = Path(__file__).parent


# ── Helpers ────────────────────────────────────────────────────────
def run(cmd: list[str], tag: str, dry_run: bool = False):
    print(f"\n[{tag}] " + " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] {tag} failed with exit code {result.returncode}")


def load_result(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── A1: Class-wise LoRA vs Single LoRA ────────────────────────────
def run_A1(dry_run: bool = False):
    """
    Both variants must already have generated images in:
      data/generated/  (class-wise LoRA, default)
      data/generated_single/  (single LoRA, requires separate generation run)
    Here we run the CNN training for both and compare.
    """
    print("\n" + "="*65)
    print("A1: Class-wise LoRA vs Single LoRA")
    print("NOTE: Requires both data/generated/ and data/generated_single/")

    # Class-wise
    run([PYTHON, str(SCRIPTS / "05_train_cnn.py"),
         "--mode", "real_generated", "--model", "efficientnet_b0",
         "--gen_multiplier", "1", "--denoise_tag", "ds035",
         "--seed", "42"],
        "A1-classwise", dry_run)

    # For single-LoRA you need to re-run generation pointed at data/generated_single/
    # and then modify the filtered dir accordingly — document this as a manual step.
    print("\n[A1] For Single LoRA baseline: run 02_train_lora.py in single-LoRA mode,")
    print("     re-run 03_generate.py pointing to single LoRA, then re-filter.")
    print("     Save as data/filtered_single/ and run CNN separately.")


# ── A2: text2img vs img2img ────────────────────────────────────────
def run_A2(dry_run: bool = False):
    """
    Compare:
      - img2img generations already in data/filtered/  (default pipeline)
      - text2img generations in data/filtered_t2i/  (must be created separately)
    """
    print("\n" + "="*65)
    print("A2: text2img vs img2img (denoise=0.35)")

    # img2img: already done in main pipeline
    run([PYTHON, str(SCRIPTS / "05_train_cnn.py"),
         "--mode", "real_generated", "--model", "efficientnet_b0",
         "--gen_multiplier", "1", "--denoise_tag", "ds035", "--seed", "42"],
        "A2-img2img", dry_run)

    print("\n[A2] For text2img: run 03_generate.py with --mode text2img (add flag),")
    print("     filter results, and train CNN pointing to data/filtered_t2i/.")


# ── A3: Filter ON vs OFF ───────────────────────────────────────────
def run_A3(dry_run: bool = False):
    """
    Compare:
      - filtered set (data/filtered/)  ← normal pipeline
      - unfiltered set (data/generated/ raw)
    We simulate 'filter off' by temporarily symlinking generated → filtered dir.
    """
    import shutil, os
    print("\n" + "="*65)
    print("A3: Quality filter ON vs OFF")

    # Filter ON (normal)
    run([PYTHON, str(SCRIPTS / "05_train_cnn.py"),
         "--mode", "real_generated", "--model", "efficientnet_b0",
         "--gen_multiplier", "1", "--denoise_tag", "ds035", "--seed", "42"],
        "A3-filter-ON", dry_run)

    # Filter OFF: use raw generated (no filtering)
    # Temporarily symlink data/generated → data/filtered_nofilter
    gen_dir  = ROOT / "data" / "generated"
    raw_link = ROOT / "data" / "filtered_nofilter"
    if not dry_run:
        if raw_link.exists() or raw_link.is_symlink():
            raw_link.unlink()
        os.symlink(gen_dir, raw_link)

    # Would need a flag in 05_train_cnn.py to point to filtered_nofilter
    # For ablation, manually copy results and compare
    print("\n[A3] Filter OFF: temporarily use data/generated/ as training source.")
    print("     Modify FILTERED_DIR in 05_train_cnn.py to data/generated/ and re-run.")


# ── A4: Multiplier sweep ───────────────────────────────────────────
def run_A4(dry_run: bool = False):
    """1× vs 2× vs 5× generated data."""
    print("\n" + "="*65)
    print("A4: Generation multiplier sweep (1× / 2× / 5×)")
    results = {}
    for mult in [1, 2, 5]:
        tag  = f"A4-x{mult}"
        rpath = RESULTS_DIR / f"real_generated_efficientnet_b0_x{mult}_ds035_seed42.json"
        run([PYTHON, str(SCRIPTS / "05_train_cnn.py"),
             "--mode", "real_generated", "--model", "efficientnet_b0",
             "--gen_multiplier", str(mult), "--denoise_tag", "ds035",
             "--seed", "42"],
            tag, dry_run)
        if not dry_run:
            r = load_result(rpath)
            results[f"x{mult}"] = {
                "test_accuracy": r.get("test_accuracy"),
                "test_macro_f1": r.get("test_macro_f1"),
            }

    if not dry_run and results:
        out = RESULTS_DIR / "A4_multiplier_sweep.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nA4 summary → {out}")
        for k, v in results.items():
            print(f"  {k}: acc={v['test_accuracy']:.4f}  f1={v['test_macro_f1']:.4f}")


# ── Summary table ──────────────────────────────────────────────────
def summarize():
    print("\n" + "="*65)
    print("Ablation summary (from saved JSONs)")
    print(f"{'Experiment':<40} {'Test Acc':>10} {'Macro-F1':>10}")
    print("-" * 65)

    # Collect all result JSONs from results/
    all_dirs = [
        ROOT / "results" / "baseline",
        ROOT / "results" / "augmented",
        ROOT / "results" / "ablation",
    ]
    for d in all_dirs:
        for p in sorted(d.glob("*.json")):
            r = load_result(p)
            acc = r.get("test_accuracy")
            f1  = r.get("test_macro_f1")
            if acc is not None:
                exp = r.get("experiment", p.stem)
                print(f"  {exp:<38} {acc:>10.4f} {f1:>10.4f}")


# ── Main ───────────────────────────────────────────────────────────
ABLATIONS = {"A1": run_A1, "A2": run_A2, "A3": run_A3, "A4": run_A4}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation",  type=str, choices=list(ABLATIONS),
                        default=None, help="Run single ablation (A1–A4)")
    parser.add_argument("--all",       action="store_true", help="Run all 4 ablations")
    parser.add_argument("--dry_run",   action="store_true")
    parser.add_argument("--summarize", action="store_true",
                        help="Print summary table from saved results")
    args = parser.parse_args()

    if args.summarize:
        summarize()
        return

    if args.all:
        for fn in ABLATIONS.values():
            fn(dry_run=args.dry_run)
    elif args.ablation:
        ABLATIONS[args.ablation](dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
