"""
Step 2: Class-wise LoRA fine-tuning of SDXL base.

Strategy A (primary): one LoRA per class, trained on sharp-filtered real images.

This script generates per-class shell scripts and runs them sequentially.
Using shell scripts avoids the macOS subprocess argv empty-string injection bug
that occurs when running the dreambooth script via Python subprocess with
a list of arguments containing equals-sign arguments.

Tuned for Apple Silicon MPS:
  - resolution 256 (4x speedup vs 512, SDXL VAE still encodes to same latent space)
  - no --train_text_encoder (UNet LoRA only; saves 50%+ memory/compute)
  - no --gradient_checkpointing (faster forward pass; memory fine at res=256)
  - mixed_precision no (fp16 MPS backward has dtype issues with SDXL UNet)
  - 100 optimizer steps per class (sufficient DreamBooth convergence for
    visually homogeneous microscopy cells)
  - accumulation_steps=2, batch_size=1 (effective batch=2)

Estimated time: ~100 steps × 180s/step ≈ 5 hours per class on MPS.

Usage:
    python 02_train_lora.py --class_name neutrophil
    python 02_train_lora.py --all          # trains all 8 classes sequentially
    python 02_train_lora.py --all --dry_run  # prints commands without running
"""
import os
import stat
import subprocess
import argparse
import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
SHARP_DIR  = ROOT / "data" / "processed" / "train_sharp"
LORA_DIR   = ROOT / "lora" / "weights"
LOG_DIR    = ROOT / "logs"
SCRIPTS_DIR = ROOT / "lora" / "scripts"   # generated per-class shell scripts
LORA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
TRAIN_SCRIPT = Path(__file__).parent / "train_dreambooth_lora_sdxl.py"

# ── Per-class prompt templates (fixed) ────────────────────────────
CLASS_PROMPTS = {
    "basophil":    "microscopy image of a single basophil white blood cell, peripheral blood smear, clinical lab imaging, clear bilobed nucleus with dark granules, sharp focus, realistic",
    "eosinophil":  "microscopy image of a single eosinophil white blood cell, peripheral blood smear, clinical lab imaging, bilobed nucleus with orange-red granules, sharp focus, realistic",
    "erythroblast":"microscopy image of a single erythroblast cell, peripheral blood smear, clinical lab imaging, large round nucleus, basophilic cytoplasm, sharp focus, realistic",
    "ig":          "microscopy image of a single immature granulocyte white blood cell, peripheral blood smear, clinical lab imaging, band or metamyelocyte nucleus, sharp focus, realistic",
    "lymphocyte":  "microscopy image of a single lymphocyte white blood cell, peripheral blood smear, clinical lab imaging, large round nucleus with scant cytoplasm, sharp focus, realistic",
    "monocyte":    "microscopy image of a single monocyte white blood cell, peripheral blood smear, clinical lab imaging, kidney-shaped nucleus with grey cytoplasm, sharp focus, realistic",
    "neutrophil":  "microscopy image of a single neutrophil white blood cell, peripheral blood smear, clinical lab imaging, multilobed nucleus with pale granules, sharp focus, realistic",
    "platelet":    "microscopy image of a single platelet thrombocyte, peripheral blood smear, clinical lab imaging, small anucleate cell, sharp focus, realistic",
}

# ── Training hyperparameters (MPS-optimised) ───────────────────────
LORA_CONFIG = {
    # Resolution: 256px feed into SDXL VAE → 32×32 latents (same ratio as 512→64).
    # 4× resolution reduction → ~4× faster per step on MPS (quadratic attention).
    "resolution":                   256,
    "train_batch_size":             1,
    "gradient_accumulation_steps":  2,     # effective batch = 2
    "learning_rate":                5e-5,  # slightly lower LR for fewer steps
    "lr_scheduler":                 "cosine",
    "lr_warmup_steps":              10,
    "max_train_steps":              100,   # ~5h per class on MPS at 256px
    "rank":                         8,     # lower rank sufficient for fine style adaptation
    "mixed_precision":              "no",  # full fp32; fp16 backward broken on MPS with SDXL
    "gradient_checkpointing":       False, # off for speed (no memory issue at 256px)
    "train_text_encoder":           False, # UNet LoRA only; saves 50% compute
    "enable_xformers":              False, # CUDA-only
    "seed":                         42,
    "checkpointing_steps":          9999,  # no intermediate checkpoints; final only
    "random_flip":                  True,
}


def count_images(class_dir: Path) -> int:
    return sum(1 for p in class_dir.rglob("*")
               if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def write_shell_script(class_name: str, class_dir: Path, out_dir: Path) -> Path:
    """Write a per-class shell script that calls train_dreambooth_lora_sdxl.py.

    Using a shell script avoids the macOS subprocess empty-string-argv bug
    where Python subprocess on macOS inserts '' entries when the Bash shell
    profile adds extra items to sys.argv.
    """
    prompt = CLASS_PROMPTS.get(
        class_name,
        f"microscopy image of a single {class_name} white blood cell, "
        "peripheral blood smear, realistic"
    )
    steps = LORA_CONFIG["max_train_steps"]
    import sys
    python_bin = sys.executable

    lines = [
        "#!/bin/bash",
        f"# Auto-generated LoRA training script for class: {class_name}",
        f"set -euo pipefail",
        "",
        f"PYTHON={python_bin}",
        f'SCRIPT="{TRAIN_SCRIPT}"',
        "",
        f'"$PYTHON" "$SCRIPT" \\',
        f'  --pretrained_model_name_or_path "{BASE_MODEL}" \\',
        f'  --instance_data_dir "{class_dir}" \\',
        f'  --output_dir "{out_dir}" \\',
        f'  --instance_prompt "{prompt}" \\',
        f'  --resolution {LORA_CONFIG["resolution"]} \\',
        f'  --train_batch_size {LORA_CONFIG["train_batch_size"]} \\',
        f'  --gradient_accumulation_steps {LORA_CONFIG["gradient_accumulation_steps"]} \\',
        f'  --learning_rate {LORA_CONFIG["learning_rate"]} \\',
        f'  --lr_scheduler {LORA_CONFIG["lr_scheduler"]} \\',
        f'  --lr_warmup_steps {LORA_CONFIG["lr_warmup_steps"]} \\',
        f'  --max_train_steps {steps} \\',
        f'  --rank {LORA_CONFIG["rank"]} \\',
        f'  --mixed_precision {LORA_CONFIG["mixed_precision"]} \\',
        f'  --seed {LORA_CONFIG["seed"]} \\',
        f'  --checkpointing_steps {LORA_CONFIG["checkpointing_steps"]} \\',
    ]
    if LORA_CONFIG["train_text_encoder"]:
        lines.append("  --train_text_encoder \\")
    if LORA_CONFIG["gradient_checkpointing"]:
        lines.append("  --gradient_checkpointing \\")
    if LORA_CONFIG["enable_xformers"]:
        lines.append("  --enable_xformers_memory_efficient_attention \\")
    if LORA_CONFIG["random_flip"]:
        lines.append("  --random_flip \\")
    # Remove trailing backslash from last argument line
    lines[-1] = lines[-1].rstrip(" \\")

    script_path = SCRIPTS_DIR / f"train_{class_name}.sh"
    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Make executable
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script_path


def train_class(class_name: str, dry_run: bool = False):
    class_dir = SHARP_DIR / class_name
    if not class_dir.exists():
        print(f"[SKIP] {class_name}: no sharp data at {class_dir}")
        return

    n = count_images(class_dir)
    steps = LORA_CONFIG["max_train_steps"]
    out_dir = LORA_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Class: {class_name}  |  images: {n}  |  steps: {steps}")
    print(f"Output: {out_dir}")

    script_path = write_shell_script(class_name, class_dir, out_dir)
    print(f"Shell script: {script_path}")

    # Save config JSON for reference
    cfg = {
        **LORA_CONFIG,
        "class": class_name,
        "n_images": n,
        "max_train_steps": steps,
        "prompt": CLASS_PROMPTS.get(class_name, ""),
    }
    with open(out_dir / "lora_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    if dry_run:
        print("[DRY RUN] Shell script written but not executed.")
        print(open(script_path).read())
        return

    log_file = LOG_DIR / f"lora_{class_name}.log"
    print(f"Logging to {log_file}")
    print(f"Training started … (est. {steps * 180 / 3600:.1f}h on MPS at 256px)")
    with open(log_file, "w") as lf:
        result = subprocess.run(
            ["/bin/bash", str(script_path)],
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        print(f"[ERROR] Training failed for {class_name} (exit={result.returncode}). "
              f"Check {log_file}")
    else:
        lora_file = out_dir / "pytorch_lora_weights.safetensors"
        size_mb = lora_file.stat().st_size / 1e6 if lora_file.exists() else 0
        print(f"[OK] Training complete for {class_name}  "
              f"(LoRA: {size_mb:.1f} MB, log: {log_file})")


def main():
    parser = argparse.ArgumentParser(
        description="Class-wise LoRA training for WBC SDXL synthesis (MPS-optimised)"
    )
    parser.add_argument("--class_name", type=str, default=None,
                        help="Single class to train")
    parser.add_argument("--all", action="store_true",
                        help="Train all discovered classes sequentially")
    parser.add_argument("--dry_run", action="store_true",
                        help="Write shell scripts but do not execute")
    args = parser.parse_args()

    if args.all:
        classes = [d.name for d in sorted(SHARP_DIR.iterdir()) if d.is_dir()]
        print(f"Training LoRA for {len(classes)} classes: {classes}")
        for cls in classes:
            train_class(cls, dry_run=args.dry_run)
    elif args.class_name:
        train_class(args.class_name, dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
