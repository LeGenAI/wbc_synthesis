"""
Script 44: Contextual multi-domain LoRA training for boundary-aware V2.

Outputs:
  lora/weights/contextual_multidomain_{class}/
  lora/scripts/train_contextual_multidomain_{class}.sh
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import torch

from boundary_aware_utils import (
    ROOT,
    CLASSES,
    DOMAINS,
    build_contextual_prompt,
)


DATA_DIR = ROOT / "data" / "processed_contextual_multidomain"
LORA_DIR = ROOT / "lora" / "weights"
LOG_DIR = ROOT / "logs"
SCRIPTS_DIR = ROOT / "lora" / "scripts"
TMP_ROOT = ROOT / "data" / "tmp_lora_contextual_mixed"
TRAIN_SCRIPT = ROOT / "scripts" / "train_dreambooth_lora_sdxl.py"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

for d in [LORA_DIR, LOG_DIR, SCRIPTS_DIR, TMP_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_CFG = {
    "resolution": 384,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 5e-5,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 20,
    "max_train_steps": 800,
    "rank": 8,
    "mixed_precision": "no",
    "gradient_checkpointing": False,
    "train_text_encoder": True,
    "enable_xformers": False,
    "seed": 42,
    "checkpointing_steps": 100,
    "random_flip": True,
    "center_crop": False,
    "num_validation_images": 2,
    "validation_epochs": 1,
    "resume_from_checkpoint": None,
}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sample_domain_images(class_name: str, n_per_domain: int, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    result = {}
    print(f"\n[data sampling: {class_name}]")
    for domain in DOMAINS:
        cls_dir = DATA_DIR / domain / class_name
        paths = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS) if cls_dir.exists() else []
        if len(paths) > n_per_domain:
            paths = rng.sample(paths, n_per_domain)
        result[domain] = paths
        print(f"  {domain}: {len(paths)} images")
    return result


def prepare_mixed_dir(class_name: str, domain_image_paths: dict[str, list[Path]]) -> tuple[Path, dict]:
    mixed_dir = TMP_ROOT / f"contextual_mixed_{class_name}"
    if mixed_dir.exists():
        shutil.rmtree(mixed_dir)
    mixed_dir.mkdir(parents=True)

    meta = []
    counts = {}
    for domain, paths in domain_image_paths.items():
        counts[domain] = 0
        prompt = build_contextual_prompt(class_name, domain)
        for i, src in enumerate(paths):
            dst_name = f"{domain}_{i:06d}{src.suffix.lower()}"
            shutil.copy2(src, mixed_dir / dst_name)
            meta.append({"file_name": dst_name, "text": prompt, "domain": domain, "class_name": class_name})
            counts[domain] += 1
    with open(mixed_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return mixed_dir, counts


def build_validation_prompts(class_name: str) -> list[dict]:
    same_domain = build_contextual_prompt(class_name, DOMAINS[0])
    cross_domain = build_contextual_prompt(class_name, DOMAINS[-1])
    return [
        {"name": "same_domain", "prompt": same_domain},
        {"name": "cross_domain", "prompt": cross_domain},
    ]


def latest_checkpoint_dir(out_dir: Path) -> Path | None:
    checkpoints = sorted(
        [p for p in out_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def export_latest_weights(out_dir: Path) -> None:
    latest = latest_checkpoint_dir(out_dir)
    if latest is None:
        return
    src = latest / "pytorch_lora_weights.safetensors"
    dst = out_dir / "pytorch_lora_weights.safetensors"
    if src.exists():
        shutil.copy2(src, dst)


def write_shell_script(class_name: str, mixed_dir: Path, out_dir: Path, cfg: dict) -> Path:
    validation_prompts = build_validation_prompts(class_name)
    fallback_prompt = validation_prompts[0]["prompt"]
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f'PYTHON="{sys.executable}"',
        f'SCRIPT="{TRAIN_SCRIPT}"',
        "",
        '"$PYTHON" "$SCRIPT" \\',
        f'  --pretrained_model_name_or_path "{BASE_MODEL}" \\',
        f'  --instance_data_dir "{mixed_dir}" \\',
        f'  --output_dir "{out_dir}" \\',
        f'  --instance_prompt "{fallback_prompt}" \\',
        "  --caption_column text \\",
        f'  --resolution {cfg["resolution"]} \\',
        f'  --train_batch_size {cfg["train_batch_size"]} \\',
        f'  --gradient_accumulation_steps {cfg["gradient_accumulation_steps"]} \\',
        f'  --learning_rate {cfg["learning_rate"]} \\',
        f'  --lr_scheduler {cfg["lr_scheduler"]} \\',
        f'  --lr_warmup_steps {cfg["lr_warmup_steps"]} \\',
        f'  --max_train_steps {cfg["max_train_steps"]} \\',
        f'  --rank {cfg["rank"]} \\',
        f'  --mixed_precision {cfg["mixed_precision"]} \\',
        f'  --seed {cfg["seed"]} \\',
        f'  --checkpointing_steps {cfg["checkpointing_steps"]} \\',
        f'  --validation_prompt "{fallback_prompt}" \\',
        f'  --num_validation_images {cfg["num_validation_images"]} \\',
        f'  --validation_epochs {cfg["validation_epochs"]} \\',
        "  --train_text_encoder \\",
        "  --random_flip",
    ]
    if cfg.get("resume_from_checkpoint"):
        lines[-1] += " \\"
        lines.append(f'  --resume_from_checkpoint "{cfg["resume_from_checkpoint"]}"')
    script_path = SCRIPTS_DIR / f"train_contextual_multidomain_{class_name}.sh"
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    (out_dir / "validation_prompts.json").write_text(json.dumps(validation_prompts, indent=2, ensure_ascii=False), encoding="utf-8")
    return script_path


def run_training(script_path: Path, class_name: str, dry_run: bool) -> bool:
    log_path = LOG_DIR / f"lora_contextual_multidomain_{class_name}.log"
    if dry_run:
        print(f"[dry_run] generated script -> {script_path}")
        return True
    mode = "a" if log_path.exists() else "w"
    with open(log_path, mode) as logf:
        if mode == "a":
            logf.write("\n\n=== resumed run ===\n")
        result = subprocess.run(["/bin/bash", str(script_path)], stdout=logf, stderr=subprocess.STDOUT)
    print(f"log -> {log_path}")
    return result.returncode == 0


def train_class(class_name: str, args) -> None:
    cfg = DEFAULT_CFG.copy()
    cfg["max_train_steps"] = args.steps
    cfg["rank"] = args.rank
    cfg["seed"] = args.seed
    cfg["checkpointing_steps"] = args.checkpointing_steps
    cfg["resolution"] = args.resolution
    cfg["num_validation_images"] = 0 if args.disable_validation else args.num_validation_images
    cfg["validation_epochs"] = args.validation_epochs
    cfg["resume_from_checkpoint"] = args.resume_from_checkpoint

    out_dir = LORA_DIR / f"contextual_multidomain_{class_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled = sample_domain_images(class_name, args.n_per_domain, args.seed)
    total_images = sum(len(v) for v in sampled.values())
    if total_images == 0:
        print(f"[skip] no contextual images found for {class_name}")
        return
    mixed_dir, counts = prepare_mixed_dir(class_name, sampled)
    (out_dir / "lora_config.json").write_text(json.dumps({
        "class_name": class_name,
        "base_model": BASE_MODEL,
        "device": get_device(),
        "lora_config": cfg,
        "domain_counts": counts,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    script_path = write_shell_script(class_name, mixed_dir, out_dir, cfg)
    ok = run_training(script_path, class_name, args.dry_run)
    if not args.dry_run:
        export_latest_weights(out_dir)
    if not args.keep_tmp and mixed_dir.exists() and not args.dry_run:
        shutil.rmtree(mixed_dir)
    print(f"[{'ok' if ok else 'fail'}] {class_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Contextual multi-domain LoRA training")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--class_name", choices=CLASSES)
    group.add_argument("--all_classes", action="store_true")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--n_per_domain", type=int, default=300)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--num_validation_images", type=int, default=2)
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help='checkpoint dir name or "latest"')
    parser.add_argument("--disable_validation", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_tmp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n{'=' * 60}")
    print("Script 44 - Contextual Multi-Domain LoRA Training")
    print(f"device={get_device()} steps={args.steps} rank={args.rank} n_per_domain={args.n_per_domain}")
    print(f"{'=' * 60}")

    class_names = CLASSES if args.all_classes else [args.class_name]
    for class_name in class_names:
        train_class(class_name, args)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
