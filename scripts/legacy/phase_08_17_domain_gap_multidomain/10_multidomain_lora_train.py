"""
Script 10: Multi-Domain LoRA Training
=======================================
4개 도메인 이미지를 균등 혼합하여 도메인별 조건부 프롬프트로 LoRA 학습.

각 이미지에 도메인별 캡션을 부여하기 위해 metadata.jsonl 을 사용하며,
기존 02_train_lora.py + train_dreambooth_lora_sdxl.py 패턴을 그대로 재사용.

출력:
  lora/weights/multidomain_{class_name}/pytorch_lora_weights.safetensors
  lora/weights/multidomain_{class_name}/lora_config.json
  lora/scripts/train_multidomain_{class_name}.sh

Usage:
    # dry-run (스크립트 생성만, 실행 안 함)
    python scripts/legacy/phase_08_17_domain_gap_multidomain/10_multidomain_lora_train.py --class_name basophil --dry_run

    # 실제 학습 (basophil)
    python scripts/legacy/phase_08_17_domain_gap_multidomain/10_multidomain_lora_train.py --class_name basophil --steps 400

    # 5클래스 순차 학습
    python scripts/legacy/phase_08_17_domain_gap_multidomain/10_multidomain_lora_train.py --all_classes --steps 400
"""

import argparse
import json
import os
import random
import shutil
import stat
import sys
import subprocess
import warnings
from pathlib import Path

import torch
from tqdm import tqdm

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "processed_multidomain"
LORA_DIR    = ROOT / "lora" / "weights"
LOG_DIR     = ROOT / "logs"
SCRIPTS_DIR = ROOT / "lora" / "scripts"
TMP_ROOT    = ROOT / "data" / "tmp_lora_mixed"
TRAIN_SCRIPT = Path(__file__).parent / "train_dreambooth_lora_sdxl.py"
BASE_MODEL  = "stabilityai/stable-diffusion-xl-base-1.0"

for d in [LORA_DIR, LOG_DIR, SCRIPTS_DIR, TMP_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

# ── 도메인 / 클래스 메타데이터 ────────────────────────────────────────
DOMAINS = [
    "domain_a_pbc",
    "domain_b_raabin",
    "domain_c_mll23",
    "domain_e_amc",
]
DOMAIN_PROMPTS = {
    "domain_a_pbc":    "May-Grünwald Giemsa stain, CellaVision automated analyzer, Barcelona Spain",
    "domain_b_raabin": "Giemsa stain, smartphone microscope camera, Iran hospital",
    "domain_c_mll23":  "Pappenheim stain, Metafer scanner, Germany clinical lab",
    "domain_e_amc":    "Romanowsky stain, miLab automated analyzer, South Korea AMC",
}
CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_MORPHOLOGY = {
    "basophil":   "bilobed nucleus with dark purple-black granules filling cytoplasm",
    "eosinophil": "bilobed nucleus with bright orange-red granules",
    "lymphocyte": "large round nucleus with scant agranular cytoplasm",
    "monocyte":   "kidney-shaped or folded nucleus with grey-blue cytoplasm",
    "neutrophil": "multilobed nucleus with pale pink granules",
}
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# ── LoRA 하이퍼파라미터 (02_train_lora.py 에서 복사, steps 조정) ──────
LORA_CONFIG = {
    "resolution":               256,   # MPS 4× 속도 최적화
    "train_batch_size":         1,
    "gradient_accumulation_steps": 2,  # effective batch = 2
    "learning_rate":            5e-5,
    "lr_scheduler":             "cosine",
    "lr_warmup_steps":          10,
    "max_train_steps":          400,   # 4도메인 × 100
    "rank":                     8,
    "mixed_precision":          "no",  # MPS fp16 backward 버그 우회
    "gradient_checkpointing":   False,
    "train_text_encoder":       False,
    "enable_xformers":          False,
    "seed":                     42,
    "checkpointing_steps":      100,
    "random_flip":              True,
    "center_crop":              True,
    "validation_prompt":        None,
    "num_validation_images":    0,
    "validation_epochs":        1,
}


# ── 디바이스 ──────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── 프롬프트 빌더 ────────────────────────────────────────────────────
def build_instance_prompt(class_name: str, domain: str) -> str:
    """도메인별 조건부 인스턴스 프롬프트 생성."""
    morphology = CLASS_MORPHOLOGY[class_name]
    domain_ctx = DOMAIN_PROMPTS[domain]
    return (
        f"microscopy image of a single {class_name} white blood cell, "
        f"peripheral blood smear, {morphology}, {domain_ctx}, "
        f"sharp focus, realistic, clinical lab imaging"
    )


# ── 이미지 경로 샘플링 ────────────────────────────────────────────────
def sample_domain_images(
    data_dir: Path,
    class_name: str,
    domains: list,
    n_per_domain: int,
    seed: int = 42,
) -> dict:
    """
    반환: {domain: [path, ...]}
    각 도메인에서 최대 n_per_domain 장 균등 샘플링.
    """
    rng    = random.Random(seed)
    result = {}
    print(f"\n  [데이터 샘플링: {class_name}]")
    for domain in domains:
        cls_dir = data_dir / domain / class_name
        if not cls_dir.exists():
            warnings.warn(f"  [WARN] 디렉토리 없음: {cls_dir}")
            result[domain] = []
            continue
        paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        if len(paths) > n_per_domain:
            paths = rng.sample(paths, n_per_domain)
        elif len(paths) < n_per_domain:
            print(f"  [INFO] {domain}/{class_name}: {len(paths)}장 "
                  f"(요청 {n_per_domain} 미달, 전수 사용)")
        result[domain] = paths
        print(f"    {domain}: {len(paths)}장")
    return result


# ── 혼합 데이터셋 디렉토리 준비 ──────────────────────────────────────
def prepare_mixed_dataset_dir(
    class_name: str,
    domain_image_paths: dict,
    out_root: Path,
) -> tuple:
    """
    이미지를 하나의 디렉토리로 복사 + metadata.jsonl 생성.
    반환: (mixed_dir: Path, metadata: dict[domain: n_images])
    """
    mixed_dir = out_root / f"mixed_{class_name}"
    if mixed_dir.exists():
        shutil.rmtree(mixed_dir)
    mixed_dir.mkdir(parents=True)

    meta_lines = []
    metadata   = {}
    total      = 0

    for domain, paths in domain_image_paths.items():
        prompt = build_instance_prompt(class_name, domain)
        count  = 0
        for i, src_path in enumerate(paths):
            fname = f"{domain}_{i:06d}{src_path.suffix.lower()}"
            dst   = mixed_dir / fname
            try:
                shutil.copy2(src_path, dst)
                meta_lines.append(
                    json.dumps({"file_name": fname, "text": prompt}, ensure_ascii=False)
                )
                count += 1
                total += 1
            except Exception as e:
                warnings.warn(f"  [WARN] 복사 실패 ({src_path}): {e}")
        metadata[domain] = count

    # metadata.jsonl 저장 (HuggingFace datasets 형식)
    jsonl_path = mixed_dir / "metadata.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines) + "\n")

    print(f"\n  혼합 데이터셋 준비 완료: {mixed_dir}")
    print(f"  도메인별: {metadata}")
    print(f"  총 {total}장 + metadata.jsonl")
    return mixed_dir, metadata


# ── 학습 쉘 스크립트 작성 ────────────────────────────────────────────
def write_multidomain_shell_script(
    class_name: str,
    mixed_dir: Path,
    out_dir: Path,
    lora_config: dict,
) -> Path:
    """
    02_train_lora.py 의 write_shell_script() 패턴 재사용.
    metadata.jsonl 이 mixed_dir 안에 생성되므로 --instance_data_dir 로 로컬 폴더를 지정하고
    --caption_column text 파라미터를 함께 넘긴다 (per-image 도메인 캡션 지원).
    --instance_prompt 는 metadata 미매칭 fallback 용으로만 사용한다.
    """
    python_bin    = sys.executable
    steps         = lora_config["max_train_steps"]
    # fallback 프롬프트 (metadata.jsonl 에서 읽지 못할 경우 사용)
    fallback_prompt = build_instance_prompt(class_name, "domain_a_pbc")

    lines = [
        "#!/bin/bash",
        f"# Multi-domain LoRA training script for class: {class_name}",
        "set -euo pipefail",
        "",
        f"PYTHON={python_bin}",
        f'SCRIPT="{TRAIN_SCRIPT}"',
        "",
        f'"$PYTHON" "$SCRIPT" \\',
        f'  --pretrained_model_name_or_path "{BASE_MODEL}" \\',
        f'  --instance_data_dir "{mixed_dir}" \\',
        f'  --output_dir "{out_dir}" \\',
        f'  --instance_prompt "{fallback_prompt}" \\',
        f'  --caption_column text \\',
        f'  --resolution {lora_config["resolution"]} \\',
        f'  --train_batch_size {lora_config["train_batch_size"]} \\',
        f'  --gradient_accumulation_steps {lora_config["gradient_accumulation_steps"]} \\',
        f'  --learning_rate {lora_config["learning_rate"]} \\',
        f'  --lr_scheduler {lora_config["lr_scheduler"]} \\',
        f'  --lr_warmup_steps {lora_config["lr_warmup_steps"]} \\',
        f'  --max_train_steps {steps} \\',
        f'  --rank {lora_config["rank"]} \\',
        f'  --mixed_precision {lora_config["mixed_precision"]} \\',
        f'  --seed {lora_config["seed"]} \\',
        f'  --checkpointing_steps {lora_config["checkpointing_steps"]} \\',
    ]
    if lora_config.get("train_text_encoder"):
        lines.append("  --train_text_encoder \\")
    if lora_config.get("gradient_checkpointing"):
        lines.append("  --gradient_checkpointing \\")
    if lora_config.get("enable_xformers"):
        lines.append("  --enable_xformers_memory_efficient_attention \\")
    if lora_config.get("center_crop"):
        lines.append("  --center_crop \\")
    if lora_config.get("random_flip"):
        lines.append("  --random_flip \\")
    validation_prompt = lora_config.get("validation_prompt")
    if validation_prompt:
        lines.extend([
            f'  --validation_prompt "{validation_prompt}" \\',
            f'  --num_validation_images {lora_config["num_validation_images"]} \\',
            f'  --validation_epochs {lora_config["validation_epochs"]} \\',
        ])

    # 마지막 역슬래시 제거
    lines[-1] = lines[-1].rstrip(" \\")

    script_path = SCRIPTS_DIR / f"train_multidomain_{class_name}.sh"
    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    print(f"\n  쉘 스크립트 생성: {script_path}")
    if True:  # 항상 스크립트 내용 일부 출력
        with open(script_path) as f:
            preview = f.readlines()
        print("  --- 스크립트 미리보기 (첫 15줄) ---")
        for line in preview[:15]:
            print("  " + line, end="")
        if len(preview) > 15:
            print(f"  ... (총 {len(preview)}줄)")
        print("  ---")

    return script_path


# ── 학습 설정 JSON 저장 ───────────────────────────────────────────────
def save_training_config(
    class_name: str,
    out_dir: Path,
    domain_image_paths: dict,
    lora_config: dict,
) -> None:
    config = {
        "class_name":     class_name,
        "base_model":     BASE_MODEL,
        "lora_config":    lora_config,
        "domains": {
            domain: {
                "n_images": len(paths),
                "prompt":   build_instance_prompt(class_name, domain),
            }
            for domain, paths in domain_image_paths.items()
        },
        "total_images": sum(len(p) for p in domain_image_paths.values()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "lora_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 학습 설정 저장: {cfg_path}")


# ── 학습 실행 ─────────────────────────────────────────────────────────
def run_training(script_path: Path, class_name: str, dry_run: bool = False) -> bool:
    log_path = LOG_DIR / f"lora_multidomain_{class_name}.log"
    if dry_run:
        print(f"\n  [DRY RUN] 실제 학습 없이 완료. 스크립트: {script_path}")
        return True

    print(f"\n  학습 시작: {class_name}")
    print(f"  로그: {log_path}")
    print(f"  예상 시간: ~{LORA_CONFIG['max_train_steps'] * 3 // 60}분 (MPS 기준)\n")

    with open(log_path, "w") as logf:
        result = subprocess.run(
            ["/bin/bash", str(script_path)],
            stdout=logf, stderr=subprocess.STDOUT,
        )

    if result.returncode == 0:
        print(f"  ✅ 학습 완료: {class_name}")
        return True
    else:
        print(f"  [ERROR] 학습 실패 (exit {result.returncode})")
        print(f"  로그 확인: {log_path}")
        return False


# ── 단일 클래스 멀티도메인 LoRA 학습 ──────────────────────────────────
def train_class_multidomain(class_name: str, args) -> None:
    print(f"\n{'='*60}")
    print(f"  Multi-Domain LoRA: {class_name}")
    print(f"{'='*60}")

    cfg = LORA_CONFIG.copy()
    cfg["max_train_steps"] = args.steps
    cfg["rank"]            = args.rank
    cfg["resolution"]      = args.resolution
    cfg["seed"]            = args.seed
    cfg["checkpointing_steps"] = args.checkpointing_steps
    cfg["center_crop"]     = not args.disable_center_crop
    if args.validation_prompt:
        cfg["validation_prompt"] = args.validation_prompt
        cfg["num_validation_images"] = args.num_validation_images
        cfg["validation_epochs"] = args.validation_epochs
    elif args.enable_default_validation:
        cfg["validation_prompt"] = build_instance_prompt(class_name, "domain_a_pbc")
        cfg["num_validation_images"] = args.num_validation_images
        cfg["validation_epochs"] = args.validation_epochs

    out_dir = LORA_DIR / f"multidomain_{class_name}"

    # 1) 도메인별 이미지 샘플링
    domain_image_paths = sample_domain_images(
        DATA_DIR, class_name, DOMAINS, args.n_per_domain, args.seed
    )
    total_imgs = sum(len(p) for p in domain_image_paths.values())
    if total_imgs == 0:
        print(f"  [ERROR] 사용 가능한 이미지가 없음: {class_name}")
        return

    # 2) 혼합 디렉토리 준비
    mixed_dir, metadata = prepare_mixed_dataset_dir(
        class_name, domain_image_paths, TMP_ROOT
    )

    # 3) 학습 설정 저장
    save_training_config(class_name, out_dir, domain_image_paths, cfg)

    # 4) 쉘 스크립트 작성
    script_path = write_multidomain_shell_script(class_name, mixed_dir, out_dir, cfg)

    # 5) 학습 실행
    success = run_training(script_path, class_name, dry_run=args.dry_run)

    # 6) 임시 디렉토리 정리
    if not args.keep_tmp and not args.dry_run:
        if mixed_dir.exists():
            shutil.rmtree(mixed_dir)
            print(f"  임시 디렉토리 삭제: {mixed_dir}")

    if success and not args.dry_run:
        lora_weights = out_dir / "pytorch_lora_weights.safetensors"
        if lora_weights.exists():
            sz = lora_weights.stat().st_size / 1e6
            print(f"  LoRA 가중치: {lora_weights} ({sz:.1f} MB)")
        else:
            print(f"  [WARN] LoRA 가중치 파일을 찾을 수 없음: {lora_weights}")


# ── argparse ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Multi-domain LoRA training for WBC")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--class_name",  choices=CLASSES,
                       help="학습할 클래스 이름")
    group.add_argument("--all_classes", action="store_true",
                       help="모든 5클래스 순차 학습")
    p.add_argument("--steps",        type=int, default=400,
                   help="학습 스텝 수 (기본: 400 = 4도메인 × 100)")
    p.add_argument("--rank",         type=int, default=8)
    p.add_argument("--n_per_domain", type=int, default=200,
                   help="도메인당 샘플 수 (기본: 200)")
    p.add_argument("--resolution",   type=int, default=256)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--checkpointing_steps", type=int, default=100,
                   help="중간 체크포인트 저장 간격 (기본: 100)")
    p.add_argument("--disable_center_crop", action="store_true",
                   help="기본 center crop을 끄고 DreamBooth 기본 random crop 사용")
    p.add_argument("--validation_prompt", type=str, default=None,
                   help="중간 validation용 프롬프트. 지정 시 validation image 생성")
    p.add_argument("--enable_default_validation", action="store_true",
                   help="validation_prompt 미지정 시 기본 morphology prompt로 validation 활성화")
    p.add_argument("--num_validation_images", type=int, default=2,
                   help="validation 시 생성할 이미지 수 (기본: 2)")
    p.add_argument("--validation_epochs", type=int, default=1,
                   help="validation 주기 epoch (기본: 1)")
    p.add_argument("--dry_run",      action="store_true",
                   help="스크립트 생성만, 실제 학습 실행 안 함")
    p.add_argument("--keep_tmp",     action="store_true",
                   help="학습 후 임시 혼합 디렉토리 유지")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  Script 10 — Multi-Domain LoRA Training")
    print(f"  device={get_device()}, steps={args.steps}, rank={args.rank}")
    print(f"  n_per_domain={args.n_per_domain}, dry_run={args.dry_run}")
    print(f"{'='*60}")

    if args.all_classes:
        for cls in CLASSES:
            args_copy = argparse.Namespace(**vars(args))
            args_copy.class_name  = cls
            args_copy.all_classes = False
            train_class_multidomain(cls, args_copy)
    else:
        train_class_multidomain(args.class_name, args)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
