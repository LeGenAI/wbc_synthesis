"""
Multi-Domain SDXL Text-to-Image LoRA Fine-tuning
=================================================
DreamBooth(스크립트 10)와의 차이:
  - train_text_to_image_lora_sdxl.py 사용 (순수 T2I LoRA)
  - --instance_prompt / subject injection 없음
  - UNet LoRA weight를 text-image 쌍 지도학습으로 직접 업데이트
  - 각 이미지에 도메인별 캡션(metadata.jsonl)으로 스타일 조건 제공

학습 데이터:
  data/processed_multidomain/
  ├── domain_a_pbc/{class}/*.jpg
  ├── domain_b_raabin/{class}/*.jpg
  ├── domain_c_mll23/{class}/*.jpg
  └── domain_e_amc/{class}/*.jpg

출력:
  lora/weights/t2i_multidomain_{class_name}/pytorch_lora_weights.safetensors
  lora/scripts/train_t2i_multidomain_{class_name}.sh
  data/tmp_t2i_mixed/mixed_{class_name}/   (학습 후 삭제, --keep_tmp로 유지)

Usage:
    python scripts/legacy/phase_08_17_domain_gap_multidomain/11_multidomain_t2i_lora_train.py --class_name basophil --dry_run
    python scripts/legacy/phase_08_17_domain_gap_multidomain/11_multidomain_t2i_lora_train.py --class_name basophil
    python scripts/legacy/phase_08_17_domain_gap_multidomain/11_multidomain_t2i_lora_train.py --all_classes
"""

import argparse
import json
import os
import random
import shutil
import stat
import subprocess
import sys
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "processed_multidomain"
TMP_ROOT    = ROOT / "data" / "tmp_t2i_mixed"
LORA_DIR    = ROOT / "lora" / "weights"
SCRIPTS_DIR = ROOT / "lora" / "scripts"
LOG_DIR     = ROOT / "logs"

# train_text_to_image_lora_sdxl.py (이 파일과 같은 scripts/ 폴더)
T2I_SCRIPT  = Path(__file__).parent / "train_text_to_image_lora_sdxl.py"
BASE_MODEL  = "stabilityai/stable-diffusion-xl-base-1.0"

# ── 5개 대상 클래스 ────────────────────────────────────────────────────
TARGET_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]

# ── 4개 도메인 ─────────────────────────────────────────────────────────
DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]

# ── 도메인별 장비/염색 설명 (캡션에 삽입) ─────────────────────────────
DOMAIN_PROMPTS = {
    "domain_a_pbc":    "May-Grünwald Giemsa stain, CellaVision automated analyzer, Barcelona Spain",
    "domain_b_raabin": "Giemsa stain, smartphone microscope camera, Iran hospital",
    "domain_c_mll23":  "Pappenheim stain, Metafer scanner, Germany clinical lab",
    "domain_e_amc":    "Romanowsky stain, miLab automated analyzer, South Korea AMC",
}

# ── 클래스별 형태 설명 (캡션에 삽입) ─────────────────────────────────
CLASS_MORPHOLOGY = {
    "basophil":    "bilobed nucleus with dark purple-black granules covering nucleus",
    "eosinophil":  "bilobed nucleus with large orange-red eosinophilic granules",
    "lymphocyte":  "large round nucleus with scant pale-blue cytoplasm",
    "monocyte":    "kidney-shaped or horseshoe nucleus with grey-blue cytoplasm",
    "neutrophil":  "multilobed nucleus with pale pink cytoplasmic granules",
}

# ── LoRA 하이퍼파라미터 (MPS 최적화) ──────────────────────────────────
LORA_CONFIG = {
    "resolution":                   256,
    "train_batch_size":             1,
    "gradient_accumulation_steps":  2,       # effective batch = 2
    "learning_rate":                1e-4,    # T2I LoRA 표준 LR (DreamBooth 5e-5보다 높게)
    "lr_scheduler":                 "cosine",
    "lr_warmup_steps":              50,      # 전체 steps의 ~12.5%
    "max_train_steps":              400,     # 4도메인 × 100
    "rank":                         8,
    "mixed_precision":              "no",    # MPS fp16 backward 버그
    "gradient_checkpointing":       False,
    "train_text_encoder":           False,   # UNet LoRA only
    "enable_xformers":              False,   # CUDA only
    "seed":                         42,
    "checkpointing_steps":          9999,    # 중간 체크포인트 없음 (최종만)
    "random_flip":                  True,
    "center_crop":                  True,    # T2I 스크립트 기본 center_crop 활성화
}


# ── 유틸 ──────────────────────────────────────────────────────────────

def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*")
               if f.suffix.lower() in {".jpg", ".jpeg", ".png"})


def build_caption(class_name: str, domain: str) -> str:
    """각 이미지용 캡션 생성 (도메인 + 클래스 형태 정보 포함)."""
    morph = CLASS_MORPHOLOGY.get(class_name, class_name)
    domain_desc = DOMAIN_PROMPTS.get(domain, domain)
    return (
        f"microscopy image of {class_name} white blood cell, "
        f"{morph}, "
        f"{domain_desc}, "
        f"peripheral blood smear, clinical lab imaging"
    )


def sample_domain_images(
    class_name: str,
    n_per_domain: int,
    seed: int = 42,
) -> dict:
    """각 도메인에서 n_per_domain장 샘플링. 도메인 이미지 부족 시 전수 사용."""
    rng = random.Random(seed)
    domain_paths = {}
    for domain in DOMAINS:
        cls_dir = DATA_DIR / domain / class_name
        imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        if not imgs:
            print(f"  [WARN] {domain}/{class_name}: 이미지 없음, 스킵")
            continue
        if len(imgs) <= n_per_domain:
            print(f"  [INFO] {domain}/{class_name}: {len(imgs)}장 (요청 {n_per_domain}장 초과, 전수 사용)")
            domain_paths[domain] = imgs
        else:
            domain_paths[domain] = rng.sample(imgs, n_per_domain)
            print(f"  [INFO] {domain}/{class_name}: {len(imgs)}장 중 {n_per_domain}장 샘플링")
    return domain_paths


def prepare_mixed_dataset_dir(
    class_name: str,
    domain_image_paths: dict,
    out_root: Path,
) -> Path:
    """
    domain_image_paths의 이미지를 하나의 디렉토리로 합치고
    metadata.jsonl (ImageFolder 형식)을 생성한다.

    metadata.jsonl 형식:
      {"file_name": "domain_a_pbc_000001.jpg", "text": "microscopy image of ..."}
    """
    mixed_dir = out_root / f"mixed_{class_name}"
    mixed_dir.mkdir(parents=True, exist_ok=True)

    # 기존 metadata.jsonl 초기화
    meta_path = mixed_dir / "metadata.jsonl"
    records = []

    total = 0
    for domain, paths in domain_image_paths.items():
        caption = build_caption(class_name, domain)
        for i, src in enumerate(paths):
            # 파일명 충돌 방지: {domain}_{idx:06d}.jpg
            dst_name = f"{domain}_{i:06d}.jpg"
            dst = mixed_dir / dst_name
            shutil.copy2(src, dst)
            records.append({"file_name": dst_name, "text": caption})
            total += 1

    # metadata.jsonl 저장 (HuggingFace ImageFolder 형식)
    with open(meta_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  → mixed_dir: {mixed_dir}  ({total}장, {len(domain_image_paths)}개 도메인)")
    print(f"  → metadata.jsonl: {len(records)}줄")
    return mixed_dir


def write_t2i_shell_script(
    class_name: str,
    mixed_dir: Path,
    out_dir: Path,
    lora_config: dict,
) -> Path:
    """
    train_text_to_image_lora_sdxl.py 호출 쉘 스크립트 생성.
    DreamBooth와 달리:
      - --train_data_dir 사용 (--instance_data_dir 없음)
      - --instance_prompt 없음
      - --caption_column text (metadata.jsonl의 "text" 컬럼)
    """
    python_bin = sys.executable
    steps = lora_config["max_train_steps"]

    lines = [
        "#!/bin/bash",
        f"# Auto-generated T2I LoRA script: {class_name} (multidomain)",
        "set -euo pipefail",
        "",
        f"PYTHON={python_bin}",
        f'SCRIPT="{T2I_SCRIPT}"',
        "",
        f'"$PYTHON" "$SCRIPT" \\',
        f'  --pretrained_model_name_or_path "{BASE_MODEL}" \\',
        f'  --train_data_dir "{mixed_dir}" \\',
        f'  --caption_column "text" \\',
        f'  --output_dir "{out_dir}" \\',
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
        f'  --report_to "tensorboard" \\',
    ]
    if lora_config["center_crop"]:
        lines.append("  --center_crop \\")
    if lora_config["train_text_encoder"]:
        lines.append("  --train_text_encoder \\")
    if lora_config["gradient_checkpointing"]:
        lines.append("  --gradient_checkpointing \\")
    if lora_config["enable_xformers"]:
        lines.append("  --enable_xformers_memory_efficient_attention \\")
    if lora_config["random_flip"]:
        lines.append("  --random_flip \\")

    # 마지막 줄 backslash 제거
    lines[-1] = lines[-1].rstrip(" \\")

    script_path = SCRIPTS_DIR / f"train_t2i_multidomain_{class_name}.sh"
    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    print(f"  → 쉘 스크립트: {script_path}")
    return script_path


def save_training_config(
    class_name: str,
    out_dir: Path,
    domain_image_paths: dict,
    lora_config: dict,
) -> None:
    """학습 메타데이터를 JSON으로 저장."""
    cfg = {
        **lora_config,
        "class": class_name,
        "training_type": "text_to_image_lora",
        "base_model": BASE_MODEL,
        "domains": {
            domain: len(paths)
            for domain, paths in domain_image_paths.items()
        },
        "total_images": sum(len(p) for p in domain_image_paths.values()),
        "captions": {
            domain: build_caption(class_name, domain)
            for domain in domain_image_paths
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "lora_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  → 학습 설정 저장: {cfg_path}")


def train_class_t2i(class_name: str, args: argparse.Namespace) -> None:
    """단일 클래스 T2I LoRA 학습 전체 파이프라인."""
    print(f"\n{'='*65}")
    print(f"  T2I LoRA Fine-tuning: {class_name.upper()}  (multidomain)")
    print(f"{'='*65}")

    # 하이퍼파라미터 오버라이드
    lora_config = dict(LORA_CONFIG)
    if args.steps:
        lora_config["max_train_steps"] = args.steps
    if args.rank:
        lora_config["rank"] = args.rank
    if args.resolution:
        lora_config["resolution"] = args.resolution
    if args.seed:
        lora_config["seed"] = args.seed

    # 1) 도메인별 이미지 샘플링
    print("\n[1/5] 도메인별 이미지 샘플링")
    domain_image_paths = sample_domain_images(class_name, args.n_per_domain, seed=lora_config["seed"])
    if not domain_image_paths:
        print(f"[ERROR] {class_name}: 유효한 도메인 데이터 없음, 스킵")
        return

    # 2) Mixed dataset 디렉토리 생성
    print("\n[2/5] Mixed dataset 디렉토리 준비")
    mixed_dir = prepare_mixed_dataset_dir(class_name, domain_image_paths, TMP_ROOT)

    # 3) 출력 디렉토리 설정
    out_dir = LORA_DIR / f"t2i_multidomain_{class_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) 쉘 스크립트 생성
    print("\n[3/5] 쉘 스크립트 생성")
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    script_path = write_t2i_shell_script(class_name, mixed_dir, out_dir, lora_config)

    # 5) 설정 저장
    print("\n[4/5] 학습 설정 저장")
    save_training_config(class_name, out_dir, domain_image_paths, lora_config)

    if args.dry_run:
        print("\n[DRY RUN] 스크립트 내용:")
        print("-" * 50)
        print(open(script_path).read())
        print("-" * 50)
        print("[DRY RUN] 학습 실행하지 않음.")
        return

    # 6) 학습 실행
    total_imgs = sum(len(p) for p in domain_image_paths.values())
    steps = lora_config["max_train_steps"]
    est_h = steps * 180 / 3600
    print(f"\n[5/5] 학습 시작")
    print(f"  이미지: {total_imgs}장 ({len(domain_image_paths)}개 도메인)")
    print(f"  Steps:  {steps}  (est. {est_h:.1f}h on MPS @ 256px)")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"t2i_lora_{class_name}.log"
    print(f"  로그:   {log_file}")

    with open(log_file, "w") as lf:
        result = subprocess.run(
            ["/bin/bash", str(script_path)],
            stdout=lf,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        print(f"[ERROR] 학습 실패 (exit={result.returncode}). 로그 확인: {log_file}")
        # 로그 마지막 30줄 출력
        lines = open(log_file).readlines()
        print("  --- 로그 끝 30줄 ---")
        for l in lines[-30:]:
            print("  " + l.rstrip())
    else:
        lora_file = out_dir / "pytorch_lora_weights.safetensors"
        if lora_file.exists():
            size_mb = lora_file.stat().st_size / 1e6
            print(f"[OK] 학습 완료: {class_name}  (LoRA: {size_mb:.1f} MB)")
        else:
            print(f"[WARN] 학습은 완료됐지만 safetensors 파일 없음. 로그 확인: {log_file}")

    # 임시 디렉토리 정리
    if not args.keep_tmp:
        shutil.rmtree(mixed_dir, ignore_errors=True)
        print(f"  [정리] tmp 삭제: {mixed_dir}")
    else:
        print(f"  [유지] tmp 유지: {mixed_dir}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-domain SDXL T2I LoRA fine-tuning (train_text_to_image_lora_sdxl.py 사용)"
    )
    parser.add_argument("--class_name", type=str, choices=TARGET_CLASSES,
                        help="학습할 단일 클래스명")
    parser.add_argument("--all_classes", action="store_true",
                        help="5개 클래스 순차 학습")
    parser.add_argument("--steps", type=int, default=400,
                        help="학습 스텝 수 (기본: 400)")
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank (기본: 8)")
    parser.add_argument("--n_per_domain", type=int, default=200,
                        help="도메인당 샘플 수 (기본: 200)")
    parser.add_argument("--resolution", type=int, default=256,
                        help="학습 해상도 (기본: 256)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="쉘 스크립트 생성만, 학습 실행 안 함")
    parser.add_argument("--keep_tmp", action="store_true",
                        help="학습 후 임시 mixed 디렉토리 유지")

    args = parser.parse_args()

    if not T2I_SCRIPT.exists():
        print(f"[ERROR] T2I 학습 스크립트 없음: {T2I_SCRIPT}")
        print("  다운로드: curl -sO https://raw.githubusercontent.com/huggingface/"
              "diffusers/main/examples/text_to_image/train_text_to_image_lora_sdxl.py")
        sys.exit(1)

    if args.all_classes:
        print(f"T2I LoRA 멀티도메인 학습: {len(TARGET_CLASSES)}개 클래스 순차 실행")
        for cls in TARGET_CLASSES:
            train_class_t2i(cls, args)
    elif args.class_name:
        train_class_t2i(args.class_name, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
