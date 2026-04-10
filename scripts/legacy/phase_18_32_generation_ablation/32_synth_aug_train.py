"""
Script 32: 합성 데이터 증강 VGG16 재학습 실험
================================================
Script 31에서 생성한 합성 이미지 900장을 VGG16 훈련 세트에 추가해
val macro-F1 개선폭을 측정한다.

비교 조건 (3가지):
  - real_only          : 실제 이미지 51,064장 (Script 30 결과 재사용)
  - real+synth         : 실제 + 합성 900장
  - real+synth_filtered: 실제 + 합성 필터링 788장 (VGG16 correct==True)

핵심 설계:
  - val set = 실제 이미지 9,012장 고정 (seed=42, 합성 누출 없음)
  - WeightedRandomSampler: 합성 가중치 = 1/real_combo_count (과대표현 방지)
  - num_samples = n_train (51,064) → epoch 길이 고정
  - VGG16 features 동결 + classifier만 학습 (Script 30과 동일)
  - 클래스별 F1 출력 (취약 클래스: monocyte, eosinophil 모니터링)

Usage:
    python3 scripts/legacy/phase_18_32_generation_ablation/32_synth_aug_train.py --dry_run          # 구조 확인
    python3 scripts/legacy/phase_18_32_generation_ablation/32_synth_aug_train.py --skip_real_only   # ~62분 (권장)
    python3 scripts/legacy/phase_18_32_generation_ablation/32_synth_aug_train.py                    # ~92분 (real_only 재학습 포함)
    python3 scripts/legacy/phase_18_32_generation_ablation/32_synth_aug_train.py --synth_mode all --skip_real_only  # 단일 조건

출력:
    models/vgg16_synth_aug_all.pt
    models/vgg16_synth_aug_filtered.pt
    results/synth_aug/summary.json
    results/synth_aug/report.md
"""

import argparse
import json
import random
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import (DataLoader, Dataset, Subset, WeightedRandomSampler,
                               random_split)
from torchvision import models, transforms
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "processed_multidomain"
SYNTH_DIR   = ROOT / "results" / "prompt_diversity" / "images"
SYNTH_SUMMARY = ROOT / "results" / "prompt_diversity" / "summary.json"
VGG16_CKPT  = ROOT / "models" / "multidomain_cnn_vgg16.pt"
OUT_DIR     = ROOT / "models"
LOG_DIR     = ROOT / "results" / "synth_aug"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── 메타데이터 ──────────────────────────────────────────────────────────
DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_LABELS = {
    "domain_a_pbc":    "PBC (Spain)",
    "domain_b_raabin": "Raabin (Iran)",
    "domain_c_mll23":  "MLL23 (Germany)",
    "domain_e_amc":    "AMC (Korea)",
}
MULTI_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_IDX     = {c: i for i, c in enumerate(MULTI_CLASSES)}

# 합성 이미지 도메인 매핑 (summary.json dom 필드 → domain_idx)
SYNTH_DOMAIN_MAP = {"PBC": 0, "Raabin": 1, "MLL23": 2, "AMC": 3}

IMG_EXTS    = {".jpg", ".jpeg", ".png"}
IMG_SIZE    = 224
NUM_WORKERS = 0  # macOS MPS 안전


# ── 디바이스 & 재현성 ──────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── MultiDomainDataset (Script 30 그대로) ─────────────────────────────
class MultiDomainDataset(Dataset):
    """4도메인 × 5클래스 이미지 수집. (PIL Image, class_idx) 반환."""

    def __init__(self, data_dir: Path, transform=None, seed: int = 42):
        self.transform = transform
        self.samples: list[tuple] = []  # (path, class_idx, domain_idx)

        for domain in DOMAINS:
            d_idx = DOMAINS.index(domain)
            for cls in MULTI_CLASSES:
                c_idx   = CLASS_IDX[cls]
                cls_dir = data_dir / domain / cls
                if not cls_dir.exists():
                    warnings.warn(f"[WARN] 없음: {cls_dir}")
                    continue
                paths = [p for p in cls_dir.iterdir()
                         if p.suffix.lower() in IMG_EXTS]
                self.samples.extend((p, c_idx, d_idx) for p in paths)

        print(f"  MultiDomainDataset: {len(self.samples)}장")

    def get_sample_weights(self) -> list:
        combo_counts = Counter((c, d) for (_, c, d) in self.samples)
        return [1.0 / combo_counts[(c, d)] for (_, c, d) in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, c_idx, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, c_idx


# ── 합성 이미지 로더 ────────────────────────────────────────────────────
def load_synth_samples(mode: str) -> list[tuple]:
    """
    summary.json에서 (Path, class_idx, domain_idx) 리스트를 반환.

    mode:
      "all"      : 900장 전체
      "filtered" : correct==True인 이미지만 (≈788장)
    """
    with open(SYNTH_SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)

    samples = []
    missing = 0

    for combo in summary["results"]:
        cls = combo["cls"]
        dom = combo["dom"]
        c_idx = CLASS_IDX[cls]
        d_idx = SYNTH_DOMAIN_MAP[dom]

        for inp in combo["inputs"]:
            inp_idx = inp["inp_idx"]

            for cond in ["A", "B", "C"]:
                seed_key = f"seeds_{cond}"
                if seed_key not in inp:
                    continue
                for sd in inp[seed_key]:
                    if mode == "filtered" and not sd.get("correct", False):
                        continue
                    so = sd["seed_offset"]
                    img_path = (SYNTH_DIR / cls / dom
                                / f"inp_{inp_idx:02d}"
                                / f"cond_{cond}_seed_{so:02d}.png")
                    if img_path.exists():
                        samples.append((img_path, c_idx, d_idx))
                    else:
                        missing += 1

    if missing:
        warnings.warn(f"[WARN] 합성 이미지 {missing}장 파일 없음 (경로 불일치)")

    return samples


# ── 합성+실제 통합 데이터셋 ────────────────────────────────────────────
class _CombinedTrainDataset(Dataset):
    """
    실제 이미지 Subset + 합성 이미지 리스트를 통합.
    transform은 항상 동일한 train_transform 사용.
    """

    def __init__(self, real_sub: Subset, synth_samples: list, transform):
        self.real_sub      = real_sub
        self.synth_samples = synth_samples
        self.transform     = transform

    def __len__(self):
        return len(self.real_sub) + len(self.synth_samples)

    def __getitem__(self, idx):
        if idx < len(self.real_sub):
            img, c_idx = self.real_sub[idx]  # PIL Image (transform 없는 원시)
        else:
            path, c_idx, _ = self.synth_samples[idx - len(self.real_sub)]
            img = Image.open(path).convert("RGB")
        return self.transform(img), c_idx


# ── Transform ──────────────────────────────────────────────────────────
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── 모델 빌드 (Script 30과 동일) ───────────────────────────────────────
def build_model(n_classes: int = 5, full_finetune: bool = False) -> nn.Module:
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, n_classes)
    if not full_finetune:
        for param in m.features.parameters():
            param.requires_grad = False
    return m


# ── 학습 한 epoch ──────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in tqdm(loader, desc="  train", ncols=70, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs

    return {"loss": total_loss / total, "acc": correct / total}


# ── 검증 (클래스별 F1 포함) ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_with_classwise(model, loader, criterion, device) -> dict:
    """Script 30의 evaluate() + classification_report로 클래스별 F1 추가."""
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  val ", ncols=70, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits       = model(imgs)
        loss         = criterion(logits, labels)

        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += bs
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report   = classification_report(
        all_labels, all_preds,
        target_names=MULTI_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    class_f1  = {cls: report[cls]["f1-score"]  for cls in MULTI_CLASSES}
    class_acc = {cls: report[cls]["precision"]  for cls in MULTI_CLASSES}
    # precision ≈ per-class 정확도(macro 기준)는 아님; recall이 더 적합
    class_recall = {cls: report[cls]["recall"] for cls in MULTI_CLASSES}

    return {
        "loss":     total_loss / total,
        "acc":      correct / total,
        "macro_f1": macro_f1,
        "class_f1": class_f1,
        "class_recall": class_recall,  # = per-class accuracy for single-label
    }


# ── 조건별 학습 함수 ────────────────────────────────────────────────────
def run_condition(
    synth_mode: str,
    args,
    device: torch.device,
    val_sub: Subset,
    full_real_ds: MultiDomainDataset,
    train_sub: Subset,
    n_train: int,
) -> dict:
    """
    synth_mode: "none" | "all" | "filtered"
    VGG16 fresh start, args.epochs epoch 학습.
    returns: {val_f1, val_acc, class_f1, class_recall, n_synth, ckpt_path, log}
    """
    label = {"none": "real_only", "all": "real+synth", "filtered": "real+synth_filtered"}[synth_mode]
    print(f"\n{'─'*60}")
    print(f"  조건: {label}")

    # ── 합성 이미지 로드 ──────────────────────────────────────────────
    synth_samples = []
    if synth_mode != "none":
        synth_samples = load_synth_samples(synth_mode)
        print(f"  합성 이미지: {len(synth_samples)}장 (mode={synth_mode})")

    # ── WeightedRandomSampler ─────────────────────────────────────────
    # 실제 이미지 가중치 (Script 30과 동일 스케일)
    all_weights   = full_real_ds.get_sample_weights()
    train_weights = [all_weights[i] for i in train_sub.indices]

    if synth_samples:
        # 합성 이미지 가중치: 해당 (domain, class) 조합의 실제 이미지 count 기준
        real_combo_counts = Counter(
            (full_real_ds.samples[i][2], full_real_ds.samples[i][1])
            for i in train_sub.indices
        )
        synth_weights = [
            1.0 / max(real_combo_counts.get((d, c), 1), 1)
            for (_, c, d) in synth_samples
        ]
        combined_weights = train_weights + synth_weights
    else:
        combined_weights = train_weights

    sampler = WeightedRandomSampler(
        combined_weights,
        num_samples=n_train,          # epoch 길이 = 실제 기준 고정
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    # ── 데이터로더 ───────────────────────────────────────────────────
    train_ds = _CombinedTrainDataset(train_sub, synth_samples, get_train_transform())
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    class _ValWrapper(Dataset):
        def __init__(self, ds, tf):
            self.ds = ds; self.tf = tf
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            img, c = self.ds[i]; return self.tf(img), c

    val_loader = DataLoader(
        _ValWrapper(val_sub, get_val_transform()),
        batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    total_train = len(train_ds)
    print(f"  train dataset: {total_train}장 "
          f"(실제 {n_train}장 + 합성 {len(synth_samples)}장), "
          f"epoch당 {n_train} samples (sampler)")

    # ── 모델 초기화 ──────────────────────────────────────────────────
    model = build_model(n_classes=len(MULTI_CLASSES),
                        full_finetune=args.full_finetune).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1
    )

    # ── 학습 루프 ─────────────────────────────────────────────────────
    if synth_mode == "none":
        ckpt_name = "vgg16_synth_aug_none.pt"
    elif synth_mode == "all":
        ckpt_name = "vgg16_synth_aug_all.pt"
    else:
        ckpt_name = "vgg16_synth_aug_filtered.pt"
    ckpt_out = OUT_DIR / ckpt_name

    best_f1   = 0.0
    best_result = {}
    log       = []
    t0        = time.time()

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = evaluate_with_classwise(model, val_loader, criterion, device)
        scheduler.step(epoch)

        entry = {
            "epoch": epoch,
            "lr":    optimizer.param_groups[0]["lr"],
            "train": tr,
            "val":   {k: v for k, v in vl.items()},
        }
        log.append(entry)

        flag = ""
        if vl["macro_f1"] > best_f1:
            best_f1 = vl["macro_f1"]
            best_result = vl.copy()
            torch.save({
                "epoch":         epoch,
                "model_name":    "vgg16",
                "synth_mode":    synth_mode,
                "n_synth":       len(synth_samples),
                "class_names":   MULTI_CLASSES,
                "model_state_dict": model.state_dict(),
                "val_f1":        vl["macro_f1"],
                "val_acc":       vl["acc"],
            }, ckpt_out)
            flag = f" ✅ best (F1={best_f1:.4f})"

        elapsed = (time.time() - t0) / 60
        print(f"  Epoch {epoch:02d}/{args.epochs} | "
              f"loss={vl['loss']:.4f} | acc={vl['acc']*100:.1f}% | "
              f"F1={vl['macro_f1']:.4f}{flag} | {elapsed:.1f}min")

    elapsed_total = (time.time() - t0) / 60
    print(f"\n  [{label}] best F1={best_f1:.4f}, 소요={elapsed_total:.1f}분")
    print("  클래스별 recall (best checkpoint 기준):")
    for cls in MULTI_CLASSES:
        r = best_result.get("class_recall", {}).get(cls, 0)
        f = best_result.get("class_f1", {}).get(cls, 0)
        mark = "🟩" if r >= 0.90 else ("🟨" if r >= 0.67 else "🟥")
        print(f"    {mark} {cls:<12}: recall={r*100:.1f}%  F1={f:.4f}")

    return {
        "val_f1":       best_f1,
        "val_acc":      best_result.get("acc", 0),
        "class_f1":     best_result.get("class_f1", {}),
        "class_recall": best_result.get("class_recall", {}),
        "n_synth":      len(synth_samples),
        "ckpt_path":    str(ckpt_out),
        "log":          log,
        "elapsed_min":  elapsed_total,
    }


# ── real_only 재사용 (--skip_real_only) ────────────────────────────────
def load_real_only_from_script30(
    args,
    device: torch.device,
    val_sub: Subset,
) -> dict:
    """
    Script 30의 `multidomain_cnn_vgg16.pt` 로드 → val_sub에서 재평가.
    도메인별 결과는 train_log.json에서 재사용.
    """
    print("\n  [real_only] Script 30 결과 재사용 (--skip_real_only)")
    if not VGG16_CKPT.exists():
        raise FileNotFoundError(f"VGG16 체크포인트 없음: {VGG16_CKPT}\n"
                                f"  → --skip_real_only 없이 실행하거나 Script 30을 먼저 실행하세요.")

    # 모델 로드
    ckpt = torch.load(VGG16_CKPT, map_location=device, weights_only=False)
    model = build_model(n_classes=len(MULTI_CLASSES), full_finetune=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # val 재평가
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    class _ValWrapper(Dataset):
        def __init__(self, ds, tf):
            self.ds = ds; self.tf = tf
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            img, c = self.ds[i]; return self.tf(img), c

    val_loader = DataLoader(
        _ValWrapper(val_sub, get_val_transform()),
        batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    vl = evaluate_with_classwise(model, val_loader, criterion, device)
    print(f"  [real_only] val F1={vl['macro_f1']:.4f}, acc={vl['acc']*100:.1f}%")
    print("  클래스별 recall:")
    for cls in MULTI_CLASSES:
        r = vl["class_recall"].get(cls, 0)
        f = vl["class_f1"].get(cls, 0)
        mark = "🟩" if r >= 0.90 else ("🟨" if r >= 0.67 else "🟥")
        print(f"    {mark} {cls:<12}: recall={r*100:.1f}%  F1={f:.4f}")

    # Script 30 train_log.json에서 도메인별 결과 로드
    train_log_path = ROOT / "results" / "vgg16_cnn_train" / "train_log.json"
    domain_results_script30 = {}
    if train_log_path.exists():
        with open(train_log_path, encoding="utf-8") as f:
            log_data = json.load(f)
        domain_results_script30 = log_data.get("domain_results", {})
        print(f"  Script 30 domain_results 로드: {list(domain_results_script30.keys())}")

    return {
        "val_f1":              vl["macro_f1"],
        "val_acc":             vl["acc"],
        "class_f1":            vl["class_f1"],
        "class_recall":        vl["class_recall"],
        "n_synth":             0,
        "ckpt_path":           str(VGG16_CKPT),
        "domain_results":      domain_results_script30,
        "skipped_training":    True,
    }


# ── 리포트 생성 ────────────────────────────────────────────────────────
def make_report(cond_results: dict, baseline_f1: float) -> str:
    lines = [
        "# Script 32 — 합성 데이터 증강 VGG16 실험 리포트",
        "",
        "**목적**: 합성 이미지(900장)를 훈련 세트에 추가했을 때 VGG16 val F1 개선폭 측정",
        "",
        "---",
        "",
        "## 1. 전체 성능 비교",
        "",
        "| 조건 | n_synth | val F1 | Δ F1 | val Acc |",
        "|------|:-------:|:------:|:----:|:-------:|",
    ]

    order = ["real_only", "real+synth", "real+synth_filtered"]
    for key in order:
        if key not in cond_results:
            continue
        r    = cond_results[key]
        f1   = r["val_f1"]
        acc  = r["val_acc"]
        n    = r["n_synth"]
        delta = f1 - baseline_f1
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        mark  = "✅" if delta > 0.001 else ("➖" if abs(delta) <= 0.001 else "❌")
        lines.append(f"| **{key}** | {n} | {f1:.4f} | {mark} {delta_str} | {acc*100:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## 2. 클래스별 recall (≈ per-class accuracy)",
        "",
        "| 클래스 | real_only | real+synth | Δ | real+synth_filtered | Δ |",
        "|--------|:---------:|:----------:|:-:|:-------------------:|:-:|",
    ]

    def badge(r):
        return "🟩" if r >= 0.90 else ("🟨" if r >= 0.67 else "🟥")

    for cls in MULTI_CLASSES:
        r0 = cond_results.get("real_only", {}).get("class_recall", {}).get(cls, 0)
        r1 = cond_results.get("real+synth", {}).get("class_recall", {}).get(cls, 0)
        r2 = cond_results.get("real+synth_filtered", {}).get("class_recall", {}).get(cls, 0)
        d1 = r1 - r0; d1_str = f"+{d1*100:.1f}%p" if d1 >= 0 else f"{d1*100:.1f}%p"
        d2 = r2 - r0; d2_str = f"+{d2*100:.1f}%p" if d2 >= 0 else f"{d2*100:.1f}%p"
        lines.append(
            f"| {cls} | {badge(r0)}{r0*100:.0f}% | "
            f"{badge(r1)}{r1*100:.0f}% | {d1_str} | "
            f"{badge(r2)}{r2*100:.0f}% | {d2_str} |"
        )

    lines += [
        "",
        "> 🟩 ≥90%: 양호 | 🟨 67~90%: 주의 | 🟥 <67%: 불량",
        "",
        "---",
        "",
        "## 3. 클래스별 F1",
        "",
        "| 클래스 | real_only | real+synth | Δ | real+synth_filtered | Δ |",
        "|--------|:---------:|:----------:|:-:|:-------------------:|:-:|",
    ]

    for cls in MULTI_CLASSES:
        f0 = cond_results.get("real_only", {}).get("class_f1", {}).get(cls, 0)
        f1 = cond_results.get("real+synth", {}).get("class_f1", {}).get(cls, 0)
        f2 = cond_results.get("real+synth_filtered", {}).get("class_f1", {}).get(cls, 0)
        d1 = f1 - f0; d1_str = f"+{d1:.4f}" if d1 >= 0 else f"{d1:.4f}"
        d2 = f2 - f0; d2_str = f"+{d2:.4f}" if d2 >= 0 else f"{d2:.4f}"
        lines.append(
            f"| {cls} | {f0:.4f} | {f1:.4f} | {d1_str} | {f2:.4f} | {d2_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. 결론",
        "",
    ]

    # 자동 결론 생성
    r1 = cond_results.get("real+synth", {})
    r2 = cond_results.get("real+synth_filtered", {})

    if r1 and r2:
        d1 = r1["val_f1"] - baseline_f1
        d2 = r2["val_f1"] - baseline_f1
        best_key = "real+synth_filtered" if r2["val_f1"] >= r1["val_f1"] else "real+synth"
        best_d   = max(d1, d2)

        if best_d > 0.003:
            lines.append(f"- ✅ **합성 데이터 증강 효과 있음**: {best_key}이 베이스라인 대비 F1 +{best_d:.4f} 개선")
        elif best_d > 0:
            lines.append(f"- ➖ **합성 데이터 증강 효과 미미**: F1 +{best_d:.4f} (0.003 미만)")
        else:
            lines.append(f"- ❌ **합성 데이터 증강 효과 없음**: F1 {best_d:.4f} (개선 없음)")

        # 취약 클래스 모니터링
        for cls in ["monocyte", "eosinophil"]:
            r0_cls = cond_results.get("real_only", {}).get("class_recall", {}).get(cls, 0)
            rb_cls = cond_results.get(best_key, {}).get("class_recall", {}).get(cls, 0)
            d = rb_cls - r0_cls
            d_str = f"+{d*100:.1f}%p" if d >= 0 else f"{d*100:.1f}%p"
            lines.append(f"- {cls}: {r0_cls*100:.0f}% → {rb_cls*100:.0f}% ({d_str})")

        if r2["val_f1"] > r1["val_f1"]:
            lines.append(f"- 필터링 효과: real+synth_filtered > real+synth "
                         f"(F1 +{(r2['val_f1']-r1['val_f1']):.4f})")
        else:
            lines.append(f"- 필터링 효과: real+synth_filtered ≤ real+synth (필터링 불필요)")

    lines += ["", "---", f"", f"*생성: Script 32*"]
    return "\n".join(lines)


# ── dry_run ────────────────────────────────────────────────────────────
def dry_run(args):
    print("\n[DRY RUN] 데이터 구조 확인...")
    print(f"\n  실제 이미지 디렉토리: {DATA_DIR}")

    # 합성 이미지 확인
    print(f"\n  합성 이미지 summary: {SYNTH_SUMMARY}")
    synth_all = load_synth_samples("all")
    synth_flt = load_synth_samples("filtered")
    print(f"  mode=all      : {len(synth_all)}장")
    print(f"  mode=filtered : {len(synth_flt)}장")

    # 클래스별 분포
    cls_counter_all = Counter(MULTI_CLASSES[c] for (_, c, _) in synth_all)
    cls_counter_flt = Counter(MULTI_CLASSES[c] for (_, c, _) in synth_flt)
    print("\n  클래스별 합성 이미지 수 (all | filtered):")
    for cls in MULTI_CLASSES:
        print(f"    {cls:<12}: {cls_counter_all.get(cls, 0):4d} | "
              f"{cls_counter_flt.get(cls, 0):4d}")

    # 실제 이미지 확인
    full_ds = MultiDomainDataset(DATA_DIR, transform=None, seed=args.seed)
    n_val   = int(len(full_ds) * args.val_ratio)
    n_train = len(full_ds) - n_val
    print(f"\n  실제 이미지 split: train={n_train}, val={n_val}")

    # 모델 확인
    print("\n  VGG16 모델 초기화 테스트...")
    device = get_device()
    model = build_model(n_classes=5, full_finetune=False).to(device)
    dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"  forward OK: input=(2,3,224,224) → output={tuple(out.shape)}")

    # Sampler 확인 (작은 스케일)
    print("\n  WeightedRandomSampler 구성 확인...")
    gen    = torch.Generator().manual_seed(args.seed)
    train_sub, val_sub = random_split(full_ds, [n_train, n_val], generator=gen)
    all_weights   = full_ds.get_sample_weights()
    train_weights = [all_weights[i] for i in train_sub.indices]
    real_combo_counts = Counter(
        (full_ds.samples[i][2], full_ds.samples[i][1]) for i in train_sub.indices
    )
    synth_weights = [
        1.0 / max(real_combo_counts.get((d, c), 1), 1)
        for (_, c, d) in synth_all
    ]
    total_w    = sum(train_weights) + sum(synth_weights)
    synth_frac = sum(synth_weights) / total_w * 100
    print(f"  실제 이미지 가중치 합: {sum(train_weights):.2f}")
    print(f"  합성 이미지 가중치 합: {sum(synth_weights):.2f}")
    print(f"  합성 가중치 비율: {synth_frac:.2f}% (과대표현 없음 목표: <5%)")

    print("\n  [DRY RUN 완료] 모든 구조 정상. 실제 학습을 위해 --dry_run 없이 실행하세요.")


# ── argparse ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Script 32: 합성 데이터 증강 VGG16 재학습 (3조건 비교)"
    )
    p.add_argument("--synth_mode",
                   choices=["none", "all", "filtered", "both"],
                   default="both",
                   help="합성 데이터 조건 (both=all+filtered 순서로 실행)")
    p.add_argument("--skip_real_only",
                   action="store_true",
                   help="Script 30의 real_only 결과 재사용 (학습 생략, ~30분 절감)")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--val_ratio",    type=float, default=0.15)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--full_finetune", action="store_true",
                   help="VGG16 전체 파라미터 학습 (기본: features 동결)")
    p.add_argument("--dry_run",      action="store_true",
                   help="데이터 구조 확인만 (학습 없음)")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device()
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Script 32 — 합성 데이터 증강 VGG16 재학습")
    print(f"  device={device}, epochs={args.epochs}, lr={args.lr}")
    print(f"  batch={args.batch_size}, label_smooth={args.label_smooth}")
    print(f"  synth_mode={args.synth_mode}, skip_real_only={args.skip_real_only}")
    print(f"{'='*60}")

    if args.dry_run:
        dry_run(args)
        return

    # ── [1/5] 실제 이미지 데이터셋 준비 (한 번만) ─────────────────────
    print("\n[1/5] 실제 이미지 데이터셋 준비...")
    full_real_ds = MultiDomainDataset(DATA_DIR, transform=None, seed=args.seed)
    n_val        = int(len(full_real_ds) * args.val_ratio)
    n_train      = len(full_real_ds) - n_val
    gen          = torch.Generator().manual_seed(args.seed)
    train_sub, val_sub = random_split(full_real_ds, [n_train, n_val], generator=gen)
    print(f"  train: {n_train}장, val: {n_val}장 (seed={args.seed} 고정)")

    # ── [2/5] 합성 이미지 로드 (all + filtered) ──────────────────────
    print("\n[2/5] 합성 이미지 확인...")
    synth_counts = {}
    for mode in ["all", "filtered"]:
        s = load_synth_samples(mode)
        synth_counts[mode] = len(s)
        cls_dist = Counter(MULTI_CLASSES[c] for (_, c, _) in s)
        print(f"  mode={mode}: {len(s)}장 | " +
              " ".join(f"{cls[:4]}={n}" for cls, n in cls_dist.items()))

    # ── [3/5] 조건별 학습 ──────────────────────────────────────────────
    print("\n[3/5] 조건별 학습...")
    cond_results = {}
    t_total = time.time()

    # 결정할 조건 목록
    if args.synth_mode == "both":
        synth_modes = ["all", "filtered"]
    elif args.synth_mode == "none":
        synth_modes = []
    else:
        synth_modes = [args.synth_mode]

    # real_only
    if args.skip_real_only:
        result_ro = load_real_only_from_script30(args, device, val_sub)
    else:
        if args.synth_mode in ("none", "both"):
            result_ro = run_condition(
                "none", args, device, val_sub, full_real_ds, train_sub, n_train
            )
        else:
            # 단일 synth 조건 실행 시에도 real_only 재사용 또는 skip
            result_ro = load_real_only_from_script30(args, device, val_sub)
    cond_results["real_only"] = result_ro

    # synth 조건들
    for mode in synth_modes:
        key = "real+synth" if mode == "all" else "real+synth_filtered"
        result = run_condition(
            mode, args, device, val_sub, full_real_ds, train_sub, n_train
        )
        cond_results[key] = result

    elapsed_total = (time.time() - t_total) / 60
    print(f"\n  전체 실험 소요: {elapsed_total:.1f}분")

    # ── [4/5] 콘솔 비교표 출력 ─────────────────────────────────────────
    print(f"\n[4/5] 비교 결과:")
    print(f"\n  {'조건':<25} {'n_synth':>8} {'val F1':>8} {'Δ F1':>8} {'val Acc':>8}")
    print(f"  {'─'*58}")
    baseline_f1 = cond_results["real_only"]["val_f1"]
    for key, r in cond_results.items():
        delta = r["val_f1"] - baseline_f1
        d_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"  {key:<25} {r['n_synth']:>8} "
              f"{r['val_f1']:>8.4f} {d_str:>8} {r['val_acc']*100:>7.1f}%")

    print(f"\n  클래스별 recall 비교 (취약 클래스 강조):")
    print(f"  {'클래스':<13} {'real_only':>10} {'real+synth':>11} {'Δ':>8} "
          f"{'real+synth_flt':>15} {'Δ':>8}")
    print(f"  {'─'*66}")
    for cls in MULTI_CLASSES:
        r0 = cond_results.get("real_only", {}).get("class_recall", {}).get(cls, 0)
        r1 = cond_results.get("real+synth", {}).get("class_recall", {}).get(cls, 0)
        r2 = cond_results.get("real+synth_filtered", {}).get("class_recall", {}).get(cls, 0)
        d1 = r1 - r0; d1s = f"+{d1*100:.1f}%p" if d1 >= 0 else f"{d1*100:.1f}%p"
        d2 = r2 - r0; d2s = f"+{d2*100:.1f}%p" if d2 >= 0 else f"{d2*100:.1f}%p"
        warn = " ⚠️" if r0 < 0.85 else ""
        print(f"  {cls:<13}{warn} {r0*100:>9.1f}% {r1*100:>10.1f}% {d1s:>8} "
              f"{r2*100:>14.1f}% {d2s:>8}")

    # ── [5/5] 결과 저장 ────────────────────────────────────────────────
    print(f"\n[5/5] 결과 저장...")

    # summary.json 직렬화 (log 제외, 핵심 지표만)
    summary_data = {
        "experiment":       "Script 32 - Synthetic Augmentation",
        "baseline_val_f1":  baseline_f1,
        "elapsed_total_min": elapsed_total,
        "conditions": {},
    }
    for key, r in cond_results.items():
        summary_data["conditions"][key] = {
            "val_f1":       r["val_f1"],
            "val_acc":      r["val_acc"],
            "delta_f1":     r["val_f1"] - baseline_f1,
            "n_synth":      r["n_synth"],
            "class_f1":     r.get("class_f1", {}),
            "class_recall": r.get("class_recall", {}),
            "ckpt_path":    r.get("ckpt_path", ""),
        }

    summary_path = LOG_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"  summary: {summary_path}")

    # report.md
    report_md = make_report(cond_results, baseline_f1)
    report_path = LOG_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  report:  {report_path}")

    print(f"\n  Done. 총 소요: {elapsed_total:.1f}분\n")


if __name__ == "__main__":
    main()
