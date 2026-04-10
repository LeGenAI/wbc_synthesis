"""
Script 16: Multi-Domain EfficientNet-B0 Training
=================================================
4개 도메인 혼합 데이터로 EfficientNet-B0 (5클래스)를 학습.
기존 baseline_cnn.pt (pbc 단일 도메인)의 cross-domain 한계를 극복.

핵심 설계:
  - 5클래스 직접 학습 (basophil, eosinophil, lymphocyte, monocyte, neutrophil)
  - WeightedRandomSampler: 도메인×클래스 조합 불균형 보정 (amc/basophil 127장 vs raabin/neutrophil 10862장)
  - ImageNet pretrained EfficientNet-B0 → 멀티도메인 fine-tune
  - 도메인별 검증으로 cross-domain F1 모니터링
  - Label smoothing (0.1) + 강한 augmentation

출력:
  models/multidomain_cnn.pt
  results/multidomain_cnn_train/train_log.json

Usage:
    python scripts/legacy/phase_08_17_domain_gap_multidomain/16_multidomain_cnn_train.py
    python scripts/legacy/phase_08_17_domain_gap_multidomain/16_multidomain_cnn_train.py --epochs 30 --lr 1e-4 --batch_size 32
    python scripts/legacy/phase_08_17_domain_gap_multidomain/16_multidomain_cnn_train.py --dry_run
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import (DataLoader, Dataset, WeightedRandomSampler,
                               random_split)
from torchvision import models, transforms
from tqdm import tqdm

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed_multidomain"
OUT_DIR  = ROOT / "models"
LOG_DIR  = ROOT / "results" / "multidomain_cnn_train"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── 메타데이터 ─────────────────────────────────────────────────────────
DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_LABELS = {
    "domain_a_pbc":    "PBC (Spain)",
    "domain_b_raabin": "Raabin (Iran)",
    "domain_c_mll23":  "MLL23 (Germany)",
    "domain_e_amc":    "AMC (Korea)",
}
MULTI_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_IDX     = {c: i for i, c in enumerate(MULTI_CLASSES)}

IMG_EXTS    = {".jpg", ".jpeg", ".png"}
IMG_SIZE    = 224
NUM_WORKERS = 0  # macOS MPS 안전


# ── 디바이스 ──────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── 재현성 ─────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── 데이터셋 ──────────────────────────────────────────────────────────
class MultiDomainDataset(Dataset):
    """
    4도메인 × 5클래스 이미지를 수집하는 데이터셋.
    (img_tensor, class_idx) 반환.
    get_sample_weights() → WeightedRandomSampler용 가중치.
    """

    def __init__(
        self,
        data_dir: Path,
        transform=None,
        seed: int = 42,
    ):
        self.transform = transform
        # (path, class_idx, domain_idx)
        self.samples: list[tuple] = []
        rng = random.Random(seed)

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

        print(f"  MultiDomainDataset: {len(self.samples)}장 "
              f"({len(DOMAINS)}도메인 × {len(MULTI_CLASSES)}클래스)")
        self._print_stats()

    def _print_stats(self):
        from collections import Counter
        combo = Counter((c, d) for (_, c, d) in self.samples)
        print("  도메인×클래스 분포 (domain | class: count):")
        for d_idx, domain in enumerate(DOMAINS):
            row = " | ".join(
                f"{MULTI_CLASSES[c_idx][:4]}={combo.get((c_idx, d_idx), 0)}"
                for c_idx in range(len(MULTI_CLASSES))
            )
            print(f"    {DOMAIN_LABELS[domain][:12]:12s}: {row}")

    def get_sample_weights(self) -> list:
        """
        각 (domain, class) 조합에 균등 가중치를 부여.
        가중치 = 1 / (해당 조합의 총 샘플 수)
        → WeightedRandomSampler로 모든 도메인×클래스 조합이 균등하게 샘플링됨.
        """
        from collections import Counter
        combo_counts = Counter((c, d) for (_, c, d) in self.samples)
        return [1.0 / combo_counts[(c, d)] for (_, c, d) in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, c_idx, _ = self.samples[idx]  # domain은 학습에 불필요
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, c_idx


# ── 변환 ───────────────────────────────────────────────────────────────
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


# ── 모델 빌드 ──────────────────────────────────────────────────────────
def build_model(n_classes: int = 5) -> nn.Module:
    """EfficientNet-B0 (ImageNet pretrained) → n_classes head."""
    m = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m


# ── 학습 한 epoch ──────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in tqdm(loader, desc="  train", ncols=70, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs           = imgs.size(0)
        total_loss  += loss.item() * bs
        correct     += (logits.argmax(1) == labels).sum().item()
        total       += bs

    return {"loss": total_loss / total, "acc": correct / total}


# ── 검증 ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  val ", ncols=70, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits       = model(imgs)
        loss         = criterion(logits, labels)

        bs           = imgs.size(0)
        total_loss  += loss.item() * bs
        preds        = logits.argmax(1)
        correct     += (preds == labels).sum().item()
        total       += bs
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss":    total_loss / total,
        "acc":     correct / total,
        "macro_f1": macro_f1,
    }


# ── 도메인별 검증 ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_per_domain(
    model: nn.Module,
    data_dir: Path,
    device: torch.device,
    n_per_class: int = 100,
    seed: int = 42,
) -> dict:
    """각 도메인에서 n_per_class장을 샘플링해 class 정확도·F1을 측정."""
    model.eval()
    val_tf = get_val_transform()
    rng    = random.Random(seed)
    results = {}

    for domain in DOMAINS:
        all_preds, all_labels = [], []
        for cls in MULTI_CLASSES:
            c_idx   = CLASS_IDX[cls]
            cls_dir = data_dir / domain / cls
            if not cls_dir.exists():
                continue
            paths = [p for p in cls_dir.iterdir()
                     if p.suffix.lower() in IMG_EXTS]
            paths = rng.sample(paths, min(n_per_class, len(paths)))
            for p in paths:
                img   = Image.open(p).convert("RGB")
                x     = val_tf(img).unsqueeze(0).to(device)
                pred  = model(x).argmax(1).item()
                all_preds.append(pred)
                all_labels.append(c_idx)

        if all_preds:
            acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
            macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            results[domain] = {"acc": acc, "macro_f1": macro_f1, "n": len(all_preds)}

    return results


# ── argparse ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Domain EfficientNet-B0 학습 (5클래스, 4도메인 혼합)"
    )
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--label_smooth", type=float, default=0.1,
                   help="Label smoothing for CrossEntropyLoss")
    p.add_argument("--val_ratio",  type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--dry_run",    action="store_true",
                   help="데이터·모델 초기화만 확인, 학습 없음")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device()
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Script 16 — Multi-Domain CNN Training")
    print(f"  device={device}, epochs={args.epochs}, lr={args.lr}")
    print(f"  batch={args.batch_size}, label_smooth={args.label_smooth}")
    print(f"{'='*60}")

    # ── [1/5] 데이터셋 ────────────────────────────────────────────────
    print("\n[1/5] 데이터셋 준비...")
    full_ds = MultiDomainDataset(DATA_DIR, transform=None, seed=args.seed)

    n_val   = int(len(full_ds) * args.val_ratio)
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(args.seed)
    train_sub, val_sub = random_split(full_ds, [n_train, n_val], generator=gen)

    # split 후 transform 주입
    class _TransformWrapper(Dataset):
        def __init__(self, ds, tf):
            self.ds = ds; self.tf = tf
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            img, c = self.ds[i]
            return self.tf(img), c

    # 학습셋: WeightedRandomSampler (도메인×클래스 균형)
    # 서브셋 인덱스 → 원본 samples에서 가중치 추출
    all_weights = full_ds.get_sample_weights()
    train_weights = [all_weights[i] for i in train_sub.indices]
    sampler = WeightedRandomSampler(
        train_weights, num_samples=n_train, replacement=True,
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        _TransformWrapper(train_sub, get_train_transform()),
        batch_size=args.batch_size, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=False,
    )
    val_loader = DataLoader(
        _TransformWrapper(val_sub, get_val_transform()),
        batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )
    print(f"  train: {n_train}장 (WeightedSampler), val: {n_val}장")

    # ── [2/5] 모델 ─────────────────────────────────────────────────────
    print("\n[2/5] 모델 초기화...")
    model = build_model(n_classes=len(MULTI_CLASSES)).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  EfficientNet-B0: {total_params:,} params total, "
          f"{trainable_params:,} trainable")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1
    )

    if args.dry_run:
        print("\n[DRY RUN] 초기화 완료. 학습 없이 종료.")
        dummy = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            out = model(dummy)
        print(f"  forward OK: output shape={out.shape}")
        return

    # ── [3/5] 학습 루프 ────────────────────────────────────────────────
    print(f"\n[3/5] 학습 시작 ({args.epochs} epochs)...")
    best_f1  = 0.0
    log      = []
    ckpt_out = OUT_DIR / "multidomain_cnn.pt"

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = evaluate(model, val_loader, criterion, device)
        scheduler.step(epoch)

        entry = {
            "epoch": epoch,
            "lr":    optimizer.param_groups[0]["lr"],
            "train": tr,
            "val":   vl,
        }
        log.append(entry)

        flag = ""
        if vl["macro_f1"] > best_f1:
            best_f1 = vl["macro_f1"]
            torch.save({
                "epoch":        epoch,
                "model_name":   "efficientnet_b0",
                "class_names":  MULTI_CLASSES,
                "model_state_dict": model.state_dict(),
                "val_f1":       vl["macro_f1"],
                "val_acc":      vl["acc"],
                "train_args":   vars(args),
            }, ckpt_out)
            flag = f" ✅ best (F1={best_f1:.4f})"

        print(f"  Epoch {epoch:02d}/{args.epochs} | "
              f"loss={vl['loss']:.4f} | acc={vl['acc']*100:.1f}% | "
              f"F1={vl['macro_f1']:.4f}{flag}")

    # ── [4/5] 도메인별 검증 ────────────────────────────────────────────
    print(f"\n[4/5] 도메인별 cross-domain 검증...")
    ckpt = torch.load(ckpt_out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    domain_results = evaluate_per_domain(model, DATA_DIR, device)

    print(f"\n  {'도메인':<20} {'acc':>6} {'F1':>6} {'n':>5}")
    print(f"  {'-'*42}")
    for domain, res in domain_results.items():
        label = DOMAIN_LABELS[domain]
        print(f"  {label:<20} {res['acc']*100:5.1f}% {res['macro_f1']:.4f} {res['n']:>5}")

    # ── [5/5] 결과 저장 ────────────────────────────────────────────────
    print(f"\n[5/5] 결과 저장...")
    log_data = {
        "best_val_f1":    best_f1,
        "epochs":         args.epochs,
        "domain_results": domain_results,
        "log":            log,
    }
    log_path = LOG_DIR / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"  로그: {log_path}")
    print(f"  모델: {ckpt_out}")
    print(f"\n  ── 최종 결과 ──────────────────────────────")
    print(f"  best val F1 (macro): {best_f1:.4f}")
    print(f"  도메인별 F1: " +
          ", ".join(f"{DOMAIN_LABELS[d][:4]}={r['macro_f1']:.3f}"
                    for d, r in domain_results.items()))
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
