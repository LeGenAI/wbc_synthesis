"""
Script 14: Dual-Head Router Training
=====================================
EfficientNet-B0 백본을 공유하는 Dual-Head 라우터 학습.

- class_head(5): multidomain_cnn.pt에서 5클래스 가중치 직접 로드 → 동결
  (구 baseline_cnn.pt 8→5 슬라이싱 방식 폐기; multidomain_cnn.pt는 5클래스 직접 학습)
- domain_head(4): MLP(1280→512→4, L2-norm) Xavier 초기화 → 학습

핵심 설계:
  - 백본(features+avgpool) + class_head 완전 동결
  - domain_head만 AdamW + CosineAnnealingLR로 학습
  - 학습 데이터: processed_multidomain/ 4도메인 × 5클래스
  - 검증 목표: class_acc ≥ 0.85, domain_acc ≥ 0.60

출력:
  models/dual_head_router.pt
  results/router_train/train_log.json

Usage:
    python scripts/legacy/phase_08_17_domain_gap_multidomain/14_train_router.py
    python scripts/legacy/phase_08_17_domain_gap_multidomain/14_train_router.py --epochs 15 --lr 3e-4 --max_per_class 200
    python scripts/legacy/phase_08_17_domain_gap_multidomain/14_train_router.py --dry_run  # 데이터 로딩만 확인
"""

import argparse
import itertools
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed_multidomain"
CKPT     = ROOT / "models" / "multidomain_cnn.pt"
OUT_DIR  = ROOT / "models"
LOG_DIR  = ROOT / "results" / "router_train"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── 메타데이터 ─────────────────────────────────────────────────────────
DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_LABELS = {
    "domain_a_pbc":    "PBC (Spain)",
    "domain_b_raabin": "Raabin (Iran)",
    "domain_c_mll23":  "MLL23 (Germany)",
    "domain_e_amc":    "AMC (Korea)",
}
DOMAIN_IDX = {d: i for i, d in enumerate(DOMAINS)}

MULTI_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_IDX     = {c: i for i, c in enumerate(MULTI_CLASSES)}

# multidomain_cnn.pt는 5클래스 직접 학습 → 슬라이싱 불필요
# MULTI_CLASSES 순서와 multidomain_cnn.pt의 class_names 순서가 일치해야 함
# multidomain_cnn.pt class_names: ['basophil','eosinophil','lymphocyte','monocyte','neutrophil']

IMG_EXTS   = {".jpg", ".jpeg", ".png"}
IMG_SIZE   = 224
NUM_WORKERS = 0  # macOS MPS 안전


# ── 디바이스 ──────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── DualHeadRouter ─────────────────────────────────────────────────────
class DualHeadRouter(nn.Module):
    """
    EfficientNet-B0 기반 Dual-Head 라우터.

    forward(x) → (class_logits[B,5], domain_logits[B,4])

    백본(features+avgpool) 및 class_head는 multidomain_cnn.pt 가중치를 재사용하고
    완전 동결한다. domain_head만 Xavier 초기화 후 학습한다.
    """

    def __init__(self, ckpt_path: Path, device: torch.device):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # ── 1. 백본: EfficientNet-B0 features + avgpool ─────────────
        # multidomain_cnn.pt는 5클래스 직접 학습 → classifier[1]이 nn.Linear(1280, 5)
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, 5)
        base.load_state_dict(ckpt["model_state_dict"])

        # children(): [features(Sequential), avgpool(AdaptiveAvgPool2d), classifier(Sequential)]
        # features + avgpool만 추출 → 출력 (B, 1280, 1, 1) → flatten → (B, 1280)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # ── 2. class_head: multidomain_cnn.pt에서 직접 로드 (슬라이싱 불필요), 동결
        w = ckpt["model_state_dict"]["classifier.1.weight"]  # [5, 1280]
        b = ckpt["model_state_dict"]["classifier.1.bias"]    # [5]
        self.class_head = nn.Linear(1280, 5, bias=True)
        self.class_head.weight = nn.Parameter(w.clone())
        self.class_head.bias   = nn.Parameter(b.clone())

        # ── 3. domain_head: MLP (L2-norm → 512 → 4), 학습 가능 ─────
        # 도메인별 feature norm이 10~10000 범위로 극단적이므로
        # L2 normalization 후 2-layer MLP로 분리도 향상
        self.domain_head = nn.Sequential(
            nn.Linear(1280, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 4, bias=True),
        )
        for m in self.domain_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # ── 4. 백본 + class_head 동결 ────────────────────────────────
        for p in itertools.chain(self.backbone.parameters(),
                                 self.class_head.parameters()):
            p.requires_grad = False

        print(f"  DualHeadRouter 초기화 완료 (domain_head: MLP 1280→512→4, L2-norm)")
        print(f"  학습 가능 파라미터: "
              f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}개 "
              f"(domain_head only)")

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) → (B, 1280) raw (class_head용, 동결 가중치와 호환)"""
        return self.backbone(x).flatten(1)

    def forward(self, x: torch.Tensor):
        raw_emb  = self.embed(x)                           # class_head: raw embedding
        norm_emb = F.normalize(raw_emb, p=2, dim=1)       # domain_head: L2-normalized
        return self.class_head(raw_emb), self.domain_head(norm_emb)


# ── RouterDataset ──────────────────────────────────────────────────────
class RouterDataset(Dataset):
    """
    processed_multidomain/{domain}/{class}/ 구조에서
    (img_tensor, class_label_5, domain_label_4) 반환.
    """

    def __init__(
        self,
        data_dir: Path,
        max_per_combo: int = 200,
        seed: int = 42,
        transform=None,
    ):
        self.transform = transform
        self.samples   = []  # [(path, class_idx_5, domain_idx_4)]
        rng = random.Random(seed)

        for domain in DOMAINS:
            d_idx = DOMAIN_IDX[domain]
            for cls in MULTI_CLASSES:
                c_idx   = CLASS_IDX[cls]
                cls_dir = data_dir / domain / cls
                if not cls_dir.exists():
                    warnings.warn(f"  [WARN] 없음: {cls_dir}")
                    continue

                paths = [p for p in cls_dir.iterdir()
                         if p.suffix.lower() in IMG_EXTS]
                if len(paths) > max_per_combo:
                    paths = rng.sample(paths, max_per_combo)
                elif len(paths) < max_per_combo:
                    print(f"  [INFO] {domain}/{cls}: {len(paths)}장 "
                          f"(요청 {max_per_combo} 미달, 전수 사용)")

                self.samples.extend((p, c_idx, d_idx) for p in paths)

        print(f"  RouterDataset: {len(self.samples)}장 "
              f"({len(DOMAINS)}도메인 × {len(MULTI_CLASSES)}클래스)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, c_idx, d_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, c_idx, d_idx


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
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── 학습 한 epoch ──────────────────────────────────────────────────────
def train_one_epoch(
    model: DualHeadRouter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_d: float,
    device: torch.device,
) -> dict:
    model.train()
    # domain_head만 학습 모드, backbone/class_head는 동결이므로 eval 효과
    model.backbone.eval()
    model.class_head.eval()

    total_loss = cls_loss_sum = dom_loss_sum = 0.0
    cls_correct = dom_correct = total = 0

    for imgs, cls_labels, dom_labels in tqdm(loader, desc="  train", ncols=70,
                                              leave=False):
        imgs       = imgs.to(device)
        cls_labels = cls_labels.to(device)
        dom_labels = dom_labels.to(device)

        optimizer.zero_grad()
        cls_logits, dom_logits = model(imgs)

        loss_cls = F.cross_entropy(cls_logits, cls_labels)
        loss_dom = F.cross_entropy(dom_logits, dom_labels)
        # domain_head에만 역전파 (class_head/backbone 동결로 자동 차단)
        loss = loss_dom + lambda_d * loss_cls
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss  += loss.item() * bs
        cls_loss_sum += loss_cls.item() * bs
        dom_loss_sum += loss_dom.item() * bs
        cls_correct  += (cls_logits.argmax(1) == cls_labels).sum().item()
        dom_correct  += (dom_logits.argmax(1) == dom_labels).sum().item()
        total        += bs

    return {
        "loss":       total_loss  / total,
        "cls_loss":   cls_loss_sum / total,
        "dom_loss":   dom_loss_sum / total,
        "class_acc":  cls_correct / total,
        "domain_acc": dom_correct / total,
    }


# ── 검증 ───────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(
    model: DualHeadRouter,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    cls_correct = dom_correct = total = 0
    cls_loss_sum = dom_loss_sum = 0.0

    for imgs, cls_labels, dom_labels in tqdm(loader, desc="  val ", ncols=70,
                                              leave=False):
        imgs       = imgs.to(device)
        cls_labels = cls_labels.to(device)
        dom_labels = dom_labels.to(device)

        cls_logits, dom_logits = model(imgs)
        cls_loss_sum += F.cross_entropy(cls_logits, cls_labels).item() * imgs.size(0)
        dom_loss_sum += F.cross_entropy(dom_logits, dom_labels).item() * imgs.size(0)
        cls_correct  += (cls_logits.argmax(1) == cls_labels).sum().item()
        dom_correct  += (dom_logits.argmax(1) == dom_labels).sum().item()
        total        += imgs.size(0)

    return {
        "cls_loss":   cls_loss_sum / total,
        "dom_loss":   dom_loss_sum / total,
        "class_acc":  cls_correct / total,
        "domain_acc": dom_correct / total,
    }


# ── argparse ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Dual-Head Router 학습 (domain_head만 학습, backbone 동결)"
    )
    p.add_argument("--cnn_ckpt",      type=Path, default=CKPT,
                   help="multidomain_cnn.pt 경로 (5클래스 직접 학습 모델)")
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--max_per_combo", type=int,   default=200,
                   help="도메인×클래스 조합당 최대 이미지 수")
    p.add_argument("--val_ratio",     type=float, default=0.15,
                   help="검증 분할 비율")
    p.add_argument("--lambda_d",      type=float, default=0.3,
                   help="class_loss의 domain_head 역전파 가중치 (로깅용)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--dry_run",       action="store_true",
                   help="데이터 로딩 및 모델 초기화만 확인, 학습 없음")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Script 14 — Dual-Head Router Training")
    print(f"  device={device}, epochs={args.epochs}, lr={args.lr}")
    print(f"  max_per_combo={args.max_per_combo}, batch={args.batch_size}")
    print(f"{'='*60}")

    # ── 모델 초기화 ──────────────────────────────────────────────────
    print("\n[1/4] 모델 초기화...")
    model = DualHeadRouter(args.cnn_ckpt, device).to(device)

    if args.dry_run:
        print("\n[DRY RUN] 모델 초기화 성공. 학습 없이 종료.")
        # smoke test: 더미 forward
        dummy = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            cl, dl = model(dummy)
        print(f"  class_logits: {cl.shape}, domain_logits: {dl.shape}")
        return

    # ── 데이터셋 ─────────────────────────────────────────────────────
    print("\n[2/4] 데이터셋 준비...")
    full_ds = RouterDataset(
        DATA_DIR,
        max_per_combo=args.max_per_combo,
        seed=args.seed,
        transform=None,  # split 후 transform 설정
    )

    n_val   = int(len(full_ds) * args.val_ratio)
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    # split 후 transform 주입 (Dataset wrapping)
    class _TransformWrapper(Dataset):
        def __init__(self, ds, tf):
            self.ds = ds; self.tf = tf
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            img, c, d = self.ds[i]
            return self.tf(img), c, d

    train_loader = DataLoader(
        _TransformWrapper(train_ds, get_train_transform()),
        batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        _TransformWrapper(val_ds, get_val_transform()),
        batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    print(f"  train: {n_train}장, val: {n_val}장")

    # ── 옵티마이저 (domain_head 파라미터만) ─────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── 학습 루프 ─────────────────────────────────────────────────────
    print(f"\n[3/4] 학습 시작 ({args.epochs} epochs)...")
    best_domain_acc = 0.0
    log = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, args.lambda_d, device)
        vl = validate(model, val_loader, device)
        scheduler.step()

        entry = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train": tr, "val": vl,
        }
        log.append(entry)

        print(f"  Epoch {epoch:02d}/{args.epochs} | "
              f"dom_loss={vl['dom_loss']:.4f} | "
              f"class_acc={vl['class_acc']*100:.1f}% | "
              f"domain_acc={vl['domain_acc']*100:.1f}%")

        # 최적 도메인 정확도 기준으로 저장
        if vl["domain_acc"] > best_domain_acc:
            best_domain_acc = vl["domain_acc"]
            ckpt_out = OUT_DIR / "dual_head_router.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_acc":        vl["class_acc"],
                "domain_acc":       vl["domain_acc"],
                "epoch":            epoch,
                "multi_classes":    MULTI_CLASSES,
                "domains":          DOMAINS,
                "args": vars(args),
            }, ckpt_out)
            print(f"    ✅ 최적 체크포인트 저장 "
                  f"(domain_acc={best_domain_acc*100:.1f}%): {ckpt_out}")

    # ── 결과 저장 ─────────────────────────────────────────────────────
    print(f"\n[4/4] 결과 저장...")
    log_path = LOG_DIR / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_domain_acc": best_domain_acc,
            "epochs": args.epochs,
            "log": log,
        }, f, indent=2, ensure_ascii=False)
    print(f"  로그: {log_path}")
    print(f"\n  최종 결과:")
    print(f"    best domain_acc: {best_domain_acc*100:.1f}%")
    print(f"    (랜덤 기준선: {100/len(DOMAINS):.1f}%)")
    print(f"    모델: {OUT_DIR / 'dual_head_router.pt'}")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
