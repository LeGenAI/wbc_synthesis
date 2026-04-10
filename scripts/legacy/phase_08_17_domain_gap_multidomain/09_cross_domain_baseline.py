"""
Script 09: Cross-Domain Baseline Evaluation
=============================================
baseline_cnn.pt (PBC Barcelona으로 학습된 EfficientNet-B0, 8클래스)를
재학습 없이 4개 도메인에서 평가 → 도메인 갭 수치화.

5개 공통 클래스 (basophil, eosinophil, lymphocyte, monocyte, neutrophil)만 사용.

출력:
  results/cross_domain/cross_domain_results.json
  results/cross_domain/confusion_matrices.png

Usage:
    python scripts/legacy/phase_08_17_domain_gap_multidomain/09_cross_domain_baseline.py
    python scripts/legacy/phase_08_17_domain_gap_multidomain/09_cross_domain_baseline.py --max_per_class 100  # 빠른 테스트
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_recall_fscore_support,
                              confusion_matrix)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "processed_multidomain"
CKPT      = ROOT / "models" / "baseline_cnn.pt"
OUT_DIR   = ROOT / "results" / "cross_domain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 도메인 / 클래스 메타데이터 ────────────────────────────────────────
DOMAINS = [
    "domain_a_pbc",
    "domain_b_raabin",
    "domain_c_mll23",
    "domain_e_amc",
]
DOMAIN_LABELS = {
    "domain_a_pbc":    "PBC (Spain)",
    "domain_b_raabin": "Raabin (Iran)",
    "domain_c_mll23":  "MLL23 (Germany)",
    "domain_e_amc":    "AMC (Korea)",
}
# 멀티도메인 5클래스 (소문자, 알파벳 순)
MULTI_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
IMG_EXTS      = {".jpg", ".jpeg", ".png"}
IMG_SIZE      = 224
BATCH_SIZE    = 64
NUM_WORKERS   = 0  # macOS MPS 안전을 위해 0


# ── 디바이스 ──────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── 모델 로드 (06_robustness_eval.py 의 load_model 재사용) ───────────
def load_baseline_model(ckpt_path: Path, device: torch.device):
    """
    반환: (model, ckpt_class_names, class_to_idx)
    baseline_cnn.pt 는 8클래스로 학습되어 있음.
    """
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name  = ckpt.get("model_name", "efficientnet_b0")
    class_names = ckpt["class_names"]
    n_cls       = len(class_names)

    if model_name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_cls)
    elif model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    elif model_name == "efficientnet_b2":
        m = models.efficientnet_b2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    else:
        raise ValueError(f"알 수 없는 모델: {model_name}")

    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    model = m.to(device)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"  모델 로드 완료: {model_name}, {n_cls}클래스")
    print(f"  ckpt 클래스: {class_names}")
    return model, class_names, class_to_idx


# ── 평가용 변환 ───────────────────────────────────────────────────────
def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── 멀티도메인 데이터셋 ───────────────────────────────────────────────
class MultiDomainDataset(Dataset):
    """
    processed_multidomain/{domain}/{class}/ 구조에서 이미지 로드.
    레이블은 ckpt_class_to_idx 공간(8클래스 인덱스)으로 반환.
    이미지 로드 실패는 초기화 단계에서 필터링.
    """
    def __init__(
        self,
        domain_dir: Path,
        multi_classes: list,
        ckpt_class_to_idx: dict,
        transform,
        max_per_class: int = 1000,
        seed: int = 42,
    ):
        self.transform = transform
        self.samples   = []  # [(path, ckpt_label_idx)]
        rng = random.Random(seed)

        for cls in multi_classes:
            ckpt_label = ckpt_class_to_idx.get(cls)
            if ckpt_label is None:
                warnings.warn(f"  [WARN] '{cls}' 가 ckpt 클래스 목록에 없음 — 스킵")
                continue

            cls_dir = domain_dir / cls
            if not cls_dir.exists():
                warnings.warn(f"  [WARN] 디렉토리 없음: {cls_dir}")
                continue

            paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
            if len(paths) > max_per_class:
                paths = rng.sample(paths, max_per_class)
            if len(paths) < max_per_class:
                warnings.warn(
                    f"  [INFO] {domain_dir.name}/{cls}: {len(paths)}장 "
                    f"(요청 {max_per_class}장 미달)"
                )

            # 로드 가능 여부 사전 필터링
            valid = []
            for p in paths:
                try:
                    img = Image.open(p)
                    img.verify()
                    valid.append((p, ckpt_label))
                except Exception:
                    pass
            self.samples.extend(valid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── 도메인 평가 ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_domain(
    model: nn.Module,
    domain_dir: Path,
    multi_classes: list,
    ckpt_class_names: list,
    ckpt_class_to_idx: dict,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    max_per_class: int = 1000,
    seed: int = 42,
) -> dict:
    """
    반환:
        accuracy, macro_f1, per_class {p/r/f1},
        preds_8cls, labels_8cls (ckpt 인덱스 공간),
        preds_5cls, labels_5cls (5클래스 재매핑),
        n_samples
    """
    tf = get_eval_transform()
    ds = MultiDomainDataset(
        domain_dir, multi_classes, ckpt_class_to_idx,
        tf, max_per_class, seed
    )
    if len(ds) == 0:
        return {"error": "데이터셋 비어있음", "n_samples": 0}

    loader = DataLoader(ds, batch_size=batch_size,
                        num_workers=NUM_WORKERS, pin_memory=False)

    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc=f"    평가", ncols=70, leave=False):
        imgs = imgs.to(device)
        out  = model(imgs)
        # 5클래스 서브셋만 소프트맥스 (다른 3클래스는 무시)
        # — ckpt 8차원 소프트맥스 중 5개 해당 인덱스를 선택하여 argmax
        valid_idx = torch.tensor(
            [ckpt_class_to_idx[c] for c in multi_classes], device=device
        )
        sub_logits = out[:, valid_idx]         # (B, 5)
        sub_pred   = sub_logits.argmax(dim=1)  # 0~4 (멀티클래스 5개 중)
        # 5클래스 인덱스 → ckpt 인덱스로 다시 변환 (labels와 같은 공간)
        pred_ckpt  = valid_idx[sub_pred].cpu().numpy()
        all_preds.extend(pred_ckpt)
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 전체 정확도/F1 (ckpt 인덱스 공간)
    acc      = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    # per-class (5클래스 서브셋 공간으로 재매핑)
    ckpt2multi = {ckpt_class_to_idx[c]: i for i, c in enumerate(multi_classes)}
    preds_5  = np.array([ckpt2multi.get(p, -1) for p in all_preds])
    labels_5 = np.array([ckpt2multi.get(l, -1) for l in all_labels])
    valid_mask = (preds_5 >= 0) & (labels_5 >= 0)
    preds_5  = preds_5[valid_mask]
    labels_5 = labels_5[valid_mask]

    prec, rec, f1s, _ = precision_recall_fscore_support(
        labels_5, preds_5, labels=list(range(len(multi_classes))),
        average=None, zero_division=0
    )
    per_class = {
        c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1s[i])}
        for i, c in enumerate(multi_classes)
    }

    return {
        "accuracy":   acc,
        "macro_f1":   macro_f1,
        "per_class":  per_class,
        "preds_ckpt": all_preds.tolist(),
        "labels_ckpt":all_labels.tolist(),
        "preds_5cls": preds_5.tolist(),
        "labels_5cls":labels_5.tolist(),
        "n_samples":  len(ds),
    }


# ── 혼동 행렬 플롯 ───────────────────────────────────────────────────
def plot_confusion_matrices(
    domain_results: dict,
    multi_classes: list,
    out_path: Path,
) -> None:
    n_valid = sum(1 for r in domain_results.values()
                  if r.get("n_samples", 0) > 0)
    if n_valid == 0:
        print("  [WARN] 혼동 행렬을 그릴 결과가 없음")
        return

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(24, 6), dpi=130)
    fig.patch.set_facecolor("#111827")

    short_names = [c[:3] for c in multi_classes]  # 짧은 레이블

    for ax, domain in zip(axes, DOMAINS):
        ax.set_facecolor("#1f2937")
        r = domain_results.get(domain, {})
        if r.get("n_samples", 0) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            ax.set_title(DOMAIN_LABELS[domain], color="white", fontsize=9)
            continue

        cm = confusion_matrix(
            r["labels_5cls"], r["preds_5cls"],
            labels=list(range(len(multi_classes)))
        )
        # 행 정규화
        cm_norm = cm.astype(float)
        row_sum = cm_norm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_norm /= row_sum

        sns.heatmap(
            cm_norm, ax=ax,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            cmap="Blues", vmin=0, vmax=1,
            xticklabels=short_names, yticklabels=short_names,
            cbar=False,
        )
        acc = r["accuracy"]
        f1  = r["macro_f1"]
        ax.set_title(f"{DOMAIN_LABELS[domain]}\nAcc={acc:.3f}  F1={f1:.3f}",
                     color="white", fontsize=9, pad=6)
        ax.set_xlabel("Predicted", color="grey", fontsize=8)
        ax.set_ylabel("True",      color="grey", fontsize=8)
        ax.tick_params(colors="grey", labelsize=7)

    fig.suptitle("Cross-Domain Baseline: Confusion Matrices (baseline_cnn.pt trained on PBC)",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ 혼동 행렬 저장: {out_path}")


# ── 결과 테이블 출력 ──────────────────────────────────────────────────
def print_results_table(
    domain_results: dict,
    multi_classes: list,
) -> None:
    print("\n  === 크로스 도메인 베이스라인 결과 ===\n")
    # 헤더
    col_w = 12
    header = f"  {'도메인':<24}" + "".join(f"{c[:8]:>{col_w}}" for c in multi_classes)
    header += f"  {'Overall Acc':>12}  {'Macro F1':>10}"
    print(header)
    print("  " + "-" * len(header))

    for domain in DOMAINS:
        r = domain_results.get(domain, {})
        if r.get("n_samples", 0) == 0:
            print(f"  {DOMAIN_LABELS[domain]:<24}  (데이터 없음)")
            continue
        row = f"  {DOMAIN_LABELS[domain]:<24}"
        for cls in multi_classes:
            f1 = r["per_class"][cls]["f1"]
            row += f"{f1:>{col_w}.3f}"
        row += f"  {r['accuracy']:>12.3f}  {r['macro_f1']:>10.3f}"
        row += f"  (n={r['n_samples']:,})"
        print(row)
    print()


# ── JSON 저장 ─────────────────────────────────────────────────────────
def save_results_json(domain_results: dict, out_path: Path) -> None:
    # numpy → list 변환은 evaluate_domain에서 이미 처리됨
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(domain_results, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 결과 JSON 저장: {out_path}")


# ── argparse ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Cross-domain baseline evaluation")
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--max_per_class", type=int, default=1000,
                   help="도메인당 클래스당 최대 이미지 수")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Script 09 — Cross-Domain Baseline Evaluation")
    print(f"  device={device}, max_per_class={args.max_per_class}")
    print(f"{'='*60}")

    # 1) 모델 로드
    model, ckpt_class_names, ckpt_class_to_idx = load_baseline_model(CKPT, device)

    # 5클래스가 ckpt 클래스에 모두 포함되어 있는지 확인
    missing = [c for c in MULTI_CLASSES if c not in ckpt_class_to_idx]
    if missing:
        print(f"  [ERROR] 다음 클래스가 ckpt에 없음: {missing}")
        return

    # 2) 도메인별 평가
    domain_results = {}
    for domain in DOMAINS:
        domain_dir = DATA_DIR / domain
        print(f"\n  [{DOMAIN_LABELS[domain]}] 평가 시작...")
        r = evaluate_domain(
            model, domain_dir,
            MULTI_CLASSES, ckpt_class_names, ckpt_class_to_idx,
            device, args.batch_size, args.max_per_class, args.seed
        )
        domain_results[domain] = r
        if r.get("n_samples", 0) > 0:
            print(f"    → Accuracy={r['accuracy']:.3f}, Macro F1={r['macro_f1']:.3f}, "
                  f"n={r['n_samples']:,}")
        else:
            print(f"    → {r.get('error', '결과 없음')}")

    # 3) 결과 출력
    print_results_table(domain_results, MULTI_CLASSES)

    # 4) 혼동 행렬 플롯
    plot_confusion_matrices(domain_results, MULTI_CLASSES,
                            OUT_DIR / "confusion_matrices.png")

    # 5) JSON 저장
    save_results_json(domain_results, OUT_DIR / "cross_domain_results.json")

    print(f"\n  결과 저장 위치: {OUT_DIR}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
