"""
Script 17: Cross-Domain Held-Out Evaluation
============================================
multidomain_cnn.pt와 dual_head_router.pt의 일반화 성능을 진짜 held-out test set으로 평가.

핵심:
  - 학습(16_multidomain_cnn_train.py)은 전체 데이터를 seed=42, val_ratio=0.15로 random_split
    → train(85%) + val(15%) 혼합이므로 val set은 엄밀한 held-out이 아님
  - 본 스크립트는 각 도메인×클래스 조합별로 파일명 정렬 후 상위 80%를 train_pool,
    하위 20%를 test_pool로 분리 (정렬 기준 파티셔닝 → 학습 random_split과 독립적으로
    held-out 보장은 어렵지만 최대한 분리된 이미지로 평가)
  - 더 엄밀하게는 seed=42 random_split을 재현해 val(15%) 인덱스만 사용

  ※ 본 스크립트는 "seed=42 동일 random_split 재현"으로 val(15%)를 held-out으로 사용.
    이 이미지들은 학습 gradient에 직접 사용되지 않았으나 best ckpt 선정에는 영향 줌.
    완전한 held-out을 위해 --test_ratio 옵션으로 정렬 파티셔닝 전략 병용 제공.

출력:
  results/crossdomain_eval/report.json
  results/crossdomain_eval/report.md

Usage:
    # val-split 재현 (기본, seed=42 val15%)
    python scripts/legacy/phase_08_17_domain_gap_multidomain/17_crossdomain_eval.py

    # 파일명 정렬 기반 held-out (상위 80% train 제외, 하위 20% test)
    python scripts/legacy/phase_08_17_domain_gap_multidomain/17_crossdomain_eval.py --strategy sort --test_ratio 0.20

    # 평가 이미지 수 제한
    python scripts/legacy/phase_08_17_domain_gap_multidomain/17_crossdomain_eval.py --max_per_combo 300
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import models, transforms
from tqdm import tqdm

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DATA_DIR     = ROOT / "data" / "processed_multidomain"
CNN_CKPT     = ROOT / "models" / "multidomain_cnn.pt"
ROUTER_CKPT  = ROOT / "models" / "dual_head_router.pt"
OUT_DIR      = ROOT / "results" / "crossdomain_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 메타데이터 ─────────────────────────────────────────────────────────
DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_LABELS = {
    "domain_a_pbc":   "PBC (Spain)",
    "domain_b_raabin": "Raabin (Iran)",
    "domain_c_mll23": "MLL23 (Germany)",
    "domain_e_amc":   "AMC (Korea)",
}
DOMAIN_IDX = {d: i for i, d in enumerate(DOMAINS)}

MULTI_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_IDX     = {c: i for i, c in enumerate(MULTI_CLASSES)}

IMG_EXTS    = {".jpg", ".jpeg", ".png"}
IMG_SIZE    = 224
NUM_WORKERS = 0

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── 디바이스 ──────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── 데이터 수집 ────────────────────────────────────────────────────────
def collect_samples(data_dir: Path) -> list[tuple]:
    """(path, class_idx, domain_idx) 전수 수집, 16번 스크립트와 동일 로직."""
    samples = []
    for domain in DOMAINS:
        d_idx = DOMAIN_IDX[domain]
        for cls in MULTI_CLASSES:
            c_idx   = CLASS_IDX[cls]
            cls_dir = data_dir / domain / cls
            if not cls_dir.exists():
                warnings.warn(f"[WARN] 없음: {cls_dir}")
                continue
            paths = sorted([p for p in cls_dir.iterdir()
                            if p.suffix.lower() in IMG_EXTS])
            samples.extend((p, c_idx, d_idx) for p in paths)
    return samples


def get_val_indices_valsplit(samples: list, val_ratio: float, seed: int) -> list[int]:
    """
    16_multidomain_cnn_train.py와 동일한 random_split을 재현해
    val(15%) 인덱스를 반환.
    """
    n = len(samples)
    n_val   = int(n * val_ratio)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    _, val_sub = random_split(list(range(n)), [n_train, n_val], generator=gen)
    return list(val_sub.indices)


def get_test_indices_sort(samples: list, test_ratio: float) -> list[int]:
    """
    각 (domain, class) 조합별로 파일명 정렬 후 하위 test_ratio 비율을 test set으로 분리.
    → 학습 random_split과 완전히 다른 파티셔닝 전략으로 최대한 독립적인 held-out.
    """
    combo_indices: dict[tuple, list[int]] = defaultdict(list)
    for idx, (path, c_idx, d_idx) in enumerate(samples):
        combo_indices[(c_idx, d_idx)].append(idx)

    test_indices = []
    for (c_idx, d_idx), idxs in combo_indices.items():
        # 이미 정렬된 samples이므로 그대로 사용
        n_test = max(1, int(len(idxs) * test_ratio))
        test_indices.extend(idxs[-n_test:])  # 마지막 n_test개 = 알파벳 후미

    return sorted(test_indices)


# ── Dataset ────────────────────────────────────────────────────────────
class EvalDataset(Dataset):
    """(img_tensor, class_idx, domain_idx) 반환."""

    def __init__(self, samples: list[tuple], indices: list[int],
                 max_per_combo: int = 0):
        raw = [samples[i] for i in indices]

        if max_per_combo > 0:
            # 조합별 cap 적용
            combo_cnt: dict[tuple, int] = defaultdict(int)
            capped = []
            for item in raw:
                key = (item[1], item[2])
                if combo_cnt[key] < max_per_combo:
                    capped.append(item)
                    combo_cnt[key] += 1
            self.items = capped
        else:
            self.items = raw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, c_idx, d_idx = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = EVAL_TRANSFORM(img)
        return img, c_idx, d_idx


# ── 모델 로드 ─────────────────────────────────────────────────────────
class CNNClassifier(nn.Module):
    """multidomain_cnn.pt: EfficientNet-B0 5클래스 분류기."""

    def __init__(self, ckpt_path: Path, device_str: str):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location=device_str, weights_only=False)
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, 5)
        base.load_state_dict(ckpt["model_state_dict"])
        self.model = base
        self.eval()

    def forward(self, x):
        return self.model(x)


class DualHeadRouter(nn.Module):
    """dual_head_router.pt: class_head(5) + domain_head(4)."""

    def __init__(self, ckpt_path: Path, device_str: str):
        super().__init__()
        ckpt  = torch.load(ckpt_path, map_location=device_str, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, 5)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        self.class_head = nn.Linear(1280, 5, bias=True)
        self.domain_head = nn.Sequential(
            nn.Linear(1280, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 4, bias=True),
        )
        self.load_state_dict(state)
        self.eval()

    def forward(self, x):
        emb  = self.backbone(x).flatten(1)
        norm = F.normalize(emb, p=2, dim=1)
        return self.class_head(emb), self.domain_head(norm)


# ── 평가 루틴 ─────────────────────────────────────────────────────────
def compute_metrics(
    y_true: list[int], y_pred: list[int], n_classes: int
) -> dict:
    """
    confusion matrix, per-class precision/recall/F1, macro-F1, accuracy 계산.
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class = {}
    f1s = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[i] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": int(cm[i, :].sum())}
        f1s.append(f1)

    acc      = float(np.diag(cm).sum()) / len(y_true) if y_true else 0.0
    macro_f1 = float(np.mean(f1s))

    return {
        "accuracy":   round(acc, 4),
        "macro_f1":   round(macro_f1, 4),
        "per_class":  per_class,
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate_cnn(
    model: CNNClassifier,
    loader: DataLoader,
    device: str,
) -> dict:
    """
    multidomain_cnn 평가.
    반환: {
      "overall": {acc, macro_f1, per_class, cm},
      "per_domain": { domain_name: {acc, macro_f1, per_class, cm} }
    }
    """
    model.model.to(device)
    model.eval()

    # 수집
    all_true_cls, all_pred_cls, all_dom = [], [], []

    for imgs, cls_labels, dom_labels in tqdm(loader, desc="  CNN eval", leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds  = logits.argmax(dim=1).cpu().tolist()
        all_pred_cls.extend(preds)
        all_true_cls.extend(cls_labels.tolist())
        all_dom.extend(dom_labels.tolist())

    # 전체
    overall = compute_metrics(all_true_cls, all_pred_cls, n_classes=5)
    overall["n"] = len(all_true_cls)

    # 도메인별
    per_domain = {}
    for d_idx, domain in enumerate(DOMAINS):
        mask = [i for i, d in enumerate(all_dom) if d == d_idx]
        if not mask:
            continue
        t = [all_true_cls[i] for i in mask]
        p = [all_pred_cls[i] for i in mask]
        m = compute_metrics(t, p, n_classes=5)
        m["n"] = len(t)
        per_domain[domain] = m

    return {"overall": overall, "per_domain": per_domain}


@torch.no_grad()
def evaluate_router(
    model: DualHeadRouter,
    loader: DataLoader,
    device: str,
) -> dict:
    """
    dual_head_router 평가.
    반환: {
      "class": {overall, per_domain},
      "domain": {overall, per_domain}
    }
    """
    model.to(device)
    model.eval()

    all_true_cls, all_pred_cls = [], []
    all_true_dom, all_pred_dom = [], []
    all_dom = []

    for imgs, cls_labels, dom_labels in tqdm(loader, desc="  Router eval", leave=False):
        imgs = imgs.to(device)
        cls_logits, dom_logits = model(imgs)
        pred_cls = cls_logits.argmax(1).cpu().tolist()
        pred_dom = dom_logits.argmax(1).cpu().tolist()
        all_true_cls.extend(cls_labels.tolist())
        all_pred_cls.extend(pred_cls)
        all_true_dom.extend(dom_labels.tolist())
        all_pred_dom.extend(pred_dom)
        all_dom.extend(dom_labels.tolist())

    # 클래스 분류
    cls_overall = compute_metrics(all_true_cls, all_pred_cls, n_classes=5)
    cls_overall["n"] = len(all_true_cls)
    cls_per_domain = {}
    for d_idx, domain in enumerate(DOMAINS):
        mask = [i for i, d in enumerate(all_dom) if d == d_idx]
        if not mask:
            continue
        t = [all_true_cls[i] for i in mask]
        p = [all_pred_cls[i] for i in mask]
        m = compute_metrics(t, p, n_classes=5)
        m["n"] = len(t)
        cls_per_domain[domain] = m

    # 도메인 분류
    dom_overall = compute_metrics(all_true_dom, all_pred_dom, n_classes=4)
    dom_overall["n"] = len(all_true_dom)

    return {
        "class": {"overall": cls_overall, "per_domain": cls_per_domain},
        "domain": {"overall": dom_overall},
    }


# ── Markdown 리포트 ────────────────────────────────────────────────────
def render_confusion(cm: list[list[int]], labels: list[str]) -> str:
    """confusion matrix → markdown table."""
    header = "| pred→ | " + " | ".join(f"**{l[:4]}**" for l in labels) + " |"
    sep    = "|" + "---|" * (len(labels) + 1)
    rows   = []
    for i, label in enumerate(labels):
        row_vals = " | ".join(str(cm[i][j]) for j in range(len(labels)))
        rows.append(f"| **{label[:4]}** | {row_vals} |")
    return "\n".join([header, sep] + rows)


def make_markdown(
    strategy: str,
    test_n: int,
    cnn_res: dict,
    router_res: dict,
    args,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    lines += [
        f"# WBC Cross-Domain Held-Out Evaluation Report",
        f"",
        f"> 평가 일시: {now}  ",
        f"> 전략: `{strategy}` ({'val-split 재현 (seed=42, val_ratio=0.15)' if strategy == 'valsplit' else f'정렬 파티셔닝 (test_ratio={args.test_ratio})'})  ",
        f"> 평가 이미지: **{test_n}장** (도메인×클래스 held-out)  ",
        f"> domain_d: **없음** (raw 데이터에도 존재하지 않는 번호)  ",
        f"",
        f"---",
        f"",
        f"## 1. multidomain_cnn.pt — 클래스 분류 성능",
        f"",
        f"### 1-1. 전체 (All domains)",
        f"",
    ]

    ov = cnn_res["overall"]
    lines += [
        f"| 지표 | 값 |",
        f"|------|-----|",
        f"| Accuracy | **{ov['accuracy']*100:.2f}%** |",
        f"| Macro F1 | **{ov['macro_f1']:.4f}** |",
        f"| N | {ov['n']} |",
        f"",
        f"**Per-Class F1:**",
        f"",
        f"| 클래스 | Precision | Recall | F1 | Support |",
        f"|--------|-----------|--------|----|---------|",
    ]
    for c_idx, cls in enumerate(MULTI_CLASSES):
        pc = ov["per_class"][c_idx]
        lines.append(
            f"| {cls} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
            f"**{pc['f1']:.4f}** | {pc['support']} |"
        )

    lines += [
        f"",
        f"**Confusion Matrix (행=실제, 열=예측):**",
        f"",
        render_confusion(ov["confusion_matrix"], MULTI_CLASSES),
        f"",
        f"### 1-2. 도메인별 (Domain-wise)",
        f"",
        f"| 도메인 | Accuracy | Macro F1 | N |",
        f"|--------|----------|----------|---|",
    ]
    for domain in DOMAINS:
        if domain not in cnn_res["per_domain"]:
            continue
        dm = cnn_res["per_domain"][domain]
        lines.append(
            f"| {DOMAIN_LABELS[domain]} | {dm['accuracy']*100:.2f}% "
            f"| {dm['macro_f1']:.4f} | {dm['n']} |"
        )

    lines += [f"", f"#### 도메인별 Per-Class F1", f""]
    for domain in DOMAINS:
        if domain not in cnn_res["per_domain"]:
            continue
        dm = cnn_res["per_domain"][domain]
        lines += [
            f"**{DOMAIN_LABELS[domain]}** (acc={dm['accuracy']*100:.2f}%, "
            f"F1={dm['macro_f1']:.4f}, n={dm['n']})",
            f"",
            f"| 클래스 | P | R | F1 | N |",
            f"|--------|---|---|----|---|",
        ]
        for c_idx, cls in enumerate(MULTI_CLASSES):
            pc = dm["per_class"][c_idx]
            lines.append(
                f"| {cls} | {pc['precision']:.3f} | {pc['recall']:.3f} "
                f"| **{pc['f1']:.3f}** | {pc['support']} |"
            )
        lines += [
            f"",
            f"Confusion matrix:",
            f"",
            render_confusion(dm["confusion_matrix"], MULTI_CLASSES),
            f"",
        ]

    # Router section
    rc = router_res["class"]
    rd = router_res["domain"]

    lines += [
        f"---",
        f"",
        f"## 2. dual_head_router.pt — 클래스 + 도메인 분류 성능",
        f"",
        f"### 2-1. 클래스 분류 (class_head)",
        f"",
        f"| 지표 | 값 |",
        f"|------|-----|",
        f"| Accuracy | **{rc['overall']['accuracy']*100:.2f}%** |",
        f"| Macro F1 | **{rc['overall']['macro_f1']:.4f}** |",
        f"| N | {rc['overall']['n']} |",
        f"",
        f"**Per-Class F1:**",
        f"",
        f"| 클래스 | P | R | F1 | Support |",
        f"|--------|---|---|----|---------|",
    ]
    for c_idx, cls in enumerate(MULTI_CLASSES):
        pc = rc["overall"]["per_class"][c_idx]
        lines.append(
            f"| {cls} | {pc['precision']:.4f} | {pc['recall']:.4f} "
            f"| **{pc['f1']:.4f}** | {pc['support']} |"
        )

    lines += [
        f"",
        f"**Confusion Matrix:**",
        f"",
        render_confusion(rc["overall"]["confusion_matrix"], MULTI_CLASSES),
        f"",
        f"### 2-2. 도메인 분류 (domain_head)",
        f"",
        f"| 지표 | 값 |",
        f"|------|-----|",
        f"| Accuracy | **{rd['overall']['accuracy']*100:.2f}%** |",
        f"| Macro F1 | **{rd['overall']['macro_f1']:.4f}** |",
        f"| N | {rd['overall']['n']} |",
        f"| 랜덤 기준선 | 25.00% |",
        f"",
        f"**Confusion Matrix (행=실제 도메인, 열=예측 도메인):**",
        f"",
        render_confusion(rd["overall"]["confusion_matrix"],
                         [DOMAIN_LABELS[d][:8] for d in DOMAINS]),
        f"",
        f"### 2-3. 도메인별 클래스 분류",
        f"",
        f"| 도메인 | Accuracy | Macro F1 | N |",
        f"|--------|----------|----------|---|",
    ]
    for domain in DOMAINS:
        if domain not in rc["per_domain"]:
            continue
        dm = rc["per_domain"][domain]
        lines.append(
            f"| {DOMAIN_LABELS[domain]} | {dm['accuracy']*100:.2f}% "
            f"| {dm['macro_f1']:.4f} | {dm['n']} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## 3. 비교 요약",
        f"",
        f"| 모델 | 전체 Acc | 전체 F1 | PBC F1 | Raabin F1 | MLL23 F1 | AMC F1 |",
        f"|------|----------|---------|--------|-----------|----------|--------|",
    ]

    def get_domain_f1(res_per_domain, domain):
        if domain in res_per_domain:
            return f"{res_per_domain[domain]['macro_f1']:.4f}"
        return "N/A"

    cnn_pd = cnn_res["per_domain"]
    rtr_pd = rc["per_domain"]
    lines += [
        f"| multidomain_cnn | {cnn_res['overall']['accuracy']*100:.2f}% "
        f"| {cnn_res['overall']['macro_f1']:.4f} "
        f"| {get_domain_f1(cnn_pd, 'domain_a_pbc')} "
        f"| {get_domain_f1(cnn_pd, 'domain_b_raabin')} "
        f"| {get_domain_f1(cnn_pd, 'domain_c_mll23')} "
        f"| {get_domain_f1(cnn_pd, 'domain_e_amc')} |",
        f"| router(class) | {rc['overall']['accuracy']*100:.2f}% "
        f"| {rc['overall']['macro_f1']:.4f} "
        f"| {get_domain_f1(rtr_pd, 'domain_a_pbc')} "
        f"| {get_domain_f1(rtr_pd, 'domain_b_raabin')} "
        f"| {get_domain_f1(rtr_pd, 'domain_c_mll23')} "
        f"| {get_domain_f1(rtr_pd, 'domain_e_amc')} |",
        f"",
        f"---",
        f"",
        f"## 4. 평가 설정",
        f"",
        f"```",
        f"strategy   : {strategy}",
        f"seed       : {args.seed}",
        f"val_ratio  : {args.val_ratio}  (valsplit 전략 시 사용)",
        f"test_ratio : {args.test_ratio} (sort 전략 시 사용)",
        f"max_per_combo: {args.max_per_combo} (0=무제한)",
        f"batch_size : {args.batch_size}",
        f"device     : {get_device()}",
        f"cnn_ckpt   : {args.cnn_ckpt}",
        f"router_ckpt: {args.router_ckpt}",
        f"```",
    ]

    return "\n".join(lines)


# ── argparse ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-Domain Held-Out Evaluation (multidomain_cnn + dual_head_router)"
    )
    p.add_argument("--strategy",      choices=["valsplit", "sort"], default="valsplit",
                   help="held-out 전략: valsplit=seed 재현, sort=파일명 정렬 파티셔닝")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--val_ratio",     type=float, default=0.15,
                   help="valsplit 전략: val 비율 (16번과 동일)")
    p.add_argument("--test_ratio",    type=float, default=0.20,
                   help="sort 전략: 각 combo 하위 비율을 test로 사용")
    p.add_argument("--max_per_combo", type=int,   default=0,
                   help="도메인×클래스 조합당 최대 평가 이미지 수 (0=전체)")
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--cnn_ckpt",      type=Path,  default=CNN_CKPT)
    p.add_argument("--router_ckpt",   type=Path,  default=ROUTER_CKPT)
    p.add_argument("--out_dir",       type=Path,  default=OUT_DIR)
    return p.parse_args()


# ── main ────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print("=" * 60)
    print("  Script 17 — Cross-Domain Held-Out Evaluation")
    print(f"  strategy={args.strategy}, device={device}")
    print("=" * 60)

    # ── [1/4] 데이터 수집 & held-out 인덱스 결정 ──────────────────────
    print("\n[1/4] 데이터 수집 및 held-out 분리...")
    samples = collect_samples(DATA_DIR)
    print(f"  전체 이미지: {len(samples)}장")

    if args.strategy == "valsplit":
        print(f"  전략: seed={args.seed} random_split 재현 (val_ratio={args.val_ratio})")
        test_indices = get_val_indices_valsplit(samples, args.val_ratio, args.seed)
        strategy_label = "valsplit"
    else:
        print(f"  전략: 파일명 정렬 파티셔닝 (test_ratio={args.test_ratio})")
        test_indices = get_test_indices_sort(samples, args.test_ratio)
        strategy_label = "sort"

    print(f"  held-out 크기: {len(test_indices)}장 "
          f"({len(test_indices)/len(samples)*100:.1f}%)")

    ds = EvalDataset(samples, test_indices, max_per_combo=args.max_per_combo)
    print(f"  평가 데이터셋: {len(ds)}장 (max_per_combo={args.max_per_combo or '무제한'})")

    # combo 통계
    combo_cnt: Counter = Counter((item[1], item[2]) for item in ds.items)
    print("  도메인×클래스 분포:")
    for d_idx, domain in enumerate(DOMAINS):
        row = " | ".join(
            f"{MULTI_CLASSES[c_idx][:4]}={combo_cnt.get((c_idx, d_idx), 0)}"
            for c_idx in range(5)
        )
        print(f"    {DOMAIN_LABELS[domain][:14]:14s}: {row}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=NUM_WORKERS)

    # ── [2/4] multidomain_cnn 평가 ────────────────────────────────────
    print("\n[2/4] multidomain_cnn.pt 평가...")
    if not args.cnn_ckpt.exists():
        print(f"  [ERROR] {args.cnn_ckpt} 없음. 건너뜀.")
        cnn_res = {}
    else:
        cnn_model = CNNClassifier(args.cnn_ckpt, device)
        cnn_res   = evaluate_cnn(cnn_model, loader, device)
        ov = cnn_res["overall"]
        print(f"  전체: acc={ov['accuracy']*100:.2f}%, macro_F1={ov['macro_f1']:.4f}, n={ov['n']}")
        for domain in DOMAINS:
            if domain in cnn_res["per_domain"]:
                dm = cnn_res["per_domain"][domain]
                print(f"  {DOMAIN_LABELS[domain]:20s}: acc={dm['accuracy']*100:.2f}%, "
                      f"F1={dm['macro_f1']:.4f}, n={dm['n']}")

    # ── [3/4] dual_head_router 평가 ───────────────────────────────────
    print("\n[3/4] dual_head_router.pt 평가...")
    if not args.router_ckpt.exists():
        print(f"  [ERROR] {args.router_ckpt} 없음. 건너뜀.")
        router_res = {}
    else:
        router_model = DualHeadRouter(args.router_ckpt, device)
        router_res   = evaluate_router(router_model, loader, device)
        rc = router_res["class"]
        rd = router_res["domain"]
        print(f"  클래스: acc={rc['overall']['accuracy']*100:.2f}%, "
              f"F1={rc['overall']['macro_f1']:.4f}")
        print(f"  도메인: acc={rd['overall']['accuracy']*100:.2f}%, "
              f"F1={rd['overall']['macro_f1']:.4f} (랜덤기준: 25.0%)")
        for domain in DOMAINS:
            if domain in rc["per_domain"]:
                dm = rc["per_domain"][domain]
                print(f"  [cls] {DOMAIN_LABELS[domain]:20s}: acc={dm['accuracy']*100:.2f}%, "
                      f"F1={dm['macro_f1']:.4f}, n={dm['n']}")

    # ── [4/4] 저장 ────────────────────────────────────────────────────
    print("\n[4/4] 결과 저장...")

    report = {
        "timestamp":   datetime.now().isoformat(),
        "strategy":    strategy_label,
        "args": {
            "seed":          args.seed,
            "val_ratio":     args.val_ratio,
            "test_ratio":    args.test_ratio,
            "max_per_combo": args.max_per_combo,
            "batch_size":    args.batch_size,
            "cnn_ckpt":      str(args.cnn_ckpt),
            "router_ckpt":   str(args.router_ckpt),
        },
        "test_n":        len(ds),
        "total_n":       len(samples),
        "domain_d_exists": False,
        "domain_d_note":   "domain_d는 raw 데이터에도 존재하지 않는 번호 (a,b,c,e만 있음)",
        "multidomain_cnn": cnn_res,
        "dual_head_router": router_res,
    }

    json_path = args.out_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    if cnn_res and router_res:
        md = make_markdown(strategy_label, len(ds), cnn_res, router_res, args)
        md_path = args.out_dir / "report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        lines = md.count("\n")
        print(f"  Markdown: {md_path}  ({lines}줄)")

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
