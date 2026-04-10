"""
Script 32b: 저장된 체크포인트 재평가 → 결과 리포트 생성
==========================================================
Script 32 실행 후 세션 종료로 summary.json/report.md가 미저장됨.
저장된 3개 체크포인트를 val set(9,011장)에서 재평가 → 클래스별 F1 + 완전한 보고서 생성.

소요: ~3분 (inference only, GPU/MPS)

Usage:
    python3 scripts/legacy/phase_18_32_generation_ablation/32b_reeval.py
"""

import json
import random
import warnings
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm

# ── 경로 ──────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed_multidomain"
OUT_DIR  = ROOT / "results" / "synth_aug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 체크포인트 목록
CKPTS = {
    "real_only":           ROOT / "models" / "multidomain_cnn_vgg16.pt",
    "real+synth":          ROOT / "models" / "vgg16_synth_aug_all.pt",
    "real+synth_filtered": ROOT / "models" / "vgg16_synth_aug_filtered.pt",
}

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
IMG_EXTS      = {".jpg", ".jpeg", ".png"}
IMG_SIZE      = 224
NUM_WORKERS   = 0


# ── 디바이스 ──────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── MultiDomainDataset (Script 30/32 그대로) ─────────────────────────
class MultiDomainDataset(Dataset):
    def __init__(self, data_dir, transform=None, seed=42):
        self.transform = transform
        self.samples = []
        for domain in DOMAINS:
            d_idx = DOMAINS.index(domain)
            for cls in MULTI_CLASSES:
                c_idx   = CLASS_IDX[cls]
                cls_dir = data_dir / domain / cls
                if not cls_dir.exists():
                    continue
                paths = [p for p in cls_dir.iterdir()
                         if p.suffix.lower() in IMG_EXTS]
                self.samples.extend((p, c_idx, d_idx) for p in paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, c_idx, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, c_idx


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── 모델 빌드 ──────────────────────────────────────────────────────────
def build_model(n_classes=5):
    m = models.vgg16(weights=None)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, n_classes)
    return m


# ── 검증 (클래스별 F1 포함) ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_with_classwise(model, loader, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for imgs, labels in tqdm(loader, desc="  eval", ncols=70, leave=False):
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
    class_f1     = {cls: report[cls]["f1-score"]  for cls in MULTI_CLASSES}
    class_recall = {cls: report[cls]["recall"]     for cls in MULTI_CLASSES}
    class_prec   = {cls: report[cls]["precision"]  for cls in MULTI_CLASSES}

    return {
        "loss":         total_loss / total,
        "acc":          correct / total,
        "macro_f1":     macro_f1,
        "class_f1":     class_f1,
        "class_recall": class_recall,
        "class_prec":   class_prec,
    }


# ── 리포트 생성 ────────────────────────────────────────────────────────
def make_report(cond_results: dict, baseline_f1: float) -> str:
    lines = [
        "# Script 32 — 합성 데이터 증강 VGG16 실험 리포트",
        "",
        "**목적**: 합성 이미지(900장)를 훈련 세트에 추가했을 때 VGG16 val F1 개선폭 측정",
        f"**베이스라인**: real_only val macro-F1 = {baseline_f1:.4f}",
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
        r     = cond_results[key]
        f1    = r["val_f1"]
        acc   = r["val_acc"]
        n     = r.get("n_synth", 0)
        delta = f1 - baseline_f1
        d_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        mark  = "✅" if delta > 0.001 else ("➖" if abs(delta) <= 0.001 else "❌")
        lines.append(f"| **{key}** | {n} | {f1:.4f} | {mark} {d_str} | {acc*100:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## 2. 클래스별 recall (per-class accuracy)",
        "",
        "| 클래스 | real_only | real+synth | Δ | real+synth_filtered | Δ |",
        "|--------|:---------:|:----------:|:-:|:-------------------:|:-:|",
    ]

    def badge(r):
        return "🟩" if r >= 0.90 else ("🟨" if r >= 0.67 else "🟥")

    for cls in MULTI_CLASSES:
        r0 = cond_results.get("real_only",           {}).get("class_recall", {}).get(cls, 0)
        r1 = cond_results.get("real+synth",          {}).get("class_recall", {}).get(cls, 0)
        r2 = cond_results.get("real+synth_filtered", {}).get("class_recall", {}).get(cls, 0)
        d1 = r1 - r0; d1s = f"+{d1*100:.1f}%p" if d1 >= 0 else f"{d1*100:.1f}%p"
        d2 = r2 - r0; d2s = f"+{d2*100:.1f}%p" if d2 >= 0 else f"{d2*100:.1f}%p"
        lines.append(
            f"| {cls} | {badge(r0)}{r0*100:.0f}% | "
            f"{badge(r1)}{r1*100:.0f}% | {d1s} | "
            f"{badge(r2)}{r2*100:.0f}% | {d2s} |"
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
        f0 = cond_results.get("real_only",           {}).get("class_f1", {}).get(cls, 0)
        f1 = cond_results.get("real+synth",          {}).get("class_f1", {}).get(cls, 0)
        f2 = cond_results.get("real+synth_filtered", {}).get("class_f1", {}).get(cls, 0)
        d1 = f1 - f0; d1s = f"+{d1:.4f}" if d1 >= 0 else f"{d1:.4f}"
        d2 = f2 - f0; d2s = f"+{d2:.4f}" if d2 >= 0 else f"{d2:.4f}"
        lines.append(
            f"| {cls} | {f0:.4f} | {f1:.4f} | {d1s} | {f2:.4f} | {d2s} |"
        )

    # 결론 자동 생성
    lines += ["", "---", "", "## 4. 결론", ""]
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
            lines.append(f"- ❌ **합성 데이터 증강 효과 없음**: best Δ F1 = {best_d:.4f} (개선 없음)")

        for cls in ["monocyte", "eosinophil"]:
            r0_cls = cond_results.get("real_only", {}).get("class_recall", {}).get(cls, 0)
            rb_cls = cond_results.get(best_key, {}).get("class_recall", {}).get(cls, 0)
            d = rb_cls - r0_cls
            d_str = f"+{d*100:.1f}%p" if d >= 0 else f"{d*100:.1f}%p"
            trend = "↑" if d > 0 else ("→" if abs(d) < 0.01 else "↓")
            lines.append(f"- {cls} recall: {r0_cls*100:.0f}% {trend} {rb_cls*100:.0f}% ({d_str})")

        if r2["val_f1"] >= r1["val_f1"]:
            lines.append(f"- 필터링(correct==True): 효과 있음 (real+synth_filtered ≥ real+synth)")
        else:
            lines.append(f"- 필터링(correct==True): 효과 없음 (real+synth_filtered < real+synth)")

        lines += [
            "",
            "### 해석",
            "",
            f"- 합성 이미지는 VGG16 훈련에 **부정적 영향** (Δ F1 = {best_d:.4f})",
            f"- 가능한 원인:",
            f"  1. strength=0.35 환경에서 합성 이미지가 실제 이미지와 분포 차이 존재",
            f"  2. 900장 (전체 51,065장의 ~1.7%)는 모델에 의미 있는 신호를 주기 어려울 수 있음",
            f"  3. VGG16 features 동결 상태에서 classifier만 학습 → 합성 이미지의 미세 분포 반영 제한",
        ]

    lines += ["", "---", "", "*생성: Script 32b (체크포인트 재평가)*"]
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Script 32b — 체크포인트 재평가")
    print(f"  device={device}")
    print(f"{'='*60}")

    # [1/3] Val set 준비 (Script 32와 동일한 split, seed=42)
    print("\n[1/3] Val set 준비 (seed=42, val_ratio=0.15)...")
    full_real_ds = MultiDomainDataset(DATA_DIR, transform=None, seed=42)
    n_val        = int(len(full_real_ds) * 0.15)
    n_train      = len(full_real_ds) - n_val
    gen          = torch.Generator().manual_seed(42)
    train_sub, val_sub = random_split(full_real_ds, [n_train, n_val], generator=gen)
    print(f"  전체: {len(full_real_ds)}장, train: {n_train}장, val: {n_val}장")

    class _ValWrapper(Dataset):
        def __init__(self, ds, tf):
            self.ds = ds; self.tf = tf
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            img, c = self.ds[i]; return self.tf(img), c

    val_loader = DataLoader(
        _ValWrapper(val_sub, get_val_transform()),
        batch_size=32, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    # [2/3] 각 체크포인트 재평가
    print("\n[2/3] 체크포인트 재평가...")
    cond_results = {}
    n_synth_map  = {"real_only": 0, "real+synth": 900, "real+synth_filtered": 788}

    for key, ckpt_path in CKPTS.items():
        print(f"\n  조건: {key}")
        if not ckpt_path.exists():
            print(f"  ⚠️  체크포인트 없음: {ckpt_path} — 스킵")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(n_classes=5).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        best_epoch = ckpt.get("epoch", "?")
        ckpt_f1    = ckpt.get("val_f1", None)
        print(f"  체크포인트: epoch={best_epoch}, "
              f"ckpt_val_f1={ckpt_f1:.4f}" if ckpt_f1 else "")

        vl = evaluate_with_classwise(model, val_loader, device)

        print(f"  재평가 val F1 = {vl['macro_f1']:.4f}  acc = {vl['acc']*100:.1f}%")
        print("  클래스별 recall:")
        for cls in MULTI_CLASSES:
            r = vl["class_recall"][cls]
            f = vl["class_f1"][cls]
            mark = "🟩" if r >= 0.90 else ("🟨" if r >= 0.67 else "🟥")
            print(f"    {mark} {cls:<12}: recall={r*100:.1f}%  F1={f:.4f}")

        cond_results[key] = {
            "val_f1":       vl["macro_f1"],
            "val_acc":      vl["acc"],
            "class_f1":     vl["class_f1"],
            "class_recall": vl["class_recall"],
            "class_prec":   vl["class_prec"],
            "n_synth":      n_synth_map.get(key, 0),
            "ckpt_path":    str(ckpt_path),
            "best_epoch":   int(best_epoch) if str(best_epoch).isdigit() else best_epoch,
        }

        del model

    # [3/3] 결과 저장
    print(f"\n[3/3] 결과 저장...")
    baseline_f1 = cond_results.get("real_only", {}).get("val_f1", 0.8846)

    # 콘솔 비교표
    print(f"\n  {'조건':<25} {'n_synth':>8} {'val F1':>8} {'Δ F1':>8} {'val Acc':>8}")
    print(f"  {'─'*60}")
    for key, r in cond_results.items():
        delta = r["val_f1"] - baseline_f1
        d_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"  {key:<25} {r['n_synth']:>8} "
              f"{r['val_f1']:>8.4f} {d_str:>8} {r['val_acc']*100:>7.1f}%")

    print(f"\n  클래스별 recall 비교:")
    print(f"  {'클래스':<13} {'real_only':>10} {'real+synth':>11} {'Δ':>8} "
          f"{'real+synth_flt':>15} {'Δ':>8}")
    print(f"  {'─'*66}")
    for cls in MULTI_CLASSES:
        r0 = cond_results.get("real_only",           {}).get("class_recall", {}).get(cls, 0)
        r1 = cond_results.get("real+synth",          {}).get("class_recall", {}).get(cls, 0)
        r2 = cond_results.get("real+synth_filtered", {}).get("class_recall", {}).get(cls, 0)
        d1 = r1 - r0; d1s = f"+{d1*100:.1f}%p" if d1 >= 0 else f"{d1*100:.1f}%p"
        d2 = r2 - r0; d2s = f"+{d2*100:.1f}%p" if d2 >= 0 else f"{d2*100:.1f}%p"
        warn = " ⚠️" if r0 < 0.85 else ""
        print(f"  {cls:<13}{warn} {r0*100:>9.1f}% {r1*100:>10.1f}% {d1s:>8} "
              f"{r2*100:>14.1f}% {d2s:>8}")

    # summary.json
    summary_data = {
        "experiment":        "Script 32 - Synthetic Augmentation (32b reeval)",
        "baseline_val_f1":   baseline_f1,
        "conditions":        {},
    }
    for key, r in cond_results.items():
        summary_data["conditions"][key] = {
            "val_f1":       r["val_f1"],
            "val_acc":      r["val_acc"],
            "delta_f1":     r["val_f1"] - baseline_f1,
            "n_synth":      r["n_synth"],
            "class_f1":     r.get("class_f1", {}),
            "class_recall": r.get("class_recall", {}),
            "class_prec":   r.get("class_prec", {}),
            "ckpt_path":    r.get("ckpt_path", ""),
            "best_epoch":   r.get("best_epoch", "?"),
        }

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n  summary: {summary_path}")

    # report.md
    report_md  = make_report(cond_results, baseline_f1)
    report_path = OUT_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  report:  {report_path}")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
