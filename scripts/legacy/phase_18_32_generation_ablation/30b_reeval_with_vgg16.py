"""
Script 30b: VGG16으로 Script 28 생성 이미지 재평가
===================================================
Script 28에서 EfficientNet-B0로 측정한 CNN acc는 모든 조건에서 99%로 동일 →
생성 이미지 품질 차이를 탐지하지 못함.

이 스크립트는 Script 28의 1500장 생성 이미지를 VGG16 (Script 30)으로 재평가해
EfficientNet-B0 vs VGG16의 민감도를 직접 비교한다.

목표:
  - EfficientNet-B0: 99% (변화 없음)
  - VGG16: 조건별 차이가 나타나면 → 더 민감한 평가 도구임을 검증

입력:
  results/ip_scale_sweep/summary.json  (Script 28 결과, 이미지 경로 포함)
  models/multidomain_cnn_vgg16.pt      (Script 30으로 학습된 VGG16)

출력:
  results/vgg16_reeval/reeval_summary.json  (조건별 VGG16 acc + 원래 EfficientNet acc)
  results/vgg16_reeval/reeval_report.md     (비교 리포트)

Usage:
    python3 scripts/legacy/phase_18_32_generation_ablation/30b_reeval_with_vgg16.py
    python3 scripts/legacy/phase_18_32_generation_ablation/30b_reeval_with_vgg16.py --dry_run   # 이미지 로드 없이 구조 확인
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
SUMMARY_28   = ROOT / "results" / "ip_scale_sweep" / "summary.json"
VGG16_CKPT   = ROOT / "models" / "multidomain_cnn_vgg16.pt"
OUT_DIR      = ROOT / "results" / "vgg16_reeval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 상수 ──────────────────────────────────────────────────────────────────
CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

CONDITIONS = ["A", "B1", "B2", "B3", "C"]
COND_LABELS = {
    "A":  "기준선 (ip=0.0)",
    "B1": "IP 극약 (ip=0.05)",
    "B2": "IP 약  (ip=0.10)",
    "B3": "IP 중약 (ip=0.15)",
    "C":  "IP 중  (ip=0.20)",
}


# ── 디바이스 ──────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── VGG16 로드 ────────────────────────────────────────────────────────────
def load_vgg16(device: torch.device) -> nn.Module:
    """Script 30에서 학습된 VGG16 로드."""
    base = models.vgg16(weights=None)
    base.classifier[6] = nn.Linear(base.classifier[6].in_features, len(CLASSES))
    ckpt = torch.load(VGG16_CKPT, map_location="cpu", weights_only=False)
    base.load_state_dict(ckpt["model_state_dict"])
    print(f"  VGG16 로드 완료: epoch={ckpt.get('epoch','?')}, "
          f"val_F1={ckpt.get('val_f1', 0.0):.4f}")
    return base.eval().to(device)


# ── 이미지 변환 ──────────────────────────────────────────────────────────
VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict_one(model: nn.Module, img_path: str, device: torch.device) -> tuple[str, float]:
    """단일 이미지 → (예측 클래스명, 신뢰도)."""
    img = Image.open(img_path).convert("RGB")
    x   = VAL_TF(img).unsqueeze(0).to(device)
    logits = model(x)
    probs  = torch.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    return CLASSES[pred_idx], probs[pred_idx].item()


# ── 이미지 경로 추출 ─────────────────────────────────────────────────────
def get_image_path(result_dir: Path, cls: str, dom: str,
                   inp_idx: int, cond: str, seed_offset: int) -> Path:
    """
    Script 28의 이미지 저장 경로 패턴:
    results/ip_scale_sweep/images/{cls}/{dom}/input_{inp_idx:02d}/cond_{cond}_seed_{seed:02d}.png
    """
    dom_short = dom  # "PBC", "Raabin", "MLL23", "AMC"
    return (result_dir / "images" / cls / dom_short /
            f"input_{inp_idx:02d}" /
            f"cond_{cond}_seed_{seed_offset:02d}.png")


# ── argparse ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="VGG16으로 Script 28 생성 이미지 재평가")
    p.add_argument("--dry_run", action="store_true",
                   help="이미지 로드 없이 구조 확인만")
    return p.parse_args()


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  Script 30b — VGG16 재평가 (Script 28 이미지 1500장)")
    print(f"  device={device}")
    print(f"{'='*60}")

    # ── [1/4] VGG16 체크포인트 확인 ──────────────────────────────────────
    print(f"\n[1/4] VGG16 모델 확인...")
    if not VGG16_CKPT.exists():
        print(f"  ⚠️  VGG16 체크포인트 없음: {VGG16_CKPT}")
        print(f"  먼저 Script 30을 실행하세요: python3 scripts/legacy/phase_18_32_generation_ablation/30_vgg16_cnn_train.py")
        return

    if args.dry_run:
        print(f"  [DRY RUN] VGG16 체크포인트 존재 확인됨.")
        print(f"  Summary 경로: {SUMMARY_28}")
        print(f"  출력 경로: {OUT_DIR}")
        return

    model = load_vgg16(device)

    # ── [2/4] Script 28 summary.json 로드 ──────────────────────────────
    print(f"\n[2/4] Script 28 summary.json 로드...")
    with open(SUMMARY_28, encoding="utf-8") as f:
        summary28 = json.load(f)

    results28   = summary28["results"]
    result_dir  = ROOT / "results" / "ip_scale_sweep"
    active_conds = summary28.get("active_conds", CONDITIONS)

    print(f"  총 cls×dom 조합: {len(results28)}")
    print(f"  조건: {active_conds}")
    print(f"  n_inputs={summary28['n_inputs']}, n_seeds={summary28['n_seeds']}")
    print(f"  총 이미지: {summary28['n_total']}장")

    # ── [3/4] VGG16 재평가 ───────────────────────────────────────────────
    print(f"\n[3/4] VGG16 재평가 진행...")

    # 집계용 딕셔너리: {cond: {cls: [correct_list]}}
    acc_by_cond_cls = {cond: {cls: [] for cls in CLASSES} for cond in active_conds}
    acc_by_cond     = {cond: [] for cond in active_conds}  # 전체
    eff_acc_by_cond = {cond: [] for cond in active_conds}  # EfficientNet (Script 28)

    # 상세 결과 저장용
    detailed = []

    n_total = sum(len(r["inputs"]) * len(active_conds) * summary28["n_seeds"]
                  for r in results28)
    bar = tqdm(total=n_total, desc="  재평가", ncols=70)

    for combo in results28:
        cls     = combo["cls"]
        dom     = combo["dom"]
        cls_idx = CLASS_IDX.get(cls, -1)

        combo_detail = {"cls": cls, "dom": dom, "inputs": []}

        for inp_data in combo["inputs"]:
            inp_idx  = inp_data["inp_idx"]
            inp_detail = {"inp_idx": inp_idx, "conds": {}}

            for cond in active_conds:
                seeds_key = f"seeds_{cond}"
                if seeds_key not in inp_data:
                    bar.update(summary28["n_seeds"])
                    continue

                seeds_data = inp_data[seeds_key]
                eff_acc    = inp_data.get(f"cond_{cond}", {}).get("cnn_acc", None)

                vgg16_corrects = []
                seed_details   = []

                for sd in seeds_data:
                    seed_offset = sd["seed_offset"]
                    img_path    = get_image_path(result_dir, cls, dom,
                                                  inp_idx, cond, seed_offset)

                    if not img_path.exists():
                        bar.update(1)
                        continue

                    pred_cls, conf = predict_one(model, str(img_path), device)
                    correct = (pred_cls == cls)
                    vgg16_corrects.append(correct)
                    acc_by_cond_cls[cond][cls].append(correct)
                    acc_by_cond[cond].append(correct)

                    seed_details.append({
                        "seed_offset": seed_offset,
                        "vgg16_pred":  pred_cls,
                        "vgg16_conf":  round(conf, 4),
                        "vgg16_correct": correct,
                        "eff_pred":    sd.get("pred", ""),
                        "eff_correct": sd.get("correct", None),
                    })
                    bar.update(1)

                vgg16_acc = sum(vgg16_corrects) / len(vgg16_corrects) if vgg16_corrects else None
                if eff_acc is not None:
                    eff_acc_by_cond[cond].append(eff_acc)

                inp_detail["conds"][cond] = {
                    "vgg16_acc":   round(vgg16_acc, 4) if vgg16_acc is not None else None,
                    "eff_acc":     round(eff_acc, 4) if eff_acc is not None else None,
                    "seeds":       seed_details,
                }

            combo_detail["inputs"].append(inp_detail)
        detailed.append(combo_detail)

    bar.close()

    # ── [4/4] 결과 저장 및 리포트 ─────────────────────────────────────────
    print(f"\n[4/4] 결과 저장...")

    # 조건별 집계 통계
    agg = {}
    for cond in active_conds:
        vgg16_list = acc_by_cond[cond]
        eff_list   = eff_acc_by_cond[cond]
        cls_accs   = {cls: (sum(acc_by_cond_cls[cond][cls]) / len(acc_by_cond_cls[cond][cls])
                            if acc_by_cond_cls[cond][cls] else None)
                      for cls in CLASSES}
        agg[cond] = {
            "label":        COND_LABELS.get(cond, cond),
            "vgg16_acc":    round(sum(vgg16_list) / len(vgg16_list), 4) if vgg16_list else None,
            "eff_acc":      round(sum(eff_list) / len(eff_list), 4) if eff_list else None,
            "n":            len(vgg16_list),
            "cls_acc":      {cls: round(v, 4) if v is not None else None
                             for cls, v in cls_accs.items()},
        }

    reeval_summary = {
        "source_summary":  str(SUMMARY_28),
        "vgg16_ckpt":      str(VGG16_CKPT),
        "n_evaluated":     sum(len(acc_by_cond[c]) for c in active_conds),
        "conditions":      agg,
        "detailed":        detailed,
    }

    out_json = OUT_DIR / "reeval_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(reeval_summary, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {out_json}")

    # 마크다운 리포트
    lines = [
        "# Script 30b: VGG16 재평가 리포트",
        "",
        "EfficientNet-B0 (Script 28)의 CNN acc와 VGG16 (Script 30)의 CNN acc를 비교.",
        "",
        "## 조건별 전체 acc 비교",
        "",
        "| 조건 | 설명 | EfficientNet acc | VGG16 acc | Δ(VGG16-Eff) | n |",
        "|------|------|:---:|:---:|:---:|---:|",
    ]

    for cond in active_conds:
        a = agg[cond]
        eff  = a["eff_acc"]
        vgg  = a["vgg16_acc"]
        delta = round(vgg - eff, 4) if (eff is not None and vgg is not None) else None
        delta_str = f"{delta:+.4f}" if delta is not None else "-"
        eff_str  = f"{eff*100:.1f}%" if eff  is not None else "-"
        vgg_str  = f"{vgg*100:.1f}%" if vgg  is not None else "-"
        lines.append(f"| **{cond}** | {a['label']} | {eff_str} | {vgg_str} | {delta_str} | {a['n']} |")

    lines += [
        "",
        "## 클래스별 VGG16 acc (조건별)",
        "",
    ]

    # 클래스 × 조건 히트맵
    header  = "| 클래스 | " + " | ".join(f"**{c}**" for c in active_conds) + " |"
    divider = "|--------|" + "|".join([":---:"] * len(active_conds)) + "|"
    lines += [header, divider]

    for cls in CLASSES:
        row = f"| {cls} |"
        for cond in active_conds:
            v = agg[cond]["cls_acc"].get(cls)
            if v is None:
                row += " - |"
            elif v >= 0.90:
                row += f" 🟩{v*100:.0f}% |"
            elif v >= 0.67:
                row += f" 🟨{v*100:.0f}% |"
            else:
                row += f" 🟥{v*100:.0f}% |"
        lines.append(row)

    lines += [
        "",
        "## 해석",
        "",
        "- 🟩 ≥90%: 분류기가 생성 이미지를 올바르게 인식",
        "- 🟨 67~90%: 부분적 인식 (주의 필요)",
        "- 🟥 <67%: 분류기가 생성 이미지를 제대로 인식 못함 (생성 품질 낮거나 분류기 민감)",
        "",
        f"EfficientNet-B0 전 조건 acc: 약 99% (변화 없음 → 생성 품질 차이 탐지 불가)",
        "",
        "> VGG16 acc가 EfficientNet보다 낮다면, VGG16이 더 민감한 평가 도구임.",
    ]

    out_md = OUT_DIR / "reeval_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  MD:   {out_md}")

    # 콘솔 요약
    print(f"\n  ── 비교 결과 ──────────────────────────────")
    print(f"  {'조건':<6} {'EfficientNet':>14} {'VGG16':>10} {'Δ':>8}")
    print(f"  {'-'*42}")
    for cond in active_conds:
        a   = agg[cond]
        eff = a["eff_acc"]
        vgg = a["vgg16_acc"]
        delta = (vgg - eff) if (eff and vgg) else 0
        print(f"  {cond:<6} {eff*100 if eff else 0:12.1f}% {vgg*100 if vgg else 0:9.1f}% {delta:+8.4f}")

    print(f"\n  Done. 결과: {OUT_DIR}\n")


if __name__ == "__main__":
    main()
