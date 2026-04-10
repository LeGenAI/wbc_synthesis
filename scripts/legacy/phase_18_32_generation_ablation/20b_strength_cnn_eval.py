"""
Script 20b: Strength별 CNN 정확도 평가
========================================
results/strength_compare/ 의 이미 생성된 이미지들에 대해
multidomain_cnn.pt로 클래스 예측 정확도와 confidence를 측정한다.

출력: 콘솔 + results/strength_compare/cnn_eval.json
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT    = Path(__file__).parent.parent
CNN_CKPT = ROOT / "models" / "multidomain_cnn.pt"
RESULT_DIR = ROOT / "results" / "strength_compare"

MULTI_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_IDX     = {c: i for i, c in enumerate(MULTI_CLASSES)}

STRENGTHS = [0.35, 0.55, 0.65, 0.75]

INPUT_LABELS = [
    {"label": "input_pbc", "domain": "PBC (Spain)",  "true_class": "basophil"},
    {"label": "input_amc", "domain": "AMC (Korea)", "true_class": "basophil"},
]

# ── CNN 로드 ───────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():    return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_cnn(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base = models.efficientnet_b0(weights=None)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, 5)
    base.load_state_dict(ckpt["model_state_dict"])
    base.eval().to(device)
    return base

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict_batch(cnn, img_paths, device):
    """list[Path] → list[dict(pred_class, pred_idx, conf, probs)]"""
    results = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        x   = TRANSFORM(img).unsqueeze(0).to(device)
        logits = cnn(x)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu()
        pred_idx  = probs.argmax().item()
        results.append({
            "path":       str(p),
            "pred_class": MULTI_CLASSES[pred_idx],
            "pred_idx":   pred_idx,
            "conf":       round(probs[pred_idx].item(), 4),
            "probs":      {c: round(probs[i].item(), 4) for i, c in enumerate(MULTI_CLASSES)},
        })
    return results


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print("CNN 정확도 평가 (multidomain_cnn.pt)")
    print(f"{'='*60}")

    cnn = load_cnn(CNN_CKPT, device)
    print(f"CNN 로드 완료: {CNN_CKPT.name}")

    all_results = []

    # ── 입력별 처리 ───────────────────────────────────────────────────────
    for inp_cfg in INPUT_LABELS:
        label      = inp_cfg["label"]
        domain     = inp_cfg["domain"]
        true_class = inp_cfg["true_class"]
        true_idx   = CLASS_IDX[true_class]

        print(f"\n{'─'*50}")
        print(f"입력: {domain}  (true_class={true_class})")
        print(f"{'─'*50}")

        # 원본 입력 이미지 CNN 예측
        input_png = RESULT_DIR / label / "input.png"
        inp_pred  = predict_batch(cnn, [input_png], device)[0]
        print(f"  [원본 입력] pred={inp_pred['pred_class']} conf={inp_pred['conf']:.4f}")

        strength_cnn = []

        for strength in STRENGTHS:
            s_dir = RESULT_DIR / label / f"s{int(strength*100):03d}"
            gen_paths = sorted(s_dir.glob("gen_*.png"))

            preds = predict_batch(cnn, gen_paths, device)

            n_correct = sum(1 for p in preds if p["pred_idx"] == true_idx)
            acc       = n_correct / len(preds)
            conf_mean = float(np.mean([p["conf"] for p in preds]))
            conf_std  = float(np.std( [p["conf"] for p in preds]))

            # per-class prediction counts
            pred_counts = {}
            for p in preds:
                pred_counts[p["pred_class"]] = pred_counts.get(p["pred_class"], 0) + 1

            print(f"\n  strength={strength}")
            print(f"    acc={n_correct}/{len(preds)} ({acc*100:.0f}%)  "
                  f"conf={conf_mean:.4f}±{conf_std:.4f}")
            print(f"    pred 분포: {pred_counts}")

            strength_cnn.append({
                "strength":    strength,
                "n_images":    len(preds),
                "n_correct":   n_correct,
                "accuracy":    round(acc, 4),
                "conf_mean":   round(conf_mean, 4),
                "conf_std":    round(conf_std, 4),
                "pred_counts": pred_counts,
                "per_image":   [
                    {"seed": i, "pred": p["pred_class"], "conf": p["conf"]}
                    for i, p in enumerate(preds)
                ],
            })

        all_results.append({
            "label":      label,
            "domain":     domain,
            "true_class": true_class,
            "input_pred": inp_pred,
            "strengths":  strength_cnn,
        })

    # ── 통합 요약 ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("📊 종합 CNN 정확도 요약")
    print(f"{'='*60}")
    print(f"\n{'Strength':>10} | {'PBC acc':>10} | {'PBC conf':>10} | {'AMC acc':>10} | {'AMC conf':>10} | {'전체 acc':>10}")
    print("-" * 70)

    for si, strength in enumerate(STRENGTHS):
        pbc_s = all_results[0]["strengths"][si]
        amc_s = all_results[1]["strengths"][si]
        total_n  = pbc_s["n_images"] + amc_s["n_images"]
        total_ok = pbc_s["n_correct"] + amc_s["n_correct"]
        total_acc = total_ok / total_n
        print(f"{strength:>10.2f} | "
              f"{pbc_s['accuracy']*100:>9.0f}% | {pbc_s['conf_mean']:>10.4f} | "
              f"{amc_s['accuracy']*100:>9.0f}% | {amc_s['conf_mean']:>10.4f} | "
              f"{total_acc*100:>9.0f}%")

    # ── JSON 저장 ────────────────────────────────────────────────────────
    out_path = RESULT_DIR / "cnn_eval.json"
    out_path.write_text(json.dumps({
        "multi_classes": MULTI_CLASSES,
        "strengths":     STRENGTHS,
        "inputs":        all_results,
    }, indent=2, ensure_ascii=False))
    print(f"\n→ 저장: {out_path}")


if __name__ == "__main__":
    main()
