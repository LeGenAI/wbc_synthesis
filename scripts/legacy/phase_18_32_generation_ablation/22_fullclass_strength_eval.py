"""
Script 22: 5클래스 × 4도메인 × Strength 전체 매트릭스 실험
=============================================================
각 클래스×도메인 조합에서 1장씩 → strength [0.35, 0.45, 0.55, 0.65, 0.75] × seed 3
= 5클래스 × 4도메인 × 5 strengths × 3 seeds = 300장 생성

목적: basophil이 특수한 케이스인지, 클래스별 optimal strength 산출

출력: results/fullclass_strength_eval/
  summary.json   : 전체 매트릭스
  report.md      : 클래스×도메인 히트맵
"""

import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# ── WBCRouter 임포트 ─────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "router_inference", ROOT / "scripts" / "15_router_inference.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
WBCRouter     = mod.WBCRouter
ssim_pair     = mod.ssim_pair
MULTI_CLASSES = mod.MULTI_CLASSES   # ['basophil','eosinophil','lymphocyte','monocyte','neutrophil']

# ── 설정 ─────────────────────────────────────────────────────────────────
ROUTER_CKPT = ROOT / "models" / "dual_head_router.pt"
CNN_CKPT    = ROOT / "models" / "multidomain_cnn.pt"
OUT_DIR     = ROOT / "results" / "fullclass_strength_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRENGTHS = [0.35, 0.45, 0.55, 0.65, 0.75]
N_SEEDS   = 3    # seed 0~2
SEED_BASE = 42

DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_SHORT = {
    "domain_a_pbc":    "PBC",
    "domain_b_raabin": "Raabin",
    "domain_c_mll23":  "MLL23",
    "domain_e_amc":    "AMC",
}
CLASS_IDX = {c: i for i, c in enumerate(MULTI_CLASSES)}

DATA_DIR = ROOT / "data" / "processed_multidomain"

# ── CNN 유틸 ─────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_cnn(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base = models.efficientnet_b0(weights=None)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, 5)
    base.load_state_dict(ckpt["model_state_dict"])
    return base.eval().to(device)

@torch.no_grad()
def cnn_predict_batch(cnn, gen_imgs, device):
    """list[PIL.Image] → list[dict(pred_class, conf)]"""
    results = []
    for img in gen_imgs:
        x = TRANSFORM(img).unsqueeze(0).to(device)
        probs = F.softmax(cnn(x), dim=1).squeeze(0).cpu()
        idx   = probs.argmax().item()
        results.append({
            "pred_class": MULTI_CLASSES[idx],
            "pred_idx":   idx,
            "conf":       round(probs[idx].item(), 4),
        })
    return results


# ── 리포트 생성 ────────────────────────────────────────────────────────────
def make_report(matrix: dict, strengths: list) -> str:
    """
    matrix[class_name][domain_key] = list of strength_results
    strength_result = {strength, ssim_mean, cnn_acc, cnn_conf_mean}
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 5클래스 × 4도메인 × Strength 전체 매트릭스",
        "",
        f"> **생성 일시:** {ts}  ",
        f"> **Strengths:** {strengths}  ",
        f"> **Seeds/strength:** {N_SEEDS}  ",
        f"> **총 조합:** 5클래스 × 4도메인 × {len(strengths)} strengths × {N_SEEDS} seeds",
        "",
        "---",
        "",
    ]

    for metric_name, metric_key, fmt, good_thr, warn_thr in [
        ("CNN accuracy (%) — 클래스 정확도",  "cnn_acc",  "{:.0%}", 1.0, 0.8),
        ("SSIM mean — 원본 보존율",           "ssim_mean", "{:.4f}", 0.95, 0.85),
        ("CNN confidence mean",               "cnn_conf",  "{:.4f}", 0.90, 0.75),
    ]:
        lines += [f"## {metric_name}", ""]

        for cls_name in MULTI_CLASSES:
            lines += [
                f"### {cls_name}",
                "",
                "| 도메인 | " + " | ".join(f"s={s}" for s in strengths) + " |",
                "|--------|" + "|".join(["--------"] * len(strengths)) + "|",
            ]
            for dom in DOMAINS:
                dom_short = DOMAIN_SHORT[dom]
                s_results = matrix.get(cls_name, {}).get(dom, [])
                if not s_results:
                    lines.append(f"| **{dom_short}** | " + " | ".join(["N/A"] * len(strengths)) + " |")
                    continue
                row = [f"**{dom_short}**"]
                for sr in s_results:
                    v = sr[metric_key]
                    if   v >= good_thr: e = "🟩"
                    elif v >= warn_thr: e = "🟨"
                    elif v >= 0.5:      e = "🟧"
                    else:               e = "🟥"
                    if metric_key == "cnn_acc":
                        row.append(f"{e} {v:.0%}")
                    else:
                        row.append(f"{e} {v:.4f}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    # ── 클래스별 권장 strength 요약 ─────────────────────────────────────
    lines += [
        "---",
        "",
        "## 클래스 × 도메인별 최적 Strength 권장",
        "> CNN acc ≥ 80% 조건에서 SSIM이 가장 낮은(variation 최대) strength 선택",
        "",
        "| 클래스 | PBC | Raabin | MLL23 | AMC |",
        "|--------|-----|--------|-------|-----|",
    ]
    for cls_name in MULTI_CLASSES:
        row = [f"**{cls_name}**"]
        for dom in DOMAINS:
            s_results = matrix.get(cls_name, {}).get(dom, [])
            best_s    = None
            best_ssim = 999.0
            for sr in s_results:
                if sr["cnn_acc"] >= 0.8 and sr["ssim_mean"] < best_ssim:
                    best_ssim = sr["ssim_mean"]
                    best_s    = sr["strength"]
            if best_s is None:
                row.append("0.35⚠️")
            else:
                row.append(f"**{best_s}** ({best_ssim:.4f})")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", ""]
    return "\n".join(lines)


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    n_total = len(MULTI_CLASSES) * len(DOMAINS) * len(STRENGTHS) * N_SEEDS
    print("=" * 65)
    print("Script 22: 5클래스 × 4도메인 × Strength 전체 매트릭스")
    print(f"  Strengths: {STRENGTHS}")
    print(f"  Seeds/strength: {N_SEEDS}")
    print(f"  총 생성: {n_total}장")
    print("=" * 65)

    device = get_device()
    print(f"\n디바이스: {device}")
    print("[1/4] CNN 로드...")
    cnn = load_cnn(CNN_CKPT, device)

    print("[2/4] WBCRouter 초기화...")
    router = WBCRouter(router_ckpt=ROUTER_CKPT, cnn_ckpt=CNN_CKPT, device=None)

    rng = random.Random(SEED_BASE)

    # matrix[class][domain] = list of per-strength dicts
    matrix   = {c: {} for c in MULTI_CLASSES}
    raw_data = []

    pipe_loaded = False
    total_done  = 0

    print(f"\n[3/4] 생성 + 평가 루프 ({n_total}장)...")

    for cls_name in MULTI_CLASSES:
        for dom in DOMAINS:
            img_dir = DATA_DIR / dom / cls_name
            if not img_dir.exists():
                print(f"  ⚠️ {dom}/{cls_name} 디렉토리 없음, skip")
                continue

            files  = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            chosen = rng.choice(files)

            out_key = f"{cls_name}_{DOMAIN_SHORT[dom]}"
            out_dir = OUT_DIR / cls_name / DOMAIN_SHORT[dom]
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  ── {cls_name} × {DOMAIN_SHORT[dom]} ({chosen.name}) ──")

            img = Image.open(chosen).convert("RGB")
            (out_dir / "input.png").write_bytes(chosen.read_bytes())

            # 파이프라인 로드 (첫 번째 조합에서만)
            if not pipe_loaded:
                route_res   = router.route(img, generate=True, seed=99)
                pipe_loaded = True
            else:
                route_res = router.route(img, generate=False)

            prompt   = route_res.get("prompt") or mod.build_class_domain_prompt(
                CLASS_IDX[cls_name],
                mod.DOMAINS.index(dom) if dom in mod.DOMAINS else 0
            )
            # generate=False 시 prompt가 없을 수 있으므로 직접 빌드
            if not prompt:
                dom_idx = mod.DOMAINS.index(dom) if dom in mod.DOMAINS else 0
                prompt  = mod.build_class_domain_prompt(CLASS_IDX[cls_name], dom_idx)

            per_strength = []

            for strength in STRENGTHS:
                s_dir = out_dir / f"s{int(strength*100):03d}"
                s_dir.mkdir(exist_ok=True)

                gen_imgs = []
                print(f"    s={strength} ", end="", flush=True)
                for seed in range(N_SEEDS):
                    gen = router._generate_once(img, prompt, denoise=strength, seed=seed)
                    gen.save(s_dir / f"gen_{seed:02d}.png")
                    gen_imgs.append(gen)
                    print("·", end="", flush=True)
                    total_done += 1
                print(f" [{total_done}/{n_total}]")

                # SSIM
                ssims = [round(ssim_pair(g, img), 4) for g in gen_imgs]

                # CNN
                preds     = cnn_predict_batch(cnn, gen_imgs, device)
                n_correct = sum(1 for p in preds if p["pred_class"] == cls_name)
                cnn_acc   = n_correct / len(preds)
                conf_mean = float(np.mean([p["conf"] for p in preds]))
                pred_dist = {}
                for p in preds:
                    pred_dist[p["pred_class"]] = pred_dist.get(p["pred_class"], 0) + 1

                print(f"      SSIM={np.mean(ssims):.4f}  "
                      f"CNN={n_correct}/{len(preds)} ({cnn_acc*100:.0f}%)  "
                      f"conf={conf_mean:.4f}  {pred_dist}")

                sr = {
                    "strength":   strength,
                    "ssim_mean":  round(float(np.mean(ssims)), 4),
                    "ssim_std":   round(float(np.std(ssims)),  4),
                    "cnn_acc":    round(cnn_acc, 4),
                    "cnn_conf":   round(conf_mean, 4),
                    "pred_dist":  pred_dist,
                    "n_correct":  n_correct,
                    "n_total":    len(preds),
                }
                per_strength.append(sr)

            matrix[cls_name][dom] = per_strength
            raw_data.append({
                "class":      cls_name,
                "domain":     dom,
                "input_file": str(chosen),
                "strengths":  per_strength,
            })

    # ── 저장 ─────────────────────────────────────────────────────────────
    print("\n[4/4] 결과 저장...")
    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strengths": STRENGTHS,
        "n_seeds":   N_SEEDS,
        "data":      raw_data,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → {json_path}")

    md = make_report(matrix, STRENGTHS)
    md_path = OUT_DIR / "report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  → {md_path}")

    # ── 터미널 요약 ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 클래스별 CNN acc 요약 (전 도메인 평균)")
    print("=" * 65)
    header = f"{'클래스':<14} | " + " | ".join(f"s={s:<4}" for s in STRENGTHS)
    print(header)
    print("-" * len(header))

    for cls_name in MULTI_CLASSES:
        vals = []
        for s_idx, strength in enumerate(STRENGTHS):
            accs = [matrix[cls_name][dom][s_idx]["cnn_acc"]
                    for dom in DOMAINS if dom in matrix[cls_name]]
            vals.append(f"{np.mean(accs)*100:5.0f}%")
        print(f"  {cls_name:<12} | " + " | ".join(vals))

    print("\n📊 도메인별 CNN acc 요약 (전 클래스 평균)")
    print("-" * len(header))
    for dom in DOMAINS:
        vals = []
        for s_idx, strength in enumerate(STRENGTHS):
            accs = [matrix[cls][dom][s_idx]["cnn_acc"]
                    for cls in MULTI_CLASSES if dom in matrix[cls]]
            vals.append(f"{np.mean(accs)*100:5.0f}%")
        print(f"  {DOMAIN_SHORT[dom]:<12} | " + " | ".join(vals))

    print(f"\n✅ 완료! → {OUT_DIR}")


if __name__ == "__main__":
    main()
