"""
Script 21: 4개 도메인 × Strength 비교 실험
============================================
각 도메인에서 basophil 1장씩 선택 → strength [0.35, 0.45, 0.55, 0.65, 0.75] × seed 5
= 총 4도메인 × 5 strengths × 5 seeds = 100장 생성

목적: 도메인별 strength sensitivity를 정량화하여
      합성 데이터 생성 시 도메인별 최적 strength 산출

출력: results/domain_strength_eval/
  {domain}/{s{strength}/gen_00~04.png
  summary.json   : SSIM + CNN acc 전체 매트릭스
  report.md      : 마크다운 리포트
"""

import gc
import importlib.util
import json
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
MULTI_CLASSES = mod.MULTI_CLASSES

# ── 설정 ─────────────────────────────────────────────────────────────────
ROUTER_CKPT = ROOT / "models" / "dual_head_router.pt"
CNN_CKPT    = ROOT / "models" / "multidomain_cnn.pt"
OUT_DIR     = ROOT / "results" / "domain_strength_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRENGTHS = [0.35, 0.45, 0.55, 0.65, 0.75]
N_SEEDS   = 5
SEED_BASE = 42  # 입력 이미지 선택 seed (random.Random(SEED_BASE))

# 4개 도메인 × 각 1장 (seed 42로 선택)
import random
DATA_DIR = ROOT / "data" / "processed_multidomain"

DOMAIN_META = {
    "domain_a_pbc":    "PBC · Spain · May-Grünwald Giemsa / CellaVision",
    "domain_b_raabin": "Raabin · Iran · Giemsa / Smartphone",
    "domain_c_mll23":  "MLL23 · Germany · Pappenheim / Metafer",
    "domain_e_amc":    "AMC · Korea · Romanowsky / miLab",
}
CLASS_IDX = {c: i for i, c in enumerate(MULTI_CLASSES)}

# ── CNN 평가 유틸 ─────────────────────────────────────────────────────────
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
def cnn_predict(cnn, img_paths, device):
    results = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        x   = TRANSFORM(img).unsqueeze(0).to(device)
        probs = F.softmax(cnn(x), dim=1).squeeze(0).cpu()
        idx   = probs.argmax().item()
        results.append({
            "pred_class": MULTI_CLASSES[idx],
            "pred_idx":   idx,
            "conf":       round(probs[idx].item(), 4),
        })
    return results


# ── 마크다운 리포트 ────────────────────────────────────────────────────────
def make_report(domain_results: list[dict], strengths: list[float]) -> str:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 4도메인 × Strength 비교 실험",
        "",
        f"> **생성 일시:** {ts}  ",
        f"> **Strengths:** {strengths}  ",
        f"> **Seeds/strength:** {N_SEEDS}  ",
        f"> **클래스:** basophil (4도메인 각 1장)",
        "",
        "---",
        "",
        "## 핵심 결과: SSIM × CNN accuracy 매트릭스",
        "",
    ]

    # ── SSIM 테이블 ──────────────────────────────────────────────────────
    lines += [
        "### SSIM mean (생성 이미지 vs 원본 입력)",
        "",
        "| 도메인 | " + " | ".join(f"s={s}" for s in strengths) + " |",
        "|--------|" + "|".join(["--------"] * len(strengths)) + "|",
    ]
    for dr in domain_results:
        row = [f"**{dr['domain_short']}**"]
        for s_res in dr["strengths"]:
            v = s_res["ssim_mean"]
            if   v >= 0.95: e = "🟩"
            elif v >= 0.85: e = "🟨"
            elif v >= 0.70: e = "🟧"
            else:           e = "🟥"
            row.append(f"{e} {v:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── CNN acc 테이블 ────────────────────────────────────────────────────
    lines += [
        "### CNN accuracy (basophil 분류 정확도, 5장 중 correct)",
        "",
        "| 도메인 | " + " | ".join(f"s={s}" for s in strengths) + " |",
        "|--------|" + "|".join(["--------"] * len(strengths)) + "|",
    ]
    for dr in domain_results:
        row = [f"**{dr['domain_short']}**"]
        for s_res in dr["strengths"]:
            acc = s_res["cnn_acc"]
            n   = s_res["n_correct"]
            tot = s_res["n_images"]
            if   acc == 1.0: e = "✅"
            elif acc >= 0.8: e = "🟡"
            elif acc >= 0.6: e = "🟠"
            else:            e = "❌"
            row.append(f"{e} {n}/{tot}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── CNN conf 테이블 ────────────────────────────────────────────────────
    lines += [
        "### CNN confidence mean",
        "",
        "| 도메인 | " + " | ".join(f"s={s}" for s in strengths) + " |",
        "|--------|" + "|".join(["--------"] * len(strengths)) + "|",
    ]
    for dr in domain_results:
        row = [f"**{dr['domain_short']}**"]
        for s_res in dr["strengths"]:
            c = s_res["cnn_conf_mean"]
            if   c >= 0.90: e = "🟩"
            elif c >= 0.75: e = "🟨"
            elif c >= 0.60: e = "🟧"
            else:           e = "🟥"
            row.append(f"{e} {c:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── per-domain 상세 ───────────────────────────────────────────────────
    lines += ["---", "", "## 도메인별 상세", ""]
    for dr in domain_results:
        lines += [
            f"### {dr['domain_short']} — {dr['domain_meta']}",
            "",
            f"**입력 이미지:** `{dr['input_name']}`  ",
            f"**원본 CNN:** {dr['input_pred']} (conf={dr['input_conf']:.4f})",
            "",
        ]

        # 이미지 그리드 (per strength)
        for s_res in dr["strengths"]:
            s = s_res["strength"]
            ssim_list = s_res["ssim_list"]
            acc       = s_res["cnn_acc"]
            conf_mean = s_res["cnn_conf_mean"]
            lines += [
                f"**strength={s}** — SSIM={s_res['ssim_mean']:.4f} | CNN acc={acc*100:.0f}% | conf={conf_mean:.4f}",
                "",
                "| " + " | ".join([f"seed {i}" for i in range(N_SEEDS)]) + " |",
                "|" + "|".join(["---"] * N_SEEDS) + "|",
            ]
            cells = []
            for i, rel in enumerate(s_res["gen_rels"]):
                ssim_v = ssim_list[i]
                pred   = s_res["per_image"][i]["pred"]
                ok     = "✅" if pred == "basophil" else f"❌({pred})"
                cells.append(
                    f'<img src="{rel}" width="120" title="SSIM={ssim_v} pred={pred}">'
                )
            lines.append("| " + " | ".join(cells) + " |")
            ssim_row = []
            for i, sv in enumerate(ssim_list):
                pred = s_res["per_image"][i]["pred"]
                ok   = "✅" if pred == "basophil" else f"❌{pred[:3]}"
                ssim_row.append(f"{ok} {sv:.4f}")
            lines.append("| " + " | ".join(ssim_row) + " |")
            lines.append("")

        lines += ["---", ""]

    # ── 도메인별 최적 strength 권장 ─────────────────────────────────────
    lines += [
        "## 도메인별 최적 Strength 권장",
        "",
        "| 도메인 | 권장 strength | 근거 |",
        "|--------|--------------|------|",
    ]
    for dr in domain_results:
        # CNN acc 100% 중 SSIM 가장 낮은 것 → variation 최대
        best_s = None
        best_ssim = 999.0
        for s_res in dr["strengths"]:
            if s_res["cnn_acc"] == 1.0 and s_res["ssim_mean"] < best_ssim:
                best_ssim = s_res["ssim_mean"]
                best_s    = s_res["strength"]
        if best_s is None:
            # acc<100%인 경우 acc≥0.8 중 SSIM 최소
            for s_res in dr["strengths"]:
                if s_res["cnn_acc"] >= 0.8 and s_res["ssim_mean"] < best_ssim:
                    best_ssim = s_res["ssim_mean"]
                    best_s    = s_res["strength"]
        note = f"CNN 100% 유지 + SSIM {best_ssim:.4f}" if best_s else "N/A"
        lines.append(f"| {dr['domain_short']} | **{best_s}** | {note} |")

    lines += ["", ""]
    return "\n".join(lines)


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Script 21: 4도메인 × Strength 비교 실험")
    print(f"  Strengths: {STRENGTHS}")
    print(f"  Seeds/strength: {N_SEEDS}")
    print(f"  총 생성: {len(DOMAIN_META) * len(STRENGTHS) * N_SEEDS}장")
    print("=" * 65)

    device = get_device()
    print(f"\n디바이스: {device}")

    # CNN 로드
    print("[1/4] CNN 로드...")
    cnn = load_cnn(CNN_CKPT, device)

    # WBCRouter 초기화
    print("[2/4] WBCRouter 초기화...")
    router = WBCRouter(router_ckpt=ROUTER_CKPT, cnn_ckpt=CNN_CKPT, device=None)

    # 각 도메인에서 basophil 1장 선택
    rng = random.Random(SEED_BASE)
    domain_inputs = []
    for domain_key in sorted(DOMAIN_META.keys()):
        bdir = DATA_DIR / domain_key / "basophil"
        files = sorted(bdir.glob("*.jpg")) + sorted(bdir.glob("*.png"))
        chosen = rng.choice(files)
        domain_inputs.append({
            "domain_key":   domain_key,
            "domain_short": domain_key.split("_")[1].upper()
                            + f" ({domain_key.split('_')[2]})",
            "domain_meta":  DOMAIN_META[domain_key],
            "input_path":   chosen,
        })
        print(f"  {domain_key}: {chosen.name}")

    # ── 생성 루프 ─────────────────────────────────────────────────────────
    print("\n[3/4] 이미지 생성 + 평가...")
    all_domain_results = []

    for di, inp_cfg in enumerate(domain_inputs):
        domain_key   = inp_cfg["domain_key"]
        domain_short = inp_cfg["domain_short"]
        inp_path     = inp_cfg["input_path"]

        out_dom = OUT_DIR / domain_key
        out_dom.mkdir(exist_ok=True)

        print(f"\n  [{di+1}/4] {domain_key} — {inp_path.name}")

        # 원본 저장
        img = Image.open(inp_path).convert("RGB")
        (out_dom / "input.png").write_bytes(inp_path.read_bytes())

        # 분류 + 파이프라인 로드 (첫 도메인에서만 실제 로드됨)
        route_res = router.route(img, generate=True, seed=99)
        pred_cls  = route_res["class_name"]
        cls_conf  = route_res["class_conf"]
        prompt    = route_res["prompt"]
        print(f"    분류: {pred_cls} (conf={cls_conf:.4f})")

        # 원본 CNN 예측
        inp_cnn = cnn_predict(cnn, [out_dom / "input.png"], device)[0]
        print(f"    원본 CNN: {inp_cnn['pred_class']} conf={inp_cnn['conf']:.4f}")

        strength_results = []

        for strength in STRENGTHS:
            s_dir = out_dom / f"s{int(strength*100):03d}"
            s_dir.mkdir(exist_ok=True)

            gen_imgs  = []
            gen_paths = []
            gen_rels  = []

            print(f"    s={strength} ", end="", flush=True)
            for seed in range(N_SEEDS):
                gen = router._generate_once(img, prompt, denoise=strength, seed=seed)
                fname = f"gen_{seed:02d}.png"
                gen.save(s_dir / fname)
                gen_imgs.append(gen)
                gen_paths.append(s_dir / fname)
                gen_rels.append(f"{domain_key}/s{int(strength*100):03d}/{fname}")
                print(f"·", end="", flush=True)
            print()

            # SSIM
            ssims = [round(ssim_pair(g, img), 4) for g in gen_imgs]

            # CNN
            preds     = cnn_predict(cnn, gen_paths, device)
            n_correct = sum(1 for p in preds if p["pred_class"] == "basophil")
            cnn_acc   = n_correct / len(preds)
            conf_mean = float(np.mean([p["conf"] for p in preds]))
            pred_counts = {}
            for p in preds:
                pred_counts[p["pred_class"]] = pred_counts.get(p["pred_class"], 0) + 1

            print(f"      SSIM={np.mean(ssims):.4f}  "
                  f"CNN acc={n_correct}/{len(preds)} ({cnn_acc*100:.0f}%)  "
                  f"conf={conf_mean:.4f}  pred={pred_counts}")

            strength_results.append({
                "strength":   strength,
                "ssim_list":  ssims,
                "ssim_mean":  round(float(np.mean(ssims)), 4),
                "ssim_std":   round(float(np.std(ssims)),  4),
                "n_images":   len(preds),
                "n_correct":  n_correct,
                "cnn_acc":    round(cnn_acc, 4),
                "cnn_conf_mean": round(conf_mean, 4),
                "cnn_conf_std":  round(float(np.std([p["conf"] for p in preds])), 4),
                "pred_counts": pred_counts,
                "gen_rels":   gen_rels,
                "per_image":  [
                    {"seed": i, "pred": p["pred_class"], "conf": p["conf"]}
                    for i, p in enumerate(preds)
                ],
            })

        all_domain_results.append({
            "domain_key":   domain_key,
            "domain_short": domain_short,
            "domain_meta":  inp_cfg["domain_meta"],
            "input_name":   inp_path.name,
            "input_pred":   inp_cnn["pred_class"],
            "input_conf":   inp_cnn["conf"],
            "router_pred":  pred_cls,
            "router_conf":  round(cls_conf, 4),
            "prompt":       prompt,
            "strengths":    strength_results,
        })

    # ── 결과 저장 ──────────────────────────────────────────────────────────
    print("\n[4/4] 결과 저장...")

    # JSON
    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps({
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strengths":   STRENGTHS,
        "n_seeds":     N_SEEDS,
        "seed_base":   SEED_BASE,
        "domains":     all_domain_results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → {json_path}")

    # Markdown
    md = make_report(all_domain_results, STRENGTHS)
    md_path = OUT_DIR / "report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  → {md_path}")

    # ── 터미널 요약 ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 최종 결과 요약")
    print("=" * 65)
    print(f"\n{'도메인':<20} | " + " | ".join(f"s={s:<4}" for s in STRENGTHS))
    print("-" * (22 + 12 * len(STRENGTHS)))

    # SSIM 행
    print("[SSIM mean]")
    for dr in all_domain_results:
        vals = " | ".join(f"{sr['ssim_mean']:.4f}" for sr in dr["strengths"])
        print(f"  {dr['domain_short']:<18} | {vals}")

    # CNN acc 행
    print("\n[CNN accuracy]")
    for dr in all_domain_results:
        vals = " | ".join(
            f"{sr['n_correct']}/{sr['n_images']} ({sr['cnn_acc']*100:.0f}%)"
            for sr in dr["strengths"]
        )
        print(f"  {dr['domain_short']:<18} | {vals}")

    # 도메인별 권장 strength
    print("\n[권장 strength (CNN 100% 유지 + variation 최대)]")
    for dr in all_domain_results:
        best_s    = None
        best_ssim = 999.0
        for sr in dr["strengths"]:
            if sr["cnn_acc"] == 1.0 and sr["ssim_mean"] < best_ssim:
                best_ssim = sr["ssim_mean"]
                best_s    = sr["strength"]
        if best_s is None:
            for sr in dr["strengths"]:
                if sr["cnn_acc"] >= 0.8 and sr["ssim_mean"] < best_ssim:
                    best_ssim = sr["ssim_mean"]
                    best_s    = sr["strength"]
        print(f"  {dr['domain_short']:<18} → strength={best_s}  (SSIM={best_ssim:.4f})")

    print(f"\n✅ 완료! → {OUT_DIR}")


if __name__ == "__main__":
    main()
