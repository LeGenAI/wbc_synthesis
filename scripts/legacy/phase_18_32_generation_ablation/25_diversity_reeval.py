"""
Script 25: 다양성 재실험 — 다중 입력 이미지 × 다중 seed
=========================================================
근본 원인 발견:
  - Script 22/23: 클래스×도메인당 입력 1장 고정 → seed만으로는 다양성 부족
  - test_generation_similarity.py: 입력 20장 × seed 42~61 → 높은 다양성
  - img2img strength=0.35 → 입력 특성의 65%가 출력에 보존됨
  - ∴ 출력 다양성 ≈ 입력 이미지 다양성 × seed 다양성

해결:
  - N_INPUTS = 5  (클래스×도메인당 5장 random.sample)
  - N_SEEDS  = 5  (각 입력마다 5가지 seed)
  - STRENGTH = 0.35 고정 (Script 22 최적값)

총 생성: 5 classes × 4 domains × 5 inputs × 5 seeds = 500장

출력:
  results/diversity_reeval/
    images/{cls}/{dom}/input_{nn}/
      input.png       # 원본 입력
      seed_00~04.png  # 5가지 seed 생성
    summary.json
    gallery.md        # 입력별 행 × seed별 열 갤러리

Usage:
    python3 scripts/legacy/phase_18_32_generation_ablation/25_diversity_reeval.py
    python3 scripts/legacy/phase_18_32_generation_ablation/25_diversity_reeval.py --n_inputs 5 --n_seeds 5
"""

import argparse
import importlib.util
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_skimage
import numpy as np
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data" / "processed_multidomain"
CNN_CKPT  = ROOT / "models" / "multidomain_cnn.pt"
ROUTER_CKPT = ROOT / "models" / "dual_head_router.pt"
OUT_DIR   = ROOT / "results" / "diversity_reeval"

# ── 실험 상수 ──────────────────────────────────────────────────────────────────
CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
DOMAIN_MAP = {           # 표시명 → 실제 디렉토리명
    "PBC":    "domain_a_pbc",
    "Raabin": "domain_b_raabin",
    "MLL23":  "domain_c_mll23",
    "AMC":    "domain_e_amc",
}
DOMAINS = list(DOMAIN_MAP.keys())   # ["PBC", "Raabin", "MLL23", "AMC"]

STRENGTH  = 0.35
SEED_BASE = 42   # seed = inp_idx * 10 + seed_offset → (0~4, 10~14, 20~24, 30~34, 40~44)


# ── WBCRouter 로드 (importlib) ────────────────────────────────────────────────
def load_router_module():
    spec = importlib.util.spec_from_file_location(
        "router_inference",
        ROOT / "scripts" / "15_router_inference.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── SSIM 계산 ─────────────────────────────────────────────────────────────────
def ssim_pair(img_a: Image.Image, img_b: Image.Image) -> float:
    a = np.array(img_a.convert("RGB").resize((256, 256))) / 255.0
    b = np.array(img_b.convert("RGB").resize((256, 256))) / 255.0
    return float(ssim_skimage(a, b, data_range=1.0, channel_axis=2))


# ── CNN 분류기 로드 ────────────────────────────────────────────────────────────
def load_cnn(device):
    base = models.efficientnet_b0(weights=None)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, len(CLASSES))
    ckpt = torch.load(CNN_CKPT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    base.load_state_dict(state)
    base.eval()
    return base.to(device)


CNN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def cnn_predict(model, img: Image.Image, device) -> tuple[str, float]:
    t = CNN_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
        probs  = torch.softmax(logits, dim=1)[0]
    idx  = probs.argmax().item()
    return CLASSES[idx], probs[idx].item()


# ── 갤러리 마크다운 생성 ────────────────────────────────────────────────────────
def acc_badge(acc: float) -> str:
    if acc >= 0.90: return f"🟩 {acc:.0%}"
    if acc >= 0.67: return f"🟨 {acc:.0%}"
    return f"🟥 {acc:.0%}"


def make_gallery(all_results: list, n_inputs: int, n_seeds: int) -> str:
    lines: list[str] = []
    lines += [
        "# WBC 다양성 재실험 갤러리 (다중 입력 × 다중 seed)",
        "",
        f"> **Strength:** {STRENGTH} 고정  ",
        f"> **N_INPUTS:** {n_inputs} (cls×dom당 입력 이미지 수)  ",
        f"> **N_SEEDS:** {n_seeds} (입력당 seed 수)  ",
        f"> **총 생성:** {5 * 4 * n_inputs * n_seeds}장  ",
        "",
        "뱃지: 🟩 CNN ≥ 90% · 🟨 ≥ 67% · 🟥 < 67%",
        "",
        "**읽는 법:**",
        "- **가로(행)**: 같은 입력 → seed만 다름 (seed 랜덤성 효과)",
        "- **세로(열 기준 비교)**: 같은 seed → 입력만 다름 (입력 다양성 효과)",
        "",
        "---",
        "",
    ]

    # 목차
    lines.append("## 목차")
    lines.append("")
    for cls in CLASSES:
        lines.append(f"### {cls}")
        for dom in DOMAINS:
            lines.append(f"- [{cls} × {dom}](#{cls.lower()}-{dom.lower()})")
        lines.append("")
    lines.append("---")
    lines.append("")

    # 클래스×도메인 섹션
    for cls in CLASSES:
        lines.append(f"# {cls.upper()}")
        lines.append("")

        for dom in DOMAINS:
            lines.append(f'<a id="{cls.lower()}-{dom.lower()}"></a>')
            lines.append("")
            lines.append(f"## {cls} × {dom}")
            lines.append("")

            # 해당 항목 찾기
            entry = next(
                (r for r in all_results if r["cls"] == cls and r["dom"] == dom), None
            )
            if entry is None:
                lines.append("> ⚠️ 데이터 없음")
                lines.append("")
                continue

            # 테이블 헤더: | 입력 이미지 | seed 0 | seed 1 | ... | CNN acc |
            seed_headers = " | ".join(f"seed {s}" for s in range(n_seeds))
            lines.append(f"| 입력 이미지 | {seed_headers} | CNN acc (25장) | SSIM mean |")
            sep_parts = [":---:"] + [":---:"] * n_seeds + [":---:", ":---:"]
            lines.append("| " + " | ".join(sep_parts) + " |")

            for inp_data in entry["inputs"]:
                inp_idx   = inp_data["inp_idx"]
                inp_rel   = f"images/{cls}/{dom}/input_{inp_idx:02d}/input.png"
                inp_cell  = f"![inp{inp_idx}]({inp_rel})"

                # seed 이미지 셀
                seed_cells = []
                for s in range(n_seeds):
                    sd = next((x for x in inp_data["seeds"] if x["seed_offset"] == s), None)
                    if sd:
                        img_rel = f"images/{cls}/{dom}/input_{inp_idx:02d}/seed_{s:02d}.png"
                        pred_ok = "✅" if sd["correct"] else "❌"
                        seed_cells.append(f"![s{s}]({img_rel}){pred_ok}")
                    else:
                        seed_cells.append("—")

                acc_val  = inp_data["cnn_acc"]
                ssim_val = inp_data["ssim_mean"]
                lines.append(
                    f"| {inp_cell} | "
                    + " | ".join(seed_cells)
                    + f" | {acc_badge(acc_val)} | {ssim_val:.4f} |"
                )

            lines.append("")

            # 종합 지표
            total_acc  = entry["overall_cnn_acc"]
            total_ssim = entry["overall_ssim_mean"]
            intra_ssim = entry.get("intra_input_ssim_mean", None)   # 같은 입력 5seed 간 SSIM
            inter_ssim = entry.get("inter_input_ssim_mean", None)   # 다른 입력 간 SSIM

            lines.append(f"> **종합 CNN acc:** {acc_badge(total_acc)}  ")
            lines.append(f"> **SSIM vs 입력 mean:** {total_ssim:.4f}  ")
            if intra_ssim is not None:
                lines.append(f"> **Intra-input SSIM** (같은 입력, seed 간): {intra_ssim:.4f}  ")
            if inter_ssim is not None:
                lines.append(f"> **Inter-input SSIM** (다른 입력 간): {inter_ssim:.4f}  ")
                if intra_ssim is not None and inter_ssim < intra_ssim:
                    lines.append(f"> ✅ inter < intra → **입력 다양성이 seed 다양성보다 큼**  ")
            lines.append("")
            lines.append("---")
            lines.append("")

    # 전체 요약 테이블
    lines.append("# 전체 결과 요약")
    lines.append("")
    lines.append("| 클래스 | 도메인 | CNN acc | SSIM mean | Intra SSIM | Inter SSIM |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for r in all_results:
        intra = f"{r.get('intra_input_ssim_mean', 0):.4f}" if r.get("intra_input_ssim_mean") else "—"
        inter = f"{r.get('inter_input_ssim_mean', 0):.4f}" if r.get("inter_input_ssim_mean") else "—"
        lines.append(
            f"| {r['cls']} | {r['dom']} | {acc_badge(r['overall_cnn_acc'])} "
            f"| {r['overall_ssim_mean']:.4f} | {intra} | {inter} |"
        )
    lines.append("")

    return "\n".join(lines)


# ── intra / inter SSIM 계산 ───────────────────────────────────────────────────
def compute_intra_inter_ssim(inp_data_list: list, n_inputs: int, n_seeds: int):
    """
    intra_input_ssim: 같은 입력의 seed 간 SSIM 평균
    inter_input_ssim: 다른 입력 간 seed_0 이미지의 SSIM 평균
    """
    # 모든 생성 이미지를 inp_idx × seed 매트릭스로 수집
    gen_imgs = {}  # (inp_idx, seed_offset) → Image
    out_base = OUT_DIR / "images"

    # 파일에서 이미지 로드
    for inp_data in inp_data_list:
        inp_idx = inp_data["inp_idx"]
        cls, dom = inp_data.get("cls"), inp_data.get("dom")
        for sd in inp_data["seeds"]:
            s = sd["seed_offset"]
            p = out_base / cls / dom / f"input_{inp_idx:02d}" / f"seed_{s:02d}.png"
            if p.exists():
                gen_imgs[(inp_idx, s)] = Image.open(p).convert("RGB")

    # intra: 같은 입력 내 seed 0~(n_seeds-1) 간 pairwise SSIM
    intra_vals = []
    for inp_data in inp_data_list:
        inp_idx = inp_data["inp_idx"]
        imgs = [gen_imgs.get((inp_idx, s)) for s in range(n_seeds)]
        imgs = [im for im in imgs if im is not None]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                intra_vals.append(ssim_pair(imgs[i], imgs[j]))

    # inter: 다른 입력 간 seed_0 이미지 pairwise SSIM
    inter_vals = []
    seed0_imgs = [gen_imgs.get((inp_data["inp_idx"], 0)) for inp_data in inp_data_list]
    seed0_imgs = [im for im in seed0_imgs if im is not None]
    for i in range(len(seed0_imgs)):
        for j in range(i + 1, len(seed0_imgs)):
            inter_vals.append(ssim_pair(seed0_imgs[i], seed0_imgs[j]))

    intra_mean = float(np.mean(intra_vals)) if intra_vals else None
    inter_mean = float(np.mean(inter_vals)) if inter_vals else None
    return intra_mean, inter_mean


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inputs", type=int, default=5, help="입력 이미지 수/cls×dom")
    parser.add_argument("--n_seeds",  type=int, default=5, help="seed 수/입력")
    parser.add_argument("--seed",     type=int, default=42, help="샘플링 random seed")
    parser.add_argument("--strength", type=float, default=STRENGTH)
    args = parser.parse_args()

    N_INPUTS  = args.n_inputs
    N_SEEDS   = args.n_seeds
    DENOISE   = args.strength
    n_total   = len(CLASSES) * len(DOMAINS) * N_INPUTS * N_SEEDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(exist_ok=True)

    print("=" * 60)
    print("Script 25: 다양성 재실험 (다중 입력 × 다중 seed)")
    print(f"  N_INPUTS={N_INPUTS}, N_SEEDS={N_SEEDS}, strength={DENOISE}")
    print(f"  총 생성 예정: {n_total}장")
    print("=" * 60)

    # ── 모듈/모델 로드 ────────────────────────────────────────────────────────
    print("\n[1/3] WBCRouter 및 CNN 로드...")
    mod     = load_router_module()
    WBCRouter = mod.WBCRouter
    build_prompt = mod.build_class_domain_prompt
    MULTI_CLASSES = mod.MULTI_CLASSES
    DOMAIN_LIST   = mod.DOMAINS       # ["domain_a_pbc", ...]
    get_device    = mod.get_device

    device  = get_device()
    router  = WBCRouter(router_ckpt=ROUTER_CKPT, cnn_ckpt=CNN_CKPT, device=device)

    # 파이프라인 워밍업
    print("  파이프라인 워밍업...")
    _sample_path = next((DATA_DIR / DOMAIN_LIST[0] / CLASSES[0]).glob("*.jpg"), None)
    if _sample_path:
        router.route(Image.open(_sample_path).convert("RGB"), generate=True, seed=99)

    cnn_model = load_cnn(device)
    print("  완료.")

    # ── 실험 루프 ─────────────────────────────────────────────────────────────
    print("\n[2/3] 이미지 생성 및 평가...")
    rng        = random.Random(args.seed)
    all_results = []
    done        = 0
    t0          = time.time()

    for cls in CLASSES:
        for dom in DOMAINS:
            dom_key  = DOMAIN_MAP[dom]   # "domain_a_pbc" 등
            dom_dir  = DATA_DIR / dom_key / cls
            files    = sorted(dom_dir.glob("*.jpg")) + sorted(dom_dir.glob("*.png"))
            if not files:
                print(f"  ⚠️  파일 없음: {dom_dir}")
                continue

            # N_INPUTS 장 random.sample (중복 없이)
            inputs = rng.sample(files, k=min(N_INPUTS, len(files)))

            print(f"\n── {cls} × {dom} ({len(inputs)}장 입력, 참조풀={len(files)}장) ──")

            # 도메인 인덱스 (라우터 프롬프트용)
            dom_idx   = DOMAIN_LIST.index(dom_key) if dom_key in DOMAIN_LIST else 0
            cls_idx   = MULTI_CLASSES.index(cls) if cls in MULTI_CLASSES else 0
            prompt    = build_prompt(cls_idx, dom_idx)

            inp_data_list = []
            for inp_idx, inp_path in enumerate(inputs):
                input_img = Image.open(inp_path).convert("RGB")

                # 저장 디렉토리
                inp_dir = OUT_DIR / "images" / cls / dom / f"input_{inp_idx:02d}"
                inp_dir.mkdir(parents=True, exist_ok=True)
                input_img.save(inp_dir / "input.png")

                seed_results = []
                gen_ssims    = []
                gen_correct  = []

                print(f"  [입력 {inp_idx+1}/{len(inputs)}] ", end="", flush=True)

                for s in range(N_SEEDS):
                    seed = inp_idx * 10 + s   # 0~4, 10~14, 20~24, ...
                    gen  = router._generate_once(input_img, prompt,
                                                 denoise=DENOISE, seed=seed)
                    gen.save(inp_dir / f"seed_{s:02d}.png")

                    # 평가
                    sv   = ssim_pair(gen, input_img)
                    pred_cls, conf = cnn_predict(cnn_model, gen, device)
                    ok   = (pred_cls == cls)

                    gen_ssims.append(sv)
                    gen_correct.append(ok)
                    seed_results.append({
                        "seed_offset": s,
                        "seed_value":  seed,
                        "ssim":        round(sv, 4),
                        "pred":        pred_cls,
                        "conf":        round(conf, 4),
                        "correct":     ok,
                    })
                    print("·", end="", flush=True)
                    done += 1

                # inp-level 집계
                cnn_acc   = float(np.mean(gen_correct))
                ssim_mean = float(np.mean(gen_ssims))
                print(f" CNN={cnn_acc:.0%} SSIM={ssim_mean:.4f} [{done}/{n_total}]")

                inp_data_list.append({
                    "inp_idx":  inp_idx,
                    "inp_path": str(inp_path),
                    "cls":      cls,
                    "dom":      dom,
                    "cnn_acc":  round(cnn_acc, 4),
                    "ssim_mean": round(ssim_mean, 4),
                    "seeds":    seed_results,
                })

            # cls×dom 수준 집계
            all_ssims   = [sd["ssim"] for inp in inp_data_list for sd in inp["seeds"]]
            all_correct = [sd["correct"] for inp in inp_data_list for sd in inp["seeds"]]
            overall_cnn  = float(np.mean(all_correct))
            overall_ssim = float(np.mean(all_ssims))

            # intra / inter SSIM (파일 경로 정보 주입)
            for inp in inp_data_list:
                inp["cls"] = cls
                inp["dom"] = dom
            intra_ssim, inter_ssim = compute_intra_inter_ssim(inp_data_list, N_INPUTS, N_SEEDS)

            all_results.append({
                "cls":    cls,
                "dom":    dom,
                "dom_key": dom_key,
                "n_inputs": len(inputs),
                "overall_cnn_acc":  round(overall_cnn,  4),
                "overall_ssim_mean": round(overall_ssim, 4),
                "intra_input_ssim_mean": round(intra_ssim, 4) if intra_ssim else None,
                "inter_input_ssim_mean": round(inter_ssim, 4) if inter_ssim else None,
                "inputs": inp_data_list,
            })

    elapsed = time.time() - t0
    print(f"\n생성 완료: {done}장, 경과 {elapsed/60:.1f}분")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    print("\n[3/3] 결과 저장...")

    summary = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_inputs":   N_INPUTS,
        "n_seeds":    N_SEEDS,
        "strength":   DENOISE,
        "seed":       args.seed,
        "n_total":    done,
        "elapsed_min": round(elapsed / 60, 1),
        "results":    all_results,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  summary.json 저장: {OUT_DIR / 'summary.json'}")

    # 갤러리 마크다운
    md = make_gallery(all_results, N_INPUTS, N_SEEDS)
    (OUT_DIR / "gallery.md").write_text(md, encoding="utf-8")
    print(f"  gallery.md 저장:   {OUT_DIR / 'gallery.md'}")

    # 간단 요약 출력
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"{'클래스':<12} {'도메인':<8} {'CNN acc':>8} {'SSIM':>7} {'Intra':>7} {'Inter':>7}")
    print("-" * 60)
    for r in all_results:
        intra = f"{r['intra_input_ssim_mean']:.4f}" if r.get("intra_input_ssim_mean") else "  —  "
        inter = f"{r['inter_input_ssim_mean']:.4f}" if r.get("inter_input_ssim_mean") else "  —  "
        print(
            f"  {r['cls']:<12} {r['dom']:<8} "
            f"{r['overall_cnn_acc']:>7.0%} "
            f"{r['overall_ssim_mean']:>7.4f} "
            f"{intra:>7} {inter:>7}"
        )


if __name__ == "__main__":
    main()
