"""
Script 20: img2img Strength 비교 실험
========================================
동일 입력 2장 (PBC basophil + AMC basophil) × strength [0.55, 0.65, 0.75] × seed 5개
= 총 30장 생성 → HTML + 마크다운 갤러리로 시각 비교

출력: results/strength_compare/
"""

import gc
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# ── WBCRouter 임포트 (숫자 시작 모듈) ──────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "router_inference",
    ROOT / "scripts" / "15_router_inference.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
WBCRouter    = mod.WBCRouter
ssim_pair    = mod.ssim_pair
MULTI_CLASSES = mod.MULTI_CLASSES

# ── 설정 ─────────────────────────────────────────────────────────────────
ROUTER_CKPT = ROOT / "models" / "dual_head_router.pt"
CNN_CKPT    = ROOT / "models" / "multidomain_cnn.pt"
OUT_DIR     = ROOT / "results" / "strength_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 비교할 strength 값
STRENGTHS = [0.35, 0.55, 0.65, 0.75]   # 0.35는 기존 베이스라인

# 각 strength별 seed 수
N_SEEDS = 5   # seed 0~4

# 입력 이미지 2장 (basophil_gallery에서 사용한 것과 동일)
INPUT_IMAGES = [
    {
        "path":   ROOT / "data/processed_multidomain/domain_a_pbc/basophil/pbc_basophil_000228.jpg",
        "domain": "PBC (Spain)",
        "label":  "input_pbc",
    },
    {
        "path":   ROOT / "data/processed_multidomain/domain_e_amc/basophil/amc_basophil_000003.jpg",
        "domain": "AMC (Korea)",
        "label":  "input_amc",
    },
]


# ── SSIM 유틸 ─────────────────────────────────────────────────────────────
def compute_ssim_list(gen_imgs, orig_img):
    return [round(ssim_pair(g, orig_img), 4) for g in gen_imgs]


# ── 마크다운 생성 ──────────────────────────────────────────────────────────
def make_markdown(results: list[dict]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# img2img Strength 비교 갤러리",
        "",
        f"> **생성 일시:** {ts}  ",
        f"> **Strengths:** {STRENGTHS}  ",
        f"> **Seeds/strength:** {N_SEEDS}  ",
        f"> **입력:** PBC basophil + AMC basophil",
        "",
        "---",
        "",
    ]

    for inp_res in results:
        label   = inp_res["label"]
        domain  = inp_res["domain"]
        inp_rel = inp_res["input_rel"]

        lines += [
            f"## {domain}",
            "",
            f"**입력 이미지:**",
            f'<img src="{inp_rel}" width="200" title="original">',
            "",
        ]

        for s_res in inp_res["strengths"]:
            strength = s_res["strength"]
            ssims    = s_res["ssim_list"]
            ssim_m   = round(float(np.mean(ssims)), 4)
            ssim_std = round(float(np.std(ssims)),  4)

            lines += [
                f"### strength = {strength}  "
                f"| SSIM: **{ssim_m}** ± {ssim_std}",
                "",
                "| seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |",
                "|--------|--------|--------|--------|--------|",
            ]

            cells = []
            for i, (rel, ssim_v) in enumerate(zip(s_res["gen_rels"], ssims)):
                cells.append(
                    f'<img src="{rel}" width="140" title="s={strength} seed={i} SSIM={ssim_v}">'
                )
            lines.append("| " + " | ".join(cells) + " |")

            # SSIM row
            ssim_row = []
            for sv in ssims:
                if sv >= 0.97:
                    emoji = "🟩"
                elif sv >= 0.95:
                    emoji = "🟨"
                elif sv >= 0.90:
                    emoji = "🟧"
                else:
                    emoji = "🟥"
                ssim_row.append(f"{emoji} {sv}")
            lines.append("| " + " | ".join(ssim_row) + " |")
            lines.append("")

        lines.append("---")
        lines.append("")

    # ── 종합 비교 테이블 ────────────────────────────────────────────────
    lines += [
        "## 종합 SSIM 비교",
        "",
        "| Strength | PBC SSIM mean ± std | AMC SSIM mean ± std | 전체 mean |",
        "|----------|---------------------|---------------------|-----------|",
    ]

    for si, strength in enumerate(STRENGTHS):
        pbc_ssims = results[0]["strengths"][si]["ssim_list"]
        amc_ssims = results[1]["strengths"][si]["ssim_list"]
        all_ssims = pbc_ssims + amc_ssims
        pbc_m = round(float(np.mean(pbc_ssims)), 4)
        pbc_s = round(float(np.std(pbc_ssims)),  4)
        amc_m = round(float(np.mean(amc_ssims)), 4)
        amc_s = round(float(np.std(amc_ssims)),  4)
        all_m = round(float(np.mean(all_ssims)), 4)
        lines.append(
            f"| **{strength}** | {pbc_m} ± {pbc_s} | {amc_m} ± {amc_s} | **{all_m}** |"
        )

    lines += [
        "",
        "## 해석",
        "",
        "| Strength | 예상 효과 |",
        "|----------|-----------|",
        "| 0.35 | 입력 보존 (baseline) — 합성 다양성 낮음 |",
        "| 0.55 | 배경·스테이닝 변화 + 세포 외곽 변형 시작 |",
        "| 0.65 | 핵 형태 일부 변화 + 과립 분포 변형 |",
        "| 0.75 | 강한 형태 변화 — CNN 정확도 하락 위험 |",
        "",
    ]

    return "\n".join(lines)


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Script 20: img2img Strength 비교 실험")
    print(f"  Strengths: {STRENGTHS}")
    print(f"  Seeds/strength: {N_SEEDS}")
    print(f"  입력 수: {len(INPUT_IMAGES)}")
    print(f"  총 생성: {len(STRENGTHS) * N_SEEDS * len(INPUT_IMAGES)}장")
    print("=" * 60)

    # ── 라우터 초기화 (파이프라인 포함) ────────────────────────────────
    print("\n[1/3] WBCRouter 초기화...")
    router = WBCRouter(
        router_ckpt=ROUTER_CKPT,
        cnn_ckpt=CNN_CKPT,
        device=None,   # 자동 감지
    )

    all_results = []

    for inp_cfg in INPUT_IMAGES:
        inp_path = Path(inp_cfg["path"])
        label    = inp_cfg["label"]
        domain   = inp_cfg["domain"]

        # 출력 디렉토리
        inp_out = OUT_DIR / label
        inp_out.mkdir(exist_ok=True)

        print(f"\n[2/3] 처리 중: {domain} ({inp_path.name})")

        # 원본 이미지 저장
        img = Image.open(inp_path).convert("RGB")
        inp_save = inp_out / "input.png"
        img.save(inp_save)

        # 라우터 분류 (1장 생성으로 파이프라인도 함께 로드)
        print("  분류 + 파이프라인 로드 (seed=99 생성 1장)...")
        route_res = router.route(img, generate=True, seed=99)
        pred_cls  = route_res["class_name"]     # key: class_name (not pred_class)
        cls_conf  = route_res["class_conf"]
        prompt    = route_res["prompt"]
        print(f"  분류: {pred_cls} (conf={cls_conf:.4f})")
        print(f"  프롬프트: {prompt[:80]}...")

        strength_results = []

        for strength in STRENGTHS:
            print(f"\n  strength={strength} 생성 중 (seed 0~{N_SEEDS-1})...")
            s_dir = inp_out / f"s{int(strength*100):03d}"
            s_dir.mkdir(exist_ok=True)

            gen_imgs = []
            gen_rels = []

            for seed in range(N_SEEDS):
                gen = router._generate_once(img, prompt, denoise=strength, seed=seed)
                fname = f"gen_{seed:02d}.png"
                gen.save(s_dir / fname)
                gen_imgs.append(gen)
                gen_rels.append(f"{label}/s{int(strength*100):03d}/{fname}")
                print(f"    seed={seed} ✓", end="", flush=True)
            print()

            ssims = compute_ssim_list(gen_imgs, img)
            print(f"    SSIM: {ssims} → mean={round(float(np.mean(ssims)), 4)}")

            strength_results.append({
                "strength": strength,
                "ssim_list": ssims,
                "gen_rels": gen_rels,
            })

        all_results.append({
            "label":      label,
            "domain":     domain,
            "input_rel":  f"{label}/input.png",
            "pred_class": pred_cls,
            "class_conf": round(cls_conf, 4),
            "prompt":     prompt,
            "strengths":  strength_results,
        })

    # ── 마크다운 저장 ───────────────────────────────────────────────────
    print("\n[3/3] 갤러리 마크다운 저장...")
    md = make_markdown(all_results)
    md_path = OUT_DIR / "gallery.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  → {md_path}")

    # ── JSON 저장 ────────────────────────────────────────────────────────
    summary = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strengths":  STRENGTHS,
        "n_seeds":    N_SEEDS,
        "inputs":     all_results,
    }
    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → {json_path}")

    print("\n" + "=" * 60)
    print("✅ 완료!")
    print(f"  결과 디렉토리: {OUT_DIR}")
    print(f"  갤러리:        {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
