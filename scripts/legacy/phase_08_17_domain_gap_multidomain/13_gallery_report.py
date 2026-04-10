"""
WBC 생성 이미지 갤러리 Markdown 보고서 생성
===========================================
3가지 모델의 basophil 생성 이미지를 갤러리 형태로 정리하고,
원본 데이터셋과의 유사도 차이를 비교 분석한다.

모델:
  1. 단일도메인 DreamBooth     (results/.../basophil/)
  2. 멀티도메인 DreamBooth     (results/.../multidomain_basophil/)
  3. 멀티도메인 T2I LoRA       (results/.../t2i_multidomain_basophil/)

출력:
  results/generation_test/basophil/gallery_report.md

Usage:
    python scripts/legacy/phase_08_17_domain_gap_multidomain/13_gallery_report.py
    python scripts/legacy/phase_08_17_domain_gap_multidomain/13_gallery_report.py --class_name basophil --cols 4
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
MULTI_DATA    = ROOT / "data" / "processed_multidomain"
GEN_ROOT      = ROOT / "results" / "generation_test"

DOMAIN_LABELS = {
    "domain_a_pbc":    "PBC · Barcelona (Spain) · May-Grünwald Giemsa / CellaVision",
    "domain_b_raabin": "Raabin · Iran · Giemsa / 스마트폰 현미경",
    "domain_c_mll23":  "MLL23 · Germany · Pappenheim / Metafer scanner",
    "domain_e_amc":    "AMC · Korea · Romanowsky / miLab analyzer",
}
DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]

# ── 유틸: SSIM (test_generation_similarity.py에서 복사) ───────────────

def ssim_pair(img_a: Image.Image, img_b: Image.Image, size: int = 224) -> float:
    a = np.array(img_a.resize((size, size)).convert("L")).astype(float) / 255.
    b = np.array(img_b.resize((size, size)).convert("L")).astype(float) / 255.
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(), b.std()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = 0.01**2, 0.03**2
    return float((2*mu_a*mu_b + C1) * (2*sig_ab + C2) /
                 ((mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2)))

def mse_pair(img_a: Image.Image, img_b: Image.Image, size: int = 224) -> float:
    a = np.array(img_a.resize((size, size))).astype(float) / 255.
    b = np.array(img_b.resize((size, size))).astype(float) / 255.
    return float(np.mean((a - b) ** 2))

def sharpness(pil_img: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── 데이터 로딩 ───────────────────────────────────────────────────────

def load_dreambooth_report(report_path: Path) -> dict:
    """DreamBooth report.json 로드 (aggregate + per_image)."""
    with open(report_path) as f:
        data = json.load(f)
    return data


def parse_t2i_domain(filename: str) -> str:
    """t2i_basophil_domain_a_pbc_0000.png → domain_a_pbc"""
    for d in DOMAINS:
        if d in filename:
            return d
    return "unknown"


# ── T2I per-image 유사도 계산 ─────────────────────────────────────────

def compute_t2i_per_image_metrics(
    gen_paths: list,
    class_name: str,
    n_pool: int = 50,
    seed: int = 42,
) -> list:
    """
    T2I 생성 이미지마다 해당 도메인 원본과 NN-SSIM / MSE 계산.
    DreamBooth report.json의 per_image 구조를 맞춤.
    """
    rng = random.Random(seed)
    # 도메인별 원본 pool 미리 로딩
    domain_pool = {}
    for domain in DOMAINS:
        pool_dir = MULTI_DATA / domain / class_name
        imgs = sorted(pool_dir.glob("*.jpg")) + sorted(pool_dir.glob("*.png"))
        sample = rng.sample(imgs, min(n_pool, len(imgs)))
        domain_pool[domain] = [Image.open(p).convert("RGB") for p in sample]

    per_image = []
    for i, gen_path in enumerate(sorted(gen_paths)):
        domain = parse_t2i_domain(gen_path.name)
        gen_img = Image.open(gen_path).convert("RGB")
        sharp = sharpness(gen_img)

        pool = domain_pool.get(domain, [])
        if pool:
            ssim_scores = [ssim_pair(gen_img, r) for r in pool]
            mse_scores  = [mse_pair(gen_img, r)  for r in pool]
            best_ssim = max(ssim_scores)
            best_mse  = min(mse_scores)
        else:
            best_ssim = float("nan")
            best_mse  = float("nan")

        per_image.append({
            "idx":           i,
            "file":          gen_path.name,
            "domain":        domain,
            "sharpness_gen": round(sharp, 2),
            "ssim_nearest":  round(best_ssim, 4) if not np.isnan(best_ssim) else None,
            "mse_nearest":   round(best_mse,  6) if not np.isnan(best_mse)  else None,
            "cnn_conf_correct": None,
            "cnn_pred_class":   "N/A (T2I)",
            "cnn_correct":      None,
        })
    return per_image


# ── Markdown 렌더링 헬퍼 ──────────────────────────────────────────────

def rel(path: Path, base: Path) -> str:
    """base 기준 상대 경로 문자열 반환 (상위 디렉터리 포함)."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        # 상위 디렉터리를 거슬러 올라가야 하는 경우 os.path.relpath 사용
        return os.path.relpath(str(path), str(base))


def img_tag(rel_path: str, width: int = 180) -> str:
    return f'<img src="{rel_path}" width="{width}">'


def cnn_badge(cnn_correct, conf) -> str:
    if cnn_correct is None:
        return "—"
    mark = "✅" if cnn_correct else "❌"
    conf_str = f"{conf:.3f}" if conf is not None else "N/A"
    return f"{mark} {conf_str}"


def render_summary_table(models_data: dict) -> str:
    """3모델 × 5지표 요약 비교표."""
    rows = [
        "| 모델 | 방식 | FD↓ | NN CosSim↑ | SSIM↑ | CNN Acc↑ | CNN Conf↑ |",
        "|------|------|-----|-----------|-------|---------|---------|",
    ]
    for label, d in models_data.items():
        agg = d["aggregate"]
        fd   = f"{agg.get('frechet_distance', 'N/A'):.2f}"      if isinstance(agg.get('frechet_distance'), float) else "N/A"
        cos  = f"{agg.get('nn_cosine_similarity_mean', agg.get('nn_cosine_sim', 'N/A')):.4f}" \
               if isinstance(agg.get('nn_cosine_similarity_mean', agg.get('nn_cosine_sim')), float) else "N/A"
        ssim = f"{agg.get('ssim_mean', 'N/A'):.4f}"             if isinstance(agg.get('ssim_mean'), float) else "—"
        acc  = f"{agg.get('cnn_accuracy', agg.get('cnn_accuracy_rate', 'N/A'))*100:.1f}%" \
               if isinstance(agg.get('cnn_accuracy', agg.get('cnn_accuracy_rate')), float) else "N/A"
        conf = f"{agg.get('cnn_conf_mean', agg.get('cnn_confidence_mean', 'N/A')):.4f}" \
               if isinstance(agg.get('cnn_conf_mean', agg.get('cnn_confidence_mean')), float) else "N/A"
        method = d.get("method", "—")
        rows.append(f"| {label} | {method} | {fd} | {cos} | {ssim} | {acc} | {conf} |")
    return "\n".join(rows)


def render_gallery_table(
    gen_paths: list,
    per_image: list,
    base_dir: Path,
    cols: int = 4,
    width: int = 180,
) -> str:
    """이미지 갤러리 테이블 (cols열) + 이미지별 SSIM/CNN 배지."""
    sorted_paths = sorted(gen_paths)
    # per_image를 파일명 기준 dict로
    pm = {d["file"]: d for d in per_image}

    header = "| " + " | ".join(["이미지"] * cols) + " |"
    sep    = "| " + " | ".join(["---"] * cols) + " |"
    lines  = [header, sep]

    chunks = [sorted_paths[i:i+cols] for i in range(0, len(sorted_paths), cols)]
    for chunk in chunks:
        img_row  = []
        info_row = []
        for p in chunk:
            rp = rel(p, base_dir)
            m  = pm.get(p.name, {})
            ssim_str = f"SSIM: {m['ssim_nearest']:.3f}" if m.get("ssim_nearest") is not None else "SSIM: —"
            cnn_str  = cnn_badge(m.get("cnn_correct"), m.get("cnn_conf_correct"))
            img_row.append(img_tag(rp, width))
            info_row.append(f"`{p.name}`<br>{ssim_str}<br>{cnn_str}")
        # 빈 칸 채우기
        while len(img_row) < cols:
            img_row.append(""); info_row.append("")
        lines.append("| " + " | ".join(img_row)  + " |")
        lines.append("| " + " | ".join(info_row) + " |")

    return "\n".join(lines)


def render_stats_table(per_image: list, has_cnn: bool = True) -> str:
    """이미지별 상세 지표 테이블."""
    if has_cnn:
        header = "| # | 파일명 | Sharpness | SSIM↑ | MSE↓ | CNN 예측 | Confidence |"
        sep    = "|---|--------|-----------|-------|------|---------|------------|"
    else:
        header = "| # | 파일명 | 도메인 | Sharpness | SSIM↑ (NN) | MSE↓ (NN) |"
        sep    = "|---|--------|--------|-----------|-----------|----------|"
    lines = [header, sep]
    for m in per_image:
        sharp = f"{m['sharpness_gen']:.2f}" if m.get("sharpness_gen") is not None else "—"
        ssim  = f"{m['ssim_nearest']:.4f}"  if m.get("ssim_nearest") is not None else "—"
        mse   = f"{m['mse_nearest']:.5f}"   if m.get("mse_nearest") is not None else "—"
        if has_cnn:
            pred = m.get("cnn_pred_class", "—")
            conf = f"{m['cnn_conf_correct']:.4f}" if m.get("cnn_conf_correct") is not None else "—"
            ok   = "✅" if m.get("cnn_correct") else ("❌" if m.get("cnn_correct") is False else "—")
            lines.append(f"| {m['idx']} | `{m['file']}` | {sharp} | {ssim} | {mse} | {ok} {pred} | {conf} |")
        else:
            domain = m.get("domain", "—")
            lines.append(f"| {m['idx']} | `{m['file']}` | {domain} | {sharp} | {ssim} | {mse} |")
    return "\n".join(lines)


def render_original_samples(class_name: str, base_dir: Path, n: int = 5, seed: int = 42) -> str:
    """4개 도메인별 원본 이미지 샘플 갤러리."""
    rng = random.Random(seed)
    lines = []
    for domain in DOMAINS:
        pool_dir = MULTI_DATA / domain / class_name
        imgs = sorted(pool_dir.glob("*.jpg")) + sorted(pool_dir.glob("*.png"))
        sample = rng.sample(imgs, min(n, len(imgs)))
        label = DOMAIN_LABELS.get(domain, domain)
        lines.append(f"\n### {label}")
        lines.append(f"총 {len(imgs)}장 중 {len(sample)}장 샘플\n")
        row = " | ".join([img_tag(rel(p, base_dir), 160) for p in sample])
        lines.append("| " + row + " |")
        lines.append("| " + " | ".join(["---"] * len(sample)) + " |")
    return "\n".join(lines)


def render_t2i_domain_sections(
    gen_paths: list,
    per_image: list,
    base_dir: Path,
    width: int = 180,
) -> str:
    """T2I 생성 이미지를 도메인별 4개 섹션으로 분리."""
    pm = {d["file"]: d for d in per_image}
    lines = []
    for domain in DOMAINS:
        label = DOMAIN_LABELS.get(domain, domain)
        domain_paths = [p for p in sorted(gen_paths) if domain in p.name]
        if not domain_paths:
            continue
        lines.append(f"\n#### {label}")
        lines.append(f"{len(domain_paths)}장 생성\n")

        img_row  = []
        info_row = []
        for p in domain_paths:
            rp = rel(p, base_dir)
            m  = pm.get(p.name, {})
            ssim_str = f"SSIM: {m['ssim_nearest']:.3f}" if m.get("ssim_nearest") is not None else "SSIM: —"
            img_row.append(img_tag(rp, width))
            info_row.append(f"`{p.name}`<br>{ssim_str}")
        lines.append("| " + " | ".join(img_row)  + " |")
        lines.append("| " + " | ".join(["---"] * len(img_row)) + " |")
        lines.append("| " + " | ".join(info_row) + " |")
    return "\n".join(lines)


# ── 메인 보고서 조립 ──────────────────────────────────────────────────

def build_gallery_report(args) -> str:
    base_dir  = GEN_ROOT / args.class_name
    gen_root  = GEN_ROOT / args.class_name

    # ── 1. DreamBooth 보고서 로드 ─────────────────────────────────
    db_single_report = load_dreambooth_report(
        gen_root / "basophil" / "report.json"
    )
    db_multi_report  = load_dreambooth_report(
        gen_root / "multidomain_basophil" / "report.json"
    )
    t2i_report = load_dreambooth_report(
        gen_root / "t2i_multidomain_basophil" / "report.json"
    )

    db_single_per = db_single_report["per_image"]
    db_multi_per  = db_multi_report["per_image"]
    db_single_agg = db_single_report["aggregate"]
    db_multi_agg  = db_multi_report["aggregate"]
    t2i_agg       = t2i_report["aggregate"]

    # ── 2. 생성 이미지 경로 수집 ──────────────────────────────────
    db_single_paths = sorted((gen_root / "basophil" / "generated").glob("*.png"))
    db_multi_paths  = sorted((gen_root / "multidomain_basophil" / "generated").glob("*.png"))
    t2i_paths       = sorted((gen_root / "t2i_multidomain_basophil" / "generated").glob("*.png"))

    # ── 3. T2I per-image 유사도 신규 계산 ────────────────────────
    print("[1/3] T2I per-image 유사도 계산 중 (SSIM NN)...")
    t2i_per = compute_t2i_per_image_metrics(
        t2i_paths, args.class_name, n_pool=args.ssim_pool
    )
    t2i_ssim_mean = float(np.mean([m["ssim_nearest"] for m in t2i_per if m["ssim_nearest"] is not None]))
    t2i_mse_mean  = float(np.mean([m["mse_nearest"]  for m in t2i_per if m["mse_nearest"]  is not None]))
    print(f"   T2I SSIM(NN 평균): {t2i_ssim_mean:.4f}  MSE(NN 평균): {t2i_mse_mean:.5f}")

    # ── 4. 요약 테이블용 데이터 준비 ──────────────────────────────
    models_data = {
        "단일도메인 DreamBooth": {
            "aggregate": db_single_agg,
            "method": "img2img (denoise=0.35)",
        },
        "멀티도메인 DreamBooth": {
            "aggregate": db_multi_agg,
            "method": "img2img (denoise=0.35)",
        },
        "멀티도메인 T2I LoRA": {
            "aggregate": {
                **t2i_agg,
                "ssim_mean": t2i_ssim_mean,    # 신규 계산값 추가
            },
            "method": "text-to-image (pure)",
        },
    }

    # ── 5. Markdown 조립 ─────────────────────────────────────────
    print("[2/3] Markdown 보고서 조립 중...")

    sections = []

    # ── 헤더 ─────────────────────────────────────────────────────
    sections.append(f"""# WBC {args.class_name.capitalize()} 생성 이미지 갤러리 보고서

> 생성 일시: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
> 대상 클래스: **{args.class_name}**
> 생성 모델: 단일도메인 DreamBooth · 멀티도메인 DreamBooth · 멀티도메인 T2I LoRA
> 기반 모델: `stabilityai/stable-diffusion-xl-base-1.0`
> 학습 해상도: 256×256 → SDXL VAE 32×32 latent
> LoRA Rank: 8 / Steps: 400

---
""")

    # ── 0. 요약 비교표 ────────────────────────────────────────────
    sections.append(f"""## 0. 모델별 종합 성능 비교

{render_summary_table(models_data)}

> **FD**: Fréchet Distance (↓ 낮을수록 실제 분포에 가깝다)
> **NN CosSim**: EfficientNet-B0 임베딩 공간의 Nearest-Neighbor 코사인 유사도
> **SSIM**: 구조적 유사도 (생성 이미지 ↔ NN 원본)
> **CNN Acc/Conf**: PBC 학습 분류기(EfficientNet-B0)의 정확도 / 신뢰도
>
> ⚠️ **방법론 차이 주의**: DreamBooth는 실제 이미지를 img2img 참조(strength=0.35)로 사용하므로
> 원본 분포와의 거리(FD)가 구조적으로 낮다. T2I는 참조 없이 순수 텍스트→이미지 생성이므로
> 동일 지표로 직접 우열 비교는 부적절하다.

---
""")

    # ── 1. 단일도메인 DreamBooth ──────────────────────────────────
    grid_path = gen_root / "basophil" / "grid.png"
    grid_tag  = img_tag(rel(grid_path, base_dir), 600) if grid_path.exists() else "*(grid 없음)*"

    sections.append(f"""## 1. 단일도메인 DreamBooth

> 학습 데이터: PBC Barcelona (MGG 염색) basophil {db_single_agg['n_generated']}장
> 방식: img2img (denoise strength = {db_single_agg['denoise_strength']})
> 지표: FD={db_single_agg['frechet_distance']:.2f} · SSIM={db_single_agg['ssim_mean']:.4f} · CNN Acc={db_single_agg['cnn_accuracy']*100:.0f}%

### 비교 그리드

{grid_tag}

### 생성 이미지 갤러리

{render_gallery_table(db_single_paths, db_single_per, base_dir, cols=args.cols)}

### 이미지별 상세 지표

{render_stats_table(db_single_per, has_cnn=True)}

---
""")

    # ── 2. 멀티도메인 DreamBooth ──────────────────────────────────
    grid_multi = gen_root / "multidomain_basophil" / "grid.png"
    grid_multi_tag = img_tag(rel(grid_multi, base_dir), 600) if grid_multi.exists() else "*(grid 없음)*"

    sections.append(f"""## 2. 멀티도메인 DreamBooth

> 학습 데이터: 4개 도메인(PBC·Raabin·MLL23·AMC) 혼합, 도메인별 조건부 프롬프트
> 방식: img2img (denoise strength = {db_multi_agg['denoise_strength']})
> 지표: FD={db_multi_agg['frechet_distance']:.2f} · SSIM={db_multi_agg['ssim_mean']:.4f} · CNN Acc={db_multi_agg['cnn_accuracy']*100:.0f}%
> **vs 단일도메인**: FD {((db_multi_agg['frechet_distance']-db_single_agg['frechet_distance'])/db_single_agg['frechet_distance']*100):+.1f}% · SSIM {((db_multi_agg['ssim_mean']-db_single_agg['ssim_mean'])/db_single_agg['ssim_mean']*100):+.1f}%

### 비교 그리드

{grid_multi_tag}

### 생성 이미지 갤러리

{render_gallery_table(db_multi_paths, db_multi_per, base_dir, cols=args.cols)}

### 이미지별 상세 지표

{render_stats_table(db_multi_per, has_cnn=True)}

---
""")

    # ── 3. T2I LoRA ───────────────────────────────────────────────
    comp_grid = gen_root / "t2i_multidomain_basophil" / "comparison_grid.png"
    comp_tag  = img_tag(rel(comp_grid, base_dir), 700) if comp_grid.exists() else "*(grid 없음)*"

    sections.append(f"""## 3. 멀티도메인 T2I LoRA

> 학습 방식: `train_text_to_image_lora_sdxl.py` (순수 text→image LoRA, 참조 이미지 없음)
> 학습 데이터: 4개 도메인 727장 (domain_a_pbc:200·domain_b_raabin:200·domain_c_mll23:200·domain_e_amc:127)
> Steps: 400 · Rank: 8 · LR: 1e-4 (cosine warmup 50 steps)
> 지표: FD={t2i_agg['frechet_distance']:.2f} · NN SSIM={t2i_ssim_mean:.4f} · CNN Acc={t2i_agg['cnn_accuracy_rate']*100:.0f}%
>
> 💡 **T2I 특성**: 생성 시 실제 이미지 참조 없이 도메인별 캡션 프롬프트만으로 생성.
> 각 도메인 원본 50장 pool과의 NN-SSIM으로 유사도를 계산함.

### 원본 vs T2I 생성 비교 그리드

{comp_tag}

### 도메인별 생성 이미지

{render_t2i_domain_sections(t2i_paths, t2i_per, base_dir)}

### 이미지별 상세 지표 (NN-SSIM, MSE)

{render_stats_table(t2i_per, has_cnn=False)}

---
""")

    # ── 4. 원본 데이터셋 샘플 ─────────────────────────────────────
    sections.append(f"""## 4. 원본 데이터셋 샘플

각 도메인 원본 basophil 이미지 {args.n_real_samples}장 (무작위 샘플)

{render_original_samples(args.class_name, base_dir, n=args.n_real_samples)}

---
""")

    # ── 5. 유사도 차이 요약 ───────────────────────────────────────
    db_s_ssim = db_single_agg['ssim_mean']
    db_m_ssim = db_multi_agg['ssim_mean']
    t2i_ssim  = t2i_ssim_mean
    db_s_fd   = db_single_agg['frechet_distance']
    db_m_fd   = db_multi_agg['frechet_distance']
    t2i_fd    = t2i_agg['frechet_distance']

    sections.append(f"""## 5. 원본 데이터셋과의 유사도 분석

### SSIM (구조적 유사도) — 높을수록 원본에 가깝다

| 모델 | SSIM | 원본 대비 |
|------|------|---------|
| 단일도메인 DreamBooth | **{db_s_ssim:.4f}** | 기준 |
| 멀티도메인 DreamBooth | **{db_m_ssim:.4f}** | {((db_m_ssim-db_s_ssim)/db_s_ssim*100):+.2f}% |
| 멀티도메인 T2I LoRA   | **{t2i_ssim:.4f}** | {((t2i_ssim-db_s_ssim)/db_s_ssim*100):+.2f}% |

> DreamBooth SSIM: img2img 원본 참조 이미지와의 SSIM (같은 이미지를 변형하므로 높은 경향)
> T2I SSIM: 각 도메인 원본 50장 pool에서의 NN-SSIM (참조 없이 생성 후 비교)

### Fréchet Distance — 낮을수록 실제 분포에 가깝다

| 모델 | FD | 원본 대비 |
|------|-----|---------|
| 단일도메인 DreamBooth | **{db_s_fd:.2f}** | 기준 |
| 멀티도메인 DreamBooth | **{db_m_fd:.2f}** | {((db_m_fd-db_s_fd)/db_s_fd*100):+.1f}% |
| 멀티도메인 T2I LoRA   | **{t2i_fd:.2f}** | {((t2i_fd-db_s_fd)/db_s_fd*100):+.1f}% |

> FD는 EfficientNet-B0(PBC 학습) penultimate layer 특성 공간에서의 Fréchet Distance.
> T2I FD가 높은 이유: 분포 참조 없는 순수 생성 + 400 steps의 짧은 학습.
> Steps 증가 또는 text encoder LoRA 함께 학습 시 개선 예상.

### CNN 분류 성능

| 모델 | Accuracy | Confidence |
|------|---------|------------|
| 단일도메인 DreamBooth | {db_single_agg['cnn_accuracy']*100:.0f}% | {db_single_agg['cnn_conf_mean']:.4f} |
| 멀티도메인 DreamBooth | {db_multi_agg['cnn_accuracy']*100:.0f}% | {db_multi_agg['cnn_conf_mean']:.4f} |
| 멀티도메인 T2I LoRA   | {t2i_agg['cnn_accuracy_rate']*100:.0f}% | {t2i_agg['cnn_confidence_mean']:.4f} |

> PBC-trained EfficientNet-B0 기준 분류 성능.
> 단일도메인 DreamBooth 실패 케이스: idx=11 (lymphocyte로 오분류, conf=0.041)

---

## 6. 방법론 노트

### 평가 방법 비교

| 항목 | DreamBooth (단일/멀티) | T2I LoRA |
|------|----------------------|---------|
| 생성 방식 | img2img (실제 이미지 참조) | text-to-image (프롬프트만) |
| denoise strength | 0.35 | N/A (전체 denoising) |
| 원본 유사도 측정 | SSIM vs 참조 이미지 | NN-SSIM vs 도메인 pool |
| 학습 스크립트 | `train_dreambooth_lora_sdxl.py` | `train_text_to_image_lora_sdxl.py` |
| `--instance_prompt` | 필요 | 불필요 |
| 캡션 방식 | per-image `metadata.jsonl` | per-image `metadata.jsonl` |

### 재현 방법

```bash
# 단일도메인 DreamBooth 평가 재현
python scripts/legacy/shared_support/test_generation_similarity.py \\
    --lora_dir lora/weights/basophil --class_name basophil --n_gen 20

# 멀티도메인 DreamBooth 평가 재현
python scripts/legacy/shared_support/test_generation_similarity.py \\
    --lora_dir lora/weights/multidomain_basophil --class_name basophil --n_gen 20

# T2I LoRA 평가 재현
python scripts/legacy/phase_08_17_domain_gap_multidomain/12_t2i_lora_eval.py --class_name basophil --n_gen 20

# 이 갤러리 보고서 재생성
python scripts/legacy/phase_08_17_domain_gap_multidomain/13_gallery_report.py --class_name basophil
```
""")

    return "\n".join(sections)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WBC 생성 이미지 갤러리 Markdown 보고서 생성"
    )
    parser.add_argument("--class_name",    type=str, default="basophil")
    parser.add_argument("--cols",          type=int, default=4,
                        help="갤러리 열 수 (기본: 4)")
    parser.add_argument("--n_real_samples",type=int, default=5,
                        help="도메인별 원본 샘플 수 (기본: 5)")
    parser.add_argument("--ssim_pool",     type=int, default=50,
                        help="T2I SSIM 계산용 원본 pool 크기 (기본: 50)")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    print(f"=== WBC 갤러리 보고서 생성: {args.class_name} ===")

    md = build_gallery_report(args)

    out_dir = GEN_ROOT / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gallery_report.md"

    print(f"[3/3] 저장 중: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    lines = md.count("\n")
    print(f"완료: {out_path}  ({lines}줄, {len(md)//1024}KB)")
    print(f"VSCode Preview: Cmd+Shift+V 또는 Cmd+K V")


if __name__ == "__main__":
    main()
