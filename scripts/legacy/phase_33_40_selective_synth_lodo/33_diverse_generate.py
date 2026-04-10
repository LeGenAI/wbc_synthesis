"""
Script 33: 멀티축 다양화 혈구 이미지 대량 생성
=============================================
멀티도메인 LoRA를 사용하여 4가지 다양성 축으로 합성 혈구 이미지를 생성.

다양성 축:
  1. 참조 도메인     (4개): PBC / Raabin / MLL23 / AMC   — morphology source
  2. 목표 스타일 도메인(1~4개): prompt로 지정하는 stain/scanner style
  3. Denoise 강도    (3개): 0.25 / 0.35 / 0.45           — 원본 구조 보존 수준
  4. 프롬프트 템플릿  (4개): 표준 / 배율 / 병리학 / 세포학 — 텍스트 조건 다양화
  5. 랜덤 시드       (N개): 확률적 노이즈 다양화

기본 설정 (단일 클래스, n_per_domain=3, n_seeds=2):
  4도메인 × 3입력 × 3강도 × 4템플릿 × 2시드 = 288장

출력:
  data/generated_diverse/{class}/{domain}/ds{ds_tag}/tpl{t}_s{s}_{idx:04d}.png
  results/diverse_generation/{class}/
    ├── report.json         (집계 + 이미지별 지표)
    ├── summary.md          (다양성 축별 요약 테이블)
    └── grid_{domain}.png   (도메인별 비교 그리드)

Usage:
    # basophil (기본값, 288장)
    python scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py --class_name basophil

    # 빠른 테스트 (48장)
    python scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py --class_name basophil --n_per_domain 1 --n_seeds 2

    # 모든 클래스
    python scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py --all_classes

    # 구조만 확인 (생성 없음)
    python scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py --class_name basophil --dry_run
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import models, transforms
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "processed_multidomain"
LORA_ROOT = ROOT / "lora" / "weights"
GEN_DIR   = ROOT / "data" / "generated_diverse"
OUT_ROOT  = ROOT / "results" / "diverse_generation"
CNN_CKPT  = ROOT / "models" / "multidomain_cnn.pt"   # 5-class EfficientNet-B0

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# ── 메타데이터 ─────────────────────────────────────────────────────────
CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]

DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_SHORT = {
    "domain_a_pbc":    "PBC",
    "domain_b_raabin": "Raabin",
    "domain_c_mll23":  "MLL23",
    "domain_e_amc":    "AMC",
}
DOMAIN_PROMPTS = {
    "domain_a_pbc":    "May-Grünwald Giemsa stain, CellaVision automated analyzer, Barcelona Spain",
    "domain_b_raabin": "Giemsa stain, smartphone microscope camera, Iran hospital",
    "domain_c_mll23":  "Pappenheim stain, Metafer scanner, Germany clinical lab",
    "domain_e_amc":    "Romanowsky stain, miLab automated analyzer, South Korea AMC",
}
CLASS_MORPHOLOGY = {
    "basophil":   "bilobed nucleus with dark purple-black granules filling cytoplasm",
    "eosinophil": "bilobed nucleus with bright orange-red granules",
    "lymphocyte": "large round nucleus with scant agranular cytoplasm",
    "monocyte":   "kidney-shaped or folded nucleus with grey-blue cytoplasm",
    "neutrophil": "multilobed nucleus with pale pink granules",
}
CLASS_CELL_TERMS = {
    "basophil": "basophilic leukocyte",
    "eosinophil": "eosinophilic leukocyte",
    "lymphocyte": "lymphocyte",
    "monocyte": "monocyte",
    "neutrophil": "neutrophilic leukocyte",
}

# ── 다양성 축 설정 ─────────────────────────────────────────────────────
DEFAULT_DENOISE_STRENGTHS = [0.25, 0.35, 0.45]

NEGATIVE_PROMPT = (
    "cartoon, illustration, text, watermark, multiple cells, "
    "heavy artifacts, unrealistic colors, deformed nucleus, blurry"
)

# 4가지 프롬프트 템플릿
def _tpl0(cls: str, morph: str, dom: str) -> str:
    """표준: 기존 고정 프롬프트 (기준선)"""
    return (
        f"microscopy image of a single {cls} white blood cell, "
        f"peripheral blood smear, {morph}, "
        f"{dom}, sharp focus, realistic, clinical lab imaging"
    )

def _tpl1(cls: str, morph: str, dom: str) -> str:
    """배율 강조: 100x oil immersion"""
    return (
        f"100x oil immersion microscopy, single {cls} leukocyte, "
        f"{morph}, {dom}, "
        f"high-resolution hematology imaging, sharp details"
    )

def _tpl2(cls: str, morph: str, dom: str) -> str:
    """병리학 관점: clinical hematology"""
    return (
        f"clinical hematology, {cls} white blood cell, "
        f"peripheral blood smear analysis, {morph}, {dom}, "
        f"bright-field microscopy, detailed nuclear morphology"
    )

def _tpl3(cls: str, morph: str, dom: str) -> str:
    """세포학 관점: class-aware cytology"""
    cell_term = CLASS_CELL_TERMS.get(cls, f"{cls} leukocyte")
    return (
        f"cytology preparation, isolated {cell_term}, "
        f"{morph}, blood film, {dom}, "
        f"professional microscopic imaging, clinical diagnostic"
    )

PROMPT_TEMPLATES = [_tpl0, _tpl1, _tpl2, _tpl3]
TEMPLATE_NAMES   = ["standard", "oil_immersion", "clinical_hematology", "cytology"]

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def get_target_domains(ref_domain: str, mode: str) -> list[str]:
    if mode == "same_domain":
        return [ref_domain]
    if mode == "cross_only":
        return [dom for dom in DOMAINS if dom != ref_domain]
    if mode == "all_pairs":
        return list(DOMAINS)
    raise ValueError(f"Unsupported cross-domain mode: {mode}")


def get_selected_strengths(args) -> list[float]:
    if args.strengths:
        return sorted(set(round(float(s), 2) for s in args.strengths))
    return list(DEFAULT_DENOISE_STRENGTHS)


# ── 유틸 ──────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def laplacian_sharpness(img: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def ssim_pair(img_a: Image.Image, img_b: Image.Image, size: int = 224) -> float:
    a = np.array(img_a.resize((size, size)).convert("L")).astype(float) / 255.
    b = np.array(img_b.resize((size, size)).convert("L")).astype(float) / 255.
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(), b.std()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return float(
        (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2) /
        ((mu_a ** 2 + mu_b ** 2 + C1) * (sig_a ** 2 + sig_b ** 2 + C2))
    )


# ── CNN 평가기 ────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_cnn(device: str) -> nn.Module:
    base = models.efficientnet_b0(weights=None)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, len(CLASSES))
    ckpt = torch.load(CNN_CKPT, map_location="cpu", weights_only=False)
    base.load_state_dict(ckpt["model_state_dict"])
    print(f"  CNN 로드: EfficientNet-B0  "
          f"(val_f1={ckpt.get('val_f1', 0.0):.4f})")
    return base.eval().to(device)


@torch.no_grad()
def cnn_predict(model: nn.Module, img: Image.Image, device: str) -> tuple[str, float]:
    x = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0].cpu()
    idx   = probs.argmax().item()
    return CLASSES[idx], float(probs[idx])


# ── SDXL 파이프라인 ───────────────────────────────────────────────────
def load_pipeline(lora_dir: Path, device: str):
    from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

    use_fp16 = (device == "cuda")
    dtype    = torch.float16 if use_fp16 else torch.float32
    variant  = "fp16"        if use_fp16 else None

    print(f"  SDXL 파이프라인 로드 중 ({'fp16' if use_fp16 else 'fp32'})...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, variant=variant, use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    lora_weights = lora_dir / "pytorch_lora_weights.safetensors"
    if lora_weights.exists():
        print(f"  LoRA 로드: {lora_dir.name}")
        pipe.load_lora_weights(str(lora_dir))
    else:
        print(f"  [WARN] LoRA 없음: {lora_dir}, base SDXL만 사용")
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


@torch.no_grad()
def generate_one(pipe, ref_img: Image.Image, prompt: str,
                 strength: float, seed: int, device: str) -> Image.Image:
    ref = ref_img.convert("RGB").resize((512, 512))
    gen = torch.Generator(device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=ref,
        strength=strength,
        guidance_scale=6.0,
        num_inference_steps=25,
        generator=gen,
    ).images[0]
    return result


# ── 시각화 ────────────────────────────────────────────────────────────
def save_domain_grid(
    ref_imgs: list[Image.Image],
    gen_rows: list[tuple[str, list[Image.Image]]],   # [(label, [imgs]), ...]
    out_path: Path,
    domain_short: str,
    cls_name: str,
    thumb: int = 160,
):
    """
    참조 이미지(행 0) + 각 생성 조건(행 1~)으로 이루어진 그리드 이미지.
    """
    n_cols  = len(ref_imgs)
    n_rows  = 1 + len(gen_rows)   # ref 행 + 생성 행들
    label_w = 180
    cell_h  = thumb + 4
    H = n_rows * cell_h + 30     # +30 헤더
    W = label_w + n_cols * thumb

    grid = Image.new("RGB", (W, H), (25, 25, 35))
    draw = ImageDraw.Draw(grid)

    # 헤더
    draw.text((4, 6), f"Class: {cls_name}  |  Domain: {domain_short}",
              fill=(255, 220, 80))

    # 참조 이미지 행
    row_y = 30
    draw.text((4, row_y + thumb // 2 - 6), "Reference", fill=(180, 255, 180))
    for c, rimg in enumerate(ref_imgs):
        grid.paste(rimg.resize((thumb, thumb)), (label_w + c * thumb, row_y))
    row_y += cell_h

    # 생성 이미지 행
    for label, gimgs in gen_rows:
        draw.text((4, row_y + thumb // 2 - 6), label[:22], fill=(180, 200, 255))
        for c, gimg in enumerate(gimgs[:n_cols]):
            grid.paste(gimg.resize((thumb, thumb)), (label_w + c * thumb, row_y))
        row_y += cell_h

    grid.save(out_path)


# ── 단일 클래스 생성 파이프라인 ───────────────────────────────────────
def run_class(cls_name: str, args) -> dict:
    print(f"\n{'='*65}")
    print(f"  클래스: {cls_name.upper()}")
    print(f"  LoRA:   multidomain_{cls_name}")
    target_domains_per_ref = len(get_target_domains(DOMAINS[0], args.cross_domain_mode))
    strengths = get_selected_strengths(args)
    print(f"  참조도메인: {len(DOMAINS)}개 × 목표스타일도메인 {target_domains_per_ref}개 "
          f"({args.cross_domain_mode}) × {args.n_per_domain}입력 "
          f"× {len(strengths)}strength × {len(PROMPT_TEMPLATES)}template "
          f"× {args.n_seeds}seed")
    n_total = (len(DOMAINS) * target_domains_per_ref * args.n_per_domain
               * len(strengths) * len(PROMPT_TEMPLATES) * args.n_seeds)
    print(f"  총 생성: {n_total}장")
    print(f"{'='*65}")

    device   = get_device()
    rng      = random.Random(args.seed)
    cls_idx  = CLASSES.index(cls_name)

    # 출력 디렉토리
    gen_cls_dir = GEN_DIR / cls_name
    out_cls_dir = OUT_ROOT / cls_name
    out_cls_dir.mkdir(parents=True, exist_ok=True)

    # CNN 로드
    print(f"\n[1/4] CNN 평가기 로드...")
    cnn = load_cnn(device)

    # 파이프라인 로드
    lora_dir = LORA_ROOT / f"multidomain_{cls_name}"
    print(f"\n[2/4] SDXL + LoRA 파이프라인 로드...")
    if args.dry_run:
        pipe = None
        print("  [DRY RUN] 파이프라인 로드 스킵")
    else:
        pipe = load_pipeline(lora_dir, device)

    # 도메인별 입력 이미지 샘플링
    print(f"\n[3/4] 도메인별 입력 이미지 샘플링...")
    domain_inputs: dict[str, list[Path]] = {}
    for dom in DOMAINS:
        cls_dir = DATA_DIR / dom / cls_name
        if not cls_dir.exists():
            print(f"  [WARN] {dom}/{cls_name} 없음, 스킵")
            continue
        imgs = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        n = min(args.n_per_domain, len(imgs))
        sampled = rng.sample(imgs, n)
        domain_inputs[dom] = sampled
        print(f"  {DOMAIN_SHORT[dom]:8s} ({dom}): {len(imgs)}장 중 {n}장 샘플링")

    # ── 생성 루프 ────────────────────────────────────────────────────
    print(f"\n[4/4] 생성 루프 시작 ({n_total}장)...")
    t0 = time.time()

    all_records = []   # 이미지별 지표 수집
    n_done = 0
    n_skip = 0

    # 도메인별 그리드용 데이터 수집
    grid_data: dict[tuple[str, str], dict] = {}   # (ref_dom, target_dom) → {refs, rows}

    for ref_dom, ref_paths in domain_inputs.items():
        ref_short = DOMAIN_SHORT[ref_dom]
        morph     = CLASS_MORPHOLOGY[cls_name]
        ds_tag_fn = lambda ds: f"ds{int(ds * 100):03d}"

        ref_imgs_pil = [Image.open(p).convert("RGB") for p in ref_paths]
        for target_dom in get_target_domains(ref_dom, args.cross_domain_mode):
            target_short = DOMAIN_SHORT[target_dom]
            dom_desc = DOMAIN_PROMPTS[target_dom]
            grid_key = (ref_dom, target_dom)
            grid_data[grid_key] = {"refs": ref_imgs_pil, "rows": {}}

            for ds in strengths:
                ds_tag = ds_tag_fn(ds)
                for t_idx, tpl_fn in enumerate(PROMPT_TEMPLATES):
                    t_name  = TEMPLATE_NAMES[t_idx]
                    prompt  = tpl_fn(cls_name, morph, dom_desc)

                    # 이 조건의 생성 이미지 (그리드용)
                    cond_label = f"{ref_short}->{target_short} ds={ds} T{t_idx}:{t_name[:12]}"
                    cond_imgs  = []

                    for inp_idx, ref_path in enumerate(ref_paths):
                        ref_img = ref_imgs_pil[inp_idx]
                        for s in range(args.n_seeds):
                            seed_val = args.seed + inp_idx * 100 + t_idx * 10 + s

                            # 출력 경로
                            if args.cross_domain_mode == "same_domain":
                                out_img_dir = (gen_cls_dir / ref_dom / ds_tag
                                               / f"tpl{t_idx}_{t_name}")
                            else:
                                out_img_dir = (gen_cls_dir / f"ref_{ref_dom}" / f"to_{target_dom}"
                                               / ds_tag / f"tpl{t_idx}_{t_name}")
                            out_img_dir.mkdir(parents=True, exist_ok=True)
                            out_img_path = out_img_dir / f"{inp_idx:04d}_s{s}.png"

                            # 이미 존재하면 스킵
                            if out_img_path.exists() and not args.force:
                                gen_img = Image.open(out_img_path).convert("RGB")
                                n_skip += 1
                            else:
                                if args.dry_run:
                                    print(
                                        f"  [DRY] {ref_short}->{target_short} ds={ds} T{t_idx} "
                                        f"inp{inp_idx} s{s} → {out_img_path.name}"
                                    )
                                    n_done += 1
                                    continue
                                gen_img = generate_one(
                                    pipe, ref_img, prompt, ds, seed_val, device
                                )
                                gen_img.save(out_img_path)
                                n_done += 1

                            if args.dry_run:
                                continue

                            # CNN 평가
                            pred_cls, conf = cnn_predict(cnn, gen_img, device)
                            correct        = (pred_cls == cls_name)
                            sharp          = laplacian_sharpness(gen_img)
                            ssim_vs_ref    = ssim_pair(gen_img, ref_img)

                            record = {
                                "file":               str(out_img_path.relative_to(ROOT)),
                                "domain":             target_dom,
                                "domain_short":       target_short,
                                "ref_domain":         ref_dom,
                                "ref_domain_short":   ref_short,
                                "prompt_domain":      target_dom,
                                "prompt_domain_short": target_short,
                                "is_cross_domain":    ref_dom != target_dom,
                                "denoise":            ds,
                                "template_idx":       t_idx,
                                "template_name":      t_name,
                                "inp_idx":            inp_idx,
                                "seed":               seed_val,
                                "sharpness":          round(sharp, 2),
                                "ssim_vs_ref":        round(ssim_vs_ref, 4),
                                "cnn_pred":           pred_cls,
                                "cnn_conf":           round(conf, 4),
                                "cnn_correct":        correct,
                            }
                            all_records.append(record)

                            # 그리드용 (inp_idx=0, s=0 만 수집)
                            if s == 0:
                                cond_imgs.append(gen_img)

                            elapsed = (time.time() - t0) / 60
                            total_sofar = n_done + n_skip
                            print(
                                f"  [{total_sofar:4d}/{n_total}] "
                                f"{ref_short}->{target_short} ds={ds} T{t_idx} inp{inp_idx} s{s} "
                                f"| pred={pred_cls}{' ✓' if correct else ' ✗'} "
                                f"conf={conf:.3f} sharp={sharp:.1f} "
                                f"ssim={ssim_vs_ref:.3f} "
                                f"| {elapsed:.1f}min"
                            )

                    if cond_imgs:
                        grid_data[grid_key]["rows"][cond_label] = cond_imgs

    if args.dry_run:
        print(f"\n[DRY RUN] 완료. 총 {n_done}장 생성 예정.")
        return {}

    # ── 집계 ─────────────────────────────────────────────────────────
    print(f"\n집계 중...")

    def agg_records(records: list) -> dict:
        if not records:
            return {}
        return {
            "n":              len(records),
            "cnn_accuracy":   round(sum(r["cnn_correct"] for r in records) / len(records), 4),
            "cnn_conf_mean":  round(float(np.mean([r["cnn_conf"] for r in records])), 4),
            "cnn_conf_std":   round(float(np.std([r["cnn_conf"] for r in records])), 4),
            "sharpness_mean": round(float(np.mean([r["sharpness"] for r in records])), 2),
            "sharpness_std":  round(float(np.std([r["sharpness"] for r in records])), 2),
            "ssim_mean":      round(float(np.mean([r["ssim_vs_ref"] for r in records])), 4),
            "ssim_std":       round(float(np.std([r["ssim_vs_ref"] for r in records])), 4),
        }

    # 전체 집계
    overall = agg_records(all_records)

    # 도메인별
    by_domain = {}
    for dom in DOMAINS:
        sub = [r for r in all_records if r["domain"] == dom]
        by_domain[dom] = agg_records(sub)

    # denoise별
    by_denoise = {}
    for ds in strengths:
        sub = [r for r in all_records if r["denoise"] == ds]
        by_denoise[str(ds)] = agg_records(sub)

    # 템플릿별
    by_template = {}
    for t_idx, t_name in enumerate(TEMPLATE_NAMES):
        sub = [r for r in all_records if r["template_idx"] == t_idx]
        by_template[f"tpl{t_idx}_{t_name}"] = agg_records(sub)

    report = {
        "class":          cls_name,
        "lora":           f"multidomain_{cls_name}",
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_generated":    len(all_records),
        "n_skip":         n_skip,
        "elapsed_min":    round((time.time() - t0) / 60, 1),
        "diversity_axes": {
            "ref_domains":   [DOMAIN_SHORT[d] for d in DOMAINS],
            "target_domains_per_ref": target_domains_per_ref,
            "cross_domain_mode": args.cross_domain_mode,
            "strengths": strengths,
            "templates": TEMPLATE_NAMES,
            "n_seeds":   args.n_seeds,
        },
        "aggregate":      overall,
        "by_domain":      by_domain,
        "by_denoise":     by_denoise,
        "by_template":    by_template,
        "per_image":      all_records,
    }

    report_path = out_cls_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  report.json → {report_path}")

    # ── 도메인별 비교 그리드 ────────────────────────────────────────
    print(f"  도메인별 그리드 생성...")
    for (ref_dom, target_dom), gd in grid_data.items():
        if not gd["refs"]:
            continue
        gen_rows = [(label, imgs) for label, imgs in list(gd["rows"].items())[:8]]
        if args.cross_domain_mode == "same_domain":
            grid_out = out_cls_dir / f"grid_{DOMAIN_SHORT[ref_dom]}.png"
        else:
            grid_out = out_cls_dir / f"grid_{DOMAIN_SHORT[ref_dom]}_to_{DOMAIN_SHORT[target_dom]}.png"
        save_domain_grid(
            gd["refs"], gen_rows, grid_out,
            f"{DOMAIN_SHORT[ref_dom]} -> {DOMAIN_SHORT[target_dom]}", cls_name
        )
        print(f"    {grid_out.name}")

    # ── 요약 마크다운 ─────────────────────────────────────────────
    md = _build_summary_md(cls_name, report)
    md_path = out_cls_dir / "summary.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  summary.md  → {md_path}")

    # ── 터미널 요약 ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  클래스: {cls_name.upper()}")
    print(f"  총 생성: {overall['n']}장  (스킵: {n_skip})")
    print(f"  CNN 정확도: {overall['cnn_accuracy']*100:.1f}%  "
          f"신뢰도: {overall['cnn_conf_mean']:.4f} ± {overall['cnn_conf_std']:.4f}")
    print(f"  선명도:     {overall['sharpness_mean']:.2f} ± {overall['sharpness_std']:.2f}")
    print(f"  SSIM:       {overall['ssim_mean']:.4f} ± {overall['ssim_std']:.4f}")
    print()

    print(f"  ── 도메인별 CNN 정확도 / SSIM ──")
    for dom, agg in by_domain.items():
        if agg:
            print(f"    {DOMAIN_SHORT[dom]:8s}: acc={agg['cnn_accuracy']*100:.0f}%  "
                  f"ssim={agg['ssim_mean']:.4f}  sharp={agg['sharpness_mean']:.1f}")

    print(f"\n  ── Denoise별 CNN 정확도 / SSIM ──")
    for ds_str, agg in by_denoise.items():
        if agg:
            print(f"    ds={ds_str}: acc={agg['cnn_accuracy']*100:.0f}%  "
                  f"ssim={agg['ssim_mean']:.4f}  sharp={agg['sharpness_mean']:.1f}")

    print(f"\n  ── 프롬프트 템플릿별 CNN 정확도 / SSIM ──")
    for t_key, agg in by_template.items():
        if agg:
            t_short = t_key.split("_", 1)[1][:20]
            print(f"    {t_short:22s}: acc={agg['cnn_accuracy']*100:.0f}%  "
                  f"ssim={agg['ssim_mean']:.4f}  sharp={agg['sharpness_mean']:.1f}")

    print(f"\n  출력: {out_cls_dir}")
    print(f"  이미지: {gen_cls_dir}")
    print(f"  소요: {report['elapsed_min']}분")
    print(f"{'='*65}")

    return report


# ── 마크다운 요약 빌더 ─────────────────────────────────────────────────
def _build_summary_md(cls_name: str, report: dict) -> str:
    ts  = report["timestamp"]
    oa  = report["aggregate"]
    ax  = report["diversity_axes"]

    lines = [
        f"# {cls_name.capitalize()} 다양화 생성 요약 리포트",
        "",
        f"> **생성 일시:** {ts}  ",
        f"> **LoRA:** multidomain_{cls_name}  ",
        f"> **기반 모델:** stabilityai/stable-diffusion-xl-base-1.0  ",
        f"> **총 생성:** {report['n_generated']}장  ",
        f"> **소요 시간:** {report['elapsed_min']}분  ",
        "",
        "---",
        "",
        "## 1. 다양성 축 설정",
        "",
        "| 축 | 값 | 수 |",
        "|---|---|---|",
        f"| 참조 도메인 | {', '.join(ax['ref_domains'])} | {len(ax['ref_domains'])} |",
        f"| 목표 스타일 도메인 | ref당 {ax['target_domains_per_ref']}개 ({ax['cross_domain_mode']}) | {ax['target_domains_per_ref']} |",
        f"| Denoise 강도 | {', '.join(str(s) for s in ax['strengths'])} | {len(ax['strengths'])} |",
        f"| 프롬프트 템플릿 | {', '.join(ax['templates'])} | {len(ax['templates'])} |",
        f"| 랜덤 시드 | seed×{ax['n_seeds']} | {ax['n_seeds']} |",
        "",
        "---",
        "",
        "## 2. 전체 집계",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 총 생성 수 | **{oa['n']}장** |",
        f"| CNN 분류 정확도 ↑ | **{oa['cnn_accuracy']*100:.1f}%** |",
        f"| CNN 신뢰도 ↑ | {oa['cnn_conf_mean']:.4f} ± {oa['cnn_conf_std']:.4f} |",
        f"| 선명도 (Laplacian) ↑ | {oa['sharpness_mean']:.2f} ± {oa['sharpness_std']:.2f} |",
        f"| SSIM vs 참조 ↓ (다양성↑) | {oa['ssim_mean']:.4f} ± {oa['ssim_std']:.4f} |",
        "",
        "> SSIM이 낮을수록 참조 이미지와 구조적으로 달라진 것 → 다양성 높음",
        "",
        "---",
        "",
        "## 3. 도메인별 성능",
        "",
        "| 도메인 | 염색/장비 | n | CNN acc ↑ | SSIM ↓ | 선명도 ↑ |",
        "|---|---|---|---|---|---|",
    ]

    DOMAIN_INFO = {
        "domain_a_pbc":    ("PBC",    "May-Grünwald Giemsa / CellaVision"),
        "domain_b_raabin": ("Raabin", "Giemsa / 스마트폰 현미경"),
        "domain_c_mll23":  ("MLL23",  "Pappenheim / Metafer scanner"),
        "domain_e_amc":    ("AMC",    "Romanowsky / miLab"),
    }
    for dom, agg in report["by_domain"].items():
        if not agg:
            continue
        short, info = DOMAIN_INFO.get(dom, (dom, ""))
        acc_flag = "✅" if agg["cnn_accuracy"] >= 0.9 else ("⚠️" if agg["cnn_accuracy"] >= 0.7 else "❌")
        lines.append(
            f"| **{short}** | {info} | {agg['n']} "
            f"| {acc_flag} {agg['cnn_accuracy']*100:.0f}% "
            f"| {agg['ssim_mean']:.4f} ± {agg['ssim_std']:.4f} "
            f"| {agg['sharpness_mean']:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Denoise 강도별 성능",
        "",
        "| Denoise | 의미 | n | CNN acc ↑ | SSIM ↓ | 선명도 ↑ |",
        "|---|---|---|---|---|---|",
    ]

    DS_DESC = {
        "0.25": "원본 구조 강보존 (약한 변형)",
        "0.35": "균형점 (현 기준선)",
        "0.45": "더 강한 재합성 (높은 다양성)",
    }
    for ds_str, agg in report["by_denoise"].items():
        if not agg:
            continue
        desc = DS_DESC.get(ds_str, "")
        acc_flag = "✅" if agg["cnn_accuracy"] >= 0.9 else ("⚠️" if agg["cnn_accuracy"] >= 0.7 else "❌")
        lines.append(
            f"| **{ds_str}** | {desc} | {agg['n']} "
            f"| {acc_flag} {agg['cnn_accuracy']*100:.0f}% "
            f"| {agg['ssim_mean']:.4f} ± {agg['ssim_std']:.4f} "
            f"| {agg['sharpness_mean']:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. 프롬프트 템플릿별 성능",
        "",
        "| 템플릿 | 설명 | n | CNN acc ↑ | SSIM ↓ | 선명도 ↑ |",
        "|---|---|---|---|---|---|",
    ]

    TPL_DESC = {
        "standard":           "표준 프롬프트 (기존 기준선)",
        "oil_immersion":      "100x oil immersion 배율 강조",
        "clinical_hematology":"병리학 관점 (nuclear morphology)",
        "cytology":           "세포학 관점 (cytology preparation)",
    }
    for t_key, agg in report["by_template"].items():
        if not agg:
            continue
        t_name = t_key.split("_", 1)[1] if "_" in t_key else t_key
        desc = TPL_DESC.get(t_name, "")
        acc_flag = "✅" if agg["cnn_accuracy"] >= 0.9 else ("⚠️" if agg["cnn_accuracy"] >= 0.7 else "❌")
        lines.append(
            f"| **{t_name}** | {desc} | {agg['n']} "
            f"| {acc_flag} {agg['cnn_accuracy']*100:.0f}% "
            f"| {agg['ssim_mean']:.4f} ± {agg['ssim_std']:.4f} "
            f"| {agg['sharpness_mean']:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. 재현 명령",
        "",
        "```bash",
        f"# {cls_name} 단일 클래스 생성",
        f"python scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py --class_name {cls_name}",
        "",
        "# 모든 클래스 순차 생성",
        "python scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py --all_classes",
        "",
        "# 빠른 테스트 (도메인당 1장, 2시드)",
        f"python scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py --class_name {cls_name} --n_per_domain 1 --n_seeds 2",
        "",
        "# 특정 denoise만 제한 실행",
        f"python scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py --class_name {cls_name} --strengths 0.25 0.35",
        "```",
        "",
        "*Generated by `scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py` — wbc_synthesis pipeline*",
    ]

    return "\n".join(lines)


# ── argparse ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="멀티축 다양화 WBC 이미지 대량 생성 (Script 33)"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--class_name",  choices=CLASSES, help="단일 클래스명")
    group.add_argument("--all_classes", action="store_true", help="5클래스 순차 생성")
    p.add_argument("--n_per_domain", type=int, default=3,
                   help="도메인당 참조 이미지 수 (기본: 3)")
    p.add_argument("--n_seeds",      type=int, default=2,
                   help="참조 이미지당 시드 수 (기본: 2)")
    p.add_argument("--seed",         type=int, default=42,
                   help="랜덤 시드 베이스 (기본: 42)")
    p.add_argument("--cross_domain_mode", choices=["same_domain", "cross_only", "all_pairs"],
                   default="same_domain",
                   help="참조 도메인과 prompt 도메인의 조합 방식")
    p.add_argument("--strengths", type=float, nargs="+",
                   help="사용할 denoise strength 목록 (예: --strengths 0.25 0.35)")
    p.add_argument("--force",        action="store_true",
                   help="기존 이미지 덮어쓰기 (기본: 스킵)")
    p.add_argument("--dry_run",      action="store_true",
                   help="생성 없이 구조·수량 확인만")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    strengths = get_selected_strengths(args)

    print(f"\n{'='*65}")
    print(f"  Script 33 — 멀티축 다양화 WBC 이미지 생성")
    print(f"  n_per_domain={args.n_per_domain}, n_seeds={args.n_seeds}")
    print(f"  cross_domain_mode={args.cross_domain_mode}")
    print(f"  strengths={strengths}")
    print(f"  force={args.force}, dry_run={args.dry_run}")
    target_domains_per_ref = len(get_target_domains(DOMAINS[0], args.cross_domain_mode))
    print(f"  다양성 축: {len(DOMAINS)}참조도메인 × {target_domains_per_ref}목표도메인 × {len(strengths)}strength "
          f"× {len(PROMPT_TEMPLATES)}template × {args.n_seeds}seed")
    print(f"{'='*65}")

    if args.all_classes:
        for cls in CLASSES:
            run_class(cls, args)
    else:
        run_class(args.class_name, args)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
