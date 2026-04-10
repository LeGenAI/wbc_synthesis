"""
Script 31: 프롬프트 다양화 실험
================================
현재 build_class_domain_prompt()는 고정 템플릿 1개만 사용:
  f"microscopy image of a single {cls} white blood cell, peripheral blood smear, {morph}, {dom}, sharp focus, ..."

→ 동일 입력 + 동일 프롬프트 + 다른 seed → seed 노이즈(35%)만 변이
→ 프롬프트를 다양화 → 동일 입력에서도 다른 스타일의 이미지 기대

비교 조건 (3가지):
  A: 기존 고정 프롬프트 (기준선, Script 25/28과 직접 비교)
  B: 매 생성마다 4개 템플릿 풀에서 랜덤 선택
  C: inp_idx에 따른 순환 선택 (재현성 보장, inp_idx % 4)

4가지 프롬프트 템플릿:
  0 (기존): microscopy image of a single {cls} ... peripheral blood smear ... sharp focus
  1 (배율): 100x oil immersion microscopy ... high-resolution hematology imaging
  2 (병리): clinical hematology ... bright-field microscopy, detailed nuclear morphology
  3 (세포): cytology preparation ... blood film ... clinical diagnostic

측정 지표:
  - CNN acc (VGG16 우선, fallback: EfficientNet-B0)
  - Inter SSIM (다른 입력간 SSIM, 낮을수록 다양)
  - Intra SSIM (같은 입력 내 seed간 SSIM)

생성 규모: 5cls × 4dom × 5입력 × 3seed × 3조건 = 900장
예상 시간: 파이프라인 로드 ~5분 + 900장 × ~5초 ≈ 80분

Usage:
    python3 scripts/legacy/phase_18_32_generation_ablation/31_prompt_diversity.py
    python3 scripts/legacy/phase_18_32_generation_ablation/31_prompt_diversity.py --n_inputs 3 --n_seeds 2  # 빠른 테스트
    python3 scripts/legacy/phase_18_32_generation_ablation/31_prompt_diversity.py --dry_run
"""

import argparse
import importlib.util
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import structural_similarity as ssim_skimage
from torchvision import models, transforms

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data" / "processed_multidomain"
CNN_CKPT     = ROOT / "models" / "multidomain_cnn.pt"        # EfficientNet (fallback)
VGG16_CKPT   = ROOT / "models" / "multidomain_cnn_vgg16.pt"  # VGG16 (우선)
ROUTER_CKPT  = ROOT / "models" / "dual_head_router.pt"
OUT_DIR      = ROOT / "results" / "prompt_diversity"

# ── 실험 상수 ──────────────────────────────────────────────────────────────────
CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
DOMAIN_MAP = {
    "PBC":    "domain_a_pbc",
    "Raabin": "domain_b_raabin",
    "MLL23":  "domain_c_mll23",
    "AMC":    "domain_e_amc",
}
DOMAINS = list(DOMAIN_MAP.keys())

STRENGTH      = 0.35
SAMPLING_SEED = 42

# 비교 조건
CONDITIONS = {
    "A": {"variant": "fixed",  "label": "기존 고정 프롬프트 (기준선)"},
    "B": {"variant": "random", "label": "매 생성마다 랜덤 템플릿 변형"},
    "C": {"variant": "cycle",  "label": "inp_idx × seed 기반 순환 (재현성)"},
}

# ── 형태학적 설명 (15_router_inference.py에서 참조) ──────────────────────────
CLASS_MORPHOLOGY = {
    "basophil":   "bilobed nucleus with dark purple-black granules filling cytoplasm",
    "eosinophil": "bilobed nucleus with bright orange-red granules",
    "lymphocyte": "large round nucleus with scant agranular cytoplasm",
    "monocyte":   "kidney-shaped or folded nucleus with grey-blue cytoplasm",
    "neutrophil": "multilobed nucleus with pale pink granules",
}

DOMAIN_PROMPTS = {
    "PBC":    "May-Grünwald Giemsa stain, CellaVision automated analyzer, Barcelona Spain",
    "Raabin": "Giemsa stain, smartphone microscope camera, Iran hospital",
    "MLL23":  "Pappenheim stain, Metafer scanner, Germany clinical lab",
    "AMC":    "Romanowsky stain, miLab automated analyzer, South Korea AMC",
}

NEGATIVE_PROMPT = (
    "cartoon, illustration, text, watermark, multiple cells, "
    "heavy artifacts, unrealistic colors, deformed nucleus, blurry"
)

# ── 4가지 프롬프트 템플릿 ──────────────────────────────────────────────────────
# Template 0: 기존 (기준선) — Script 15/25/28과 동일
def _tpl0(cls, morph, dom):
    return (
        f"microscopy image of a single {cls} white blood cell, "
        f"peripheral blood smear, {morph}, "
        f"{dom}, "
        f"sharp focus, realistic, clinical lab imaging"
    )

# Template 1: 배율 강조 (100x oil immersion)
def _tpl1(cls, morph, dom):
    return (
        f"100x oil immersion microscopy, single {cls} leukocyte, "
        f"{morph}, {dom}, "
        f"high-resolution hematology imaging"
    )

# Template 2: 병리학 관점 (clinical hematology)
def _tpl2(cls, morph, dom):
    return (
        f"clinical hematology, {cls} white blood cell, "
        f"peripheral blood smear analysis, {morph}, {dom}, "
        f"bright-field microscopy, detailed nuclear morphology"
    )

# Template 3: 세포학 관점 (cytology)
def _tpl3(cls, morph, dom):
    return (
        f"cytology preparation, isolated {cls} granulocyte, "
        f"{morph}, blood film, {dom}, "
        f"professional microscopic imaging, clinical diagnostic"
    )

PROMPT_TEMPLATES = [_tpl0, _tpl1, _tpl2, _tpl3]
N_TEMPLATES = len(PROMPT_TEMPLATES)


def build_prompt(cls_name: str, dom_short: str,
                 variant: str, inp_idx: int = 0, seed_offset: int = 0,
                 rng: random.Random = None) -> tuple[str, int]:
    """
    클래스명 + 도메인 단축명으로 프롬프트 생성.
    반환: (prompt_str, template_idx)
    """
    morph = CLASS_MORPHOLOGY[cls_name]
    dom   = DOMAIN_PROMPTS[dom_short]

    if variant == "fixed":
        idx = 0
    elif variant == "random":
        idx = rng.randint(0, N_TEMPLATES - 1)
    elif variant == "cycle":
        # inp_idx × N_SEEDS + seed_offset → 결정적 순환
        idx = (inp_idx * 3 + seed_offset) % N_TEMPLATES
    else:
        idx = 0

    return PROMPT_TEMPLATES[idx](cls_name, morph, dom), idx


# ── WBCRouter 로드 ────────────────────────────────────────────────────────────
def load_router_module():
    spec = importlib.util.spec_from_file_location(
        "router_inference",
        ROOT / "scripts" / "15_router_inference.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── CNN 평가기 (VGG16 우선, fallback EfficientNet-B0) ──────────────────────────
def load_cnn(device: str) -> tuple[nn.Module, str]:
    """VGG16 우선 로드. 없으면 EfficientNet-B0 fallback."""
    if VGG16_CKPT.exists():
        base = models.vgg16(weights=None)
        base.classifier[6] = nn.Linear(base.classifier[6].in_features, len(CLASSES))
        ckpt = torch.load(VGG16_CKPT, map_location="cpu", weights_only=False)
        base.load_state_dict(ckpt["model_state_dict"])
        print(f"  CNN: VGG16 로드 (val_F1={ckpt.get('val_f1', 0.0):.4f})")
        return base.eval().to(device), "vgg16"
    else:
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, len(CLASSES))
        ckpt = torch.load(CNN_CKPT, map_location="cpu", weights_only=False)
        base.load_state_dict(ckpt["model_state_dict"])
        print(f"  CNN: EfficientNet-B0 fallback (val_F1={ckpt.get('val_f1', 0.0):.4f})")
        return base.eval().to(device), "efficientnet_b0"


VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def cnn_predict(model: nn.Module, img: Image.Image, device: str) -> tuple[str, float]:
    x = VAL_TF(img).unsqueeze(0).to(device)
    logits = model(x)
    probs  = torch.softmax(logits, dim=1)[0]
    idx    = probs.argmax().item()
    return CLASSES[idx], probs[idx].item()


# ── SSIM ─────────────────────────────────────────────────────────────────────
def ssim_pair(img_a: Image.Image, img_b: Image.Image, size: int = 224) -> float:
    a = np.array(img_a.resize((size, size)).convert("L")).astype(float) / 255.
    b = np.array(img_b.resize((size, size)).convert("L")).astype(float) / 255.
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(), b.std()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = 0.01**2, 0.03**2
    return float((2*mu_a*mu_b + C1) * (2*sig_ab + C2) /
                 ((mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2)))


# ── 재개 감지 ────────────────────────────────────────────────────────────────
def _img_is_complete(img_dir: Path, n_seeds: int, cond_keys: list) -> bool:
    """해당 입력의 모든 조건×seed 이미지가 이미 존재하는지 확인."""
    for cond in cond_keys:
        for s in range(n_seeds):
            if not (img_dir / f"cond_{cond}_seed_{s:02d}.png").exists():
                return False
    return True


# ── 갤러리 MD ────────────────────────────────────────────────────────────────
def make_gallery(results: list, out_dir: Path, cond_keys: list, n_seeds: int) -> str:
    lines = [
        "# Script 31: 프롬프트 다양화 실험 갤러리",
        "",
        "## 조건 설명",
        "",
    ]
    for cond, info in CONDITIONS.items():
        if cond in cond_keys:
            lines.append(f"- **{cond}**: {info['label']}")
    lines += [
        "",
        "## 프롬프트 템플릿",
        "",
        "| # | 설명 | 예시 (basophil, PBC) |",
        "|---|------|----------------------|",
        "| 0 | 기존 고정 | microscopy image of a single basophil white blood cell, peripheral blood smear, ... |",
        "| 1 | 100x 배율 강조 | 100x oil immersion microscopy, single basophil leukocyte, ... |",
        "| 2 | 병리학 관점 | clinical hematology, basophil white blood cell, peripheral blood smear analysis, ... |",
        "| 3 | 세포학 관점 | cytology preparation, isolated basophil granulocyte, ... blood film, ... |",
        "",
    ]

    for combo in results:
        cls = combo["cls"]
        dom = combo["dom"]
        img_base = f"images/{cls}/{dom}"
        lines += [f"## {cls} × {dom}", ""]

        for inp_data in combo["inputs"]:
            inp_idx = inp_data["inp_idx"]
            inp_img = f"{img_base}/inp_{inp_idx:02d}/input.png"
            lines += [
                f"### 입력 {inp_idx:02d}",
                "",
                f"| 원본 입력 |",
                f"|:---:|",
                f"| ![inp]({inp_img}) |",
                "",
            ]

            # 조건별 행
            header  = "| 조건 | " + " | ".join(f"seed {s}" for s in range(n_seeds)) + " | CNN acc | SSIM |"
            divider = "|------|" + "|".join([":---:"] * n_seeds) + "|:---:|:---:|"
            lines += [header, divider]

            for cond in cond_keys:
                cond_key = f"cond_{cond}"
                cd = inp_data.get(cond_key, {})
                acc  = cd.get("cnn_acc")
                ssim = cd.get("ssim_mean")

                seeds_imgs = " | ".join(
                    f'![s{s}]({img_base}/inp_{inp_idx:02d}/cond_{cond}_seed_{s:02d}.png)'
                    for s in range(n_seeds)
                )
                acc_str  = (f"🟩{acc*100:.0f}%" if acc >= 0.90 else
                            f"🟨{acc*100:.0f}%" if acc >= 0.67 else
                            f"🟥{acc*100:.0f}%") if acc is not None else "-"
                ssim_str = f"{ssim:.4f}" if ssim is not None else "-"
                lines.append(f"| **{cond}** | {seeds_imgs} | {acc_str} | {ssim_str} |")

            lines.append("")

    return "\n".join(lines)


# ── argparse ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="프롬프트 다양화 실험 (Script 31)")
    p.add_argument("--n_inputs",  type=int, default=5,
                   help="cls×dom 조합당 입력 이미지 수 (기본 5)")
    p.add_argument("--n_seeds",   type=int, default=3,
                   help="입력당 seed 수 (기본 3)")
    p.add_argument("--conds",     nargs="+", default=list(CONDITIONS.keys()),
                   help="실행할 조건 (기본: A B C)")
    p.add_argument("--dry_run",   action="store_true",
                   help="이미지 생성 없이 구조 확인만")
    return p.parse_args()


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    args  = parse_args()
    t0    = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_root = OUT_DIR / "images"

    print(f"\n{'='*60}")
    print(f"  Script 31 — 프롬프트 다양화 실험")
    print(f"  조건: {args.conds}")
    print(f"  n_inputs={args.n_inputs}, n_seeds={args.n_seeds}")
    n_total = len(CLASSES) * len(DOMAINS) * args.n_inputs * args.n_seeds * len(args.conds)
    print(f"  총 생성 예정: {n_total}장")
    print(f"{'='*60}")

    # ── [1/5] WBCRouter 로드 ─────────────────────────────────────────────
    print("\n[1/5] WBCRouter 로드...")
    if args.dry_run:
        print("  [DRY RUN] 스킵")
        device = "cpu"
        router = None
    else:
        mod    = load_router_module()
        router = mod.WBCRouter(
            router_ckpt=ROUTER_CKPT if ROUTER_CKPT.exists() else None,
            cnn_ckpt=CNN_CKPT,
        )
        device = router.device
        print(f"  WBCRouter 초기화 완료: device={device}")
        # 파이프라인 명시적 사전 로딩 (_generate_once 직접 호출 전 필요)
        print("  SDXL 파이프라인 로드 중 (~5~7분)...")
        router._load_lora_pipeline()
        print("  파이프라인 로드 완료")

    # ── [2/5] CNN 평가기 로드 ────────────────────────────────────────────
    print("\n[2/5] CNN 평가기 로드...")
    if args.dry_run:
        cnn_model, cnn_name = None, "dry_run"
    else:
        cnn_model, cnn_name = load_cnn(device)

    # ── [3/5] 데이터 파일 목록 수집 ─────────────────────────────────────
    print("\n[3/5] 데이터 파일 목록 수집...")
    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    rng_sample = random.Random(SAMPLING_SEED)
    rng_prompt = random.Random(SAMPLING_SEED + 1)  # 프롬프트 랜덤용 별도 RNG

    file_pool: dict[str, dict[str, list]] = {}
    for dom_short, dom_key in DOMAIN_MAP.items():
        file_pool[dom_short] = {}
        for cls in CLASSES:
            cls_dir = DATA_DIR / dom_key / cls
            if cls_dir.exists():
                paths = sorted([p for p in cls_dir.iterdir()
                                if p.suffix.lower() in IMG_EXTS])
                file_pool[dom_short][cls] = paths
                print(f"  {dom_short}/{cls}: {len(paths)}장")

    # ── [4/5] 생성 루프 ──────────────────────────────────────────────────
    print(f"\n[4/5] 생성 시작 (총 {n_total}장, 예상 ~{n_total*5/60:.0f}분)...")

    results = []
    n_done  = 0
    n_skip  = 0

    for cls in CLASSES:
        for dom in DOMAINS:
            files = file_pool.get(dom, {}).get(cls, [])
            if len(files) < args.n_inputs:
                print(f"  [WARN] {dom}/{cls}: 파일 {len(files)}장 < n_inputs={args.n_inputs}, 전수 사용")
            chosen_inputs = rng_sample.sample(files, k=min(args.n_inputs, len(files)))

            combo_entry = {"cls": cls, "dom": dom, "inputs": []}

            for inp_idx, inp_path in enumerate(chosen_inputs):
                inp_img    = Image.open(inp_path).convert("RGB")
                inp_dir    = img_root / cls / dom / f"inp_{inp_idx:02d}"
                inp_dir.mkdir(parents=True, exist_ok=True)

                # 입력 이미지 저장
                inp_save = inp_dir / "input.png"
                if not inp_save.exists():
                    inp_img.save(inp_save)

                # 완료 확인
                if _img_is_complete(inp_dir, args.n_seeds, args.conds):
                    n_skip += args.n_seeds * len(args.conds)
                    # 기존 데이터 재로드 (정확한 집계를 위해)
                    inp_entry = {"inp_idx": inp_idx, "inp_path": str(inp_path)}
                    for cond in args.conds:
                        seeds_detail = []
                        ssim_vals = []
                        cnn_corrects = []
                        for s in range(args.n_seeds):
                            gen_path = inp_dir / f"cond_{cond}_seed_{s:02d}.png"
                            if gen_path.exists():
                                gen_img = Image.open(gen_path).convert("RGB")
                                sv = ssim_pair(gen_img, inp_img)
                                if cnn_model:
                                    pred, conf = cnn_predict(cnn_model, gen_img, device)
                                    correct = (pred == cls)
                                    seeds_detail.append({
                                        "seed_offset": s, "ssim": round(sv, 4),
                                        "pred": pred, "conf": round(conf, 4),
                                        "correct": correct, "prompt_idx": -1,
                                    })
                                    cnn_corrects.append(correct)
                                else:
                                    seeds_detail.append({"seed_offset": s, "ssim": round(sv, 4)})
                                ssim_vals.append(sv)
                        cnn_acc = (sum(cnn_corrects)/len(cnn_corrects)
                                   if cnn_corrects else None)
                        inp_entry[f"cond_{cond}"] = {
                            "cnn_acc": round(cnn_acc, 4) if cnn_acc is not None else None,
                            "ssim_mean": round(float(np.mean(ssim_vals)), 4) if ssim_vals else None,
                        }
                        inp_entry[f"seeds_{cond}"] = seeds_detail
                    combo_entry["inputs"].append(inp_entry)
                    continue

                if args.dry_run:
                    print(f"  [DRY] {cls}/{dom}/inp_{inp_idx:02d}: 생성 스킵")
                    continue

                inp_entry = {"inp_idx": inp_idx, "inp_path": str(inp_path)}
                gen_imgs_per_cond: dict[str, list] = {c: [] for c in args.conds}

                for cond in args.conds:
                    variant = CONDITIONS[cond]["variant"]
                    seeds_detail = []
                    for s in range(args.n_seeds):
                        prompt, tpl_idx = build_prompt(
                            cls, dom, variant,
                            inp_idx=inp_idx, seed_offset=s,
                            rng=rng_prompt
                        )
                        seed_val = inp_idx * 10 + s

                        try:
                            gen = router._generate_once(
                                inp_img, prompt, denoise=STRENGTH, seed=seed_val
                            )
                        except Exception as e:
                            print(f"  [ERR] {cls}/{dom}/inp_{inp_idx}/cond_{cond}/s{s}: {e}")
                            gen = inp_img  # fallback

                        # 저장
                        gen_path = inp_dir / f"cond_{cond}_seed_{s:02d}.png"
                        gen.save(gen_path)

                        sv = ssim_pair(gen, inp_img)
                        pred, conf = cnn_predict(cnn_model, gen, device)
                        correct = (pred == cls)

                        seeds_detail.append({
                            "seed_offset": s,
                            "seed_value":  seed_val,
                            "prompt_idx":  tpl_idx,
                            "ssim":        round(sv, 4),
                            "pred":        pred,
                            "conf":        round(conf, 4),
                            "correct":     correct,
                        })
                        gen_imgs_per_cond[cond].append(gen)
                        n_done += 1

                    # cond 요약
                    ssim_vals    = [sd["ssim"] for sd in seeds_detail]
                    cnn_corrects = [sd["correct"] for sd in seeds_detail]
                    inp_entry[f"cond_{cond}"] = {
                        "cnn_acc":  round(sum(cnn_corrects)/len(cnn_corrects), 4),
                        "ssim_mean": round(float(np.mean(ssim_vals)), 4),
                    }
                    inp_entry[f"seeds_{cond}"] = seeds_detail

                combo_entry["inputs"].append(inp_entry)
                elapsed = (time.time() - t0) / 60
                print(f"  [{cls}/{dom}/inp{inp_idx:02d}] done | "
                      f"elapsed={elapsed:.1f}min | n_done={n_done}/{n_total}")

            results.append(combo_entry)

    if args.dry_run:
        print("\n[DRY RUN] 완료. 실제 생성 없음.")
        return

    # ── [5/5] 결과 저장 ──────────────────────────────────────────────────
    print(f"\n[5/5] 결과 저장...")
    elapsed_min = (time.time() - t0) / 60

    # 조건별 집계
    def collect_stats(results, cond):
        all_acc = []
        all_ssim = []
        cls_acc = {c: [] for c in CLASSES}
        inter_ssim = []  # 다른 inp 간 SSIM (intra cond)

        for combo in results:
            cls = combo["cls"]
            gen_list_all = []  # 이 cls×dom 조합 내 모든 생성 이미지 경로
            for inp_data in combo["inputs"]:
                cd = inp_data.get(f"cond_{cond}", {})
                if cd.get("cnn_acc") is not None:
                    all_acc.append(cd["cnn_acc"])
                    cls_acc[cls].append(cd["cnn_acc"])
                if cd.get("ssim_mean") is not None:
                    all_ssim.append(cd["ssim_mean"])

                for sd in inp_data.get(f"seeds_{cond}", []):
                    if "ssim" in sd:
                        pass  # already in ssim_mean
                    # 이미지 경로 수집 (inter SSIM 계산용)
                    dom   = combo["dom"]
                    inp_i = inp_data["inp_idx"]
                    s     = sd["seed_offset"]
                    ip    = (img_root / cls / dom / f"inp_{inp_i:02d}" /
                             f"cond_{cond}_seed_{s:02d}.png")
                    if ip.exists():
                        gen_list_all.append(ip)

            # inter SSIM: 동일 cls×dom 내 다른 입력 이미지의 seed_0끼리
            seed0_imgs = []
            for inp_data in combo["inputs"]:
                p = (img_root / combo["cls"] / combo["dom"] /
                     f"inp_{inp_data['inp_idx']:02d}" /
                     f"cond_{cond}_seed_00.png")
                if p.exists():
                    try:
                        seed0_imgs.append(Image.open(p).convert("RGB"))
                    except Exception:
                        pass

            # inter SSIM: 쌍으로 계산
            for i in range(len(seed0_imgs)):
                for j in range(i+1, len(seed0_imgs)):
                    inter_ssim.append(ssim_pair(seed0_imgs[i], seed0_imgs[j]))

        return {
            "cnn_acc_mean":    round(float(np.mean(all_acc)), 4) if all_acc else None,
            "cnn_acc_std":     round(float(np.std(all_acc)), 4) if all_acc else None,
            "ssim_mean":       round(float(np.mean(all_ssim)), 4) if all_ssim else None,
            "inter_ssim_mean": round(float(np.mean(inter_ssim)), 4) if inter_ssim else None,
            "inter_ssim_std":  round(float(np.std(inter_ssim)), 4) if inter_ssim else None,
            "cls_acc": {c: round(float(np.mean(v)), 4) if v else None
                        for c, v in cls_acc.items()},
        }

    agg = {cond: collect_stats(results, cond) for cond in args.conds}

    summary = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_inputs":     args.n_inputs,
        "n_seeds":      args.n_seeds,
        "strength":     STRENGTH,
        "sampling_seed": SAMPLING_SEED,
        "active_conds": args.conds,
        "conditions":   {c: CONDITIONS[c] for c in args.conds},
        "n_total":      n_total,
        "n_done":       n_done,
        "n_skip":       n_skip,
        "elapsed_min":  round(elapsed_min, 1),
        "cnn_model":    cnn_name,
        "aggregated":   agg,
        "results":      results,
    }

    out_json = OUT_DIR / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {out_json}")

    # 갤러리 MD
    gallery_md = make_gallery(results, OUT_DIR, args.conds, args.n_seeds)
    out_md = OUT_DIR / "gallery.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(gallery_md)
    print(f"  MD:   {out_md}")

    # 콘솔 요약
    print(f"\n  ── 결과 요약 ──────────────────────────────")
    print(f"  CNN 모델: {cnn_name}")
    print(f"  {'조건':<4} {'CNN acc':>10} {'SSIM':>8} {'Inter SSIM':>12}")
    print(f"  {'-'*38}")
    for cond in args.conds:
        a = agg[cond]
        acc  = a.get("cnn_acc_mean")
        ssim = a.get("ssim_mean")
        issr = a.get("inter_ssim_mean")
        print(f"  {cond:<4} {(acc*100 if acc else 0):9.1f}% {(ssim or 0):8.4f} {(issr or 0):12.4f}")

    print(f"\n  클래스별 CNN acc:")
    print(f"  {'cls':<12} " + " ".join(f"{c:>6}" for c in args.conds))
    for cls in CLASSES:
        row = f"  {cls:<12} "
        for cond in args.conds:
            v = agg[cond]["cls_acc"].get(cls)
            row += f"{v*100:5.1f}% " if v is not None else "   - % "
        print(row)

    print(f"\n  elapsed: {elapsed_min:.1f}분, n_done={n_done}, n_skip={n_skip}")
    print(f"  Done. 결과: {OUT_DIR}\n")


if __name__ == "__main__":
    main()
