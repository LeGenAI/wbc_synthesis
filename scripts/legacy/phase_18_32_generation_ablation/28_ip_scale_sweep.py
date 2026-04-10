"""
Script 28: IP-Adapter ip_scale 세밀 탐색 (저강도 실험)
====================================================
Script 27 결과: ip_scale=0.2/0.35에서 basophil 외 CNN acc가 크게 하락.
  - basophil: 98% (내성 있음)
  - eosinophil/lymphocyte: 27~40% (매우 취약)
  - monocyte: 55~58%, neutrophil: 68~72%

목표: ip_scale=0.05~0.15 저강도 구간에서 CNN acc 90%+ 유지하면서
      다양성(Inter SSIM 감소)을 달성하는 최적 ip_scale 탐색.

비교 조건 (5가지, 모두 n_ref=1로 통일):
  A:  ip_scale=0.0,  n_ref=0  (기준선, Script 27 A와 동일)
  B1: ip_scale=0.05, n_ref=1  (IP 극약)
  B2: ip_scale=0.10, n_ref=1  (IP 약)
  B3: ip_scale=0.15, n_ref=1  (IP 중약)
  C:  ip_scale=0.20, n_ref=1  (Script 27 B와 동일, 재현성 확인)

총 생성: 5cls × 4dom × 5입력 × 3seed × 5조건 = 1500장
예상 시간: 파이프라인+IP-Adapter 로드 ~7분 + 1500장 × ~5초 ≈ 110분

Usage:
    python3 scripts/legacy/phase_18_32_generation_ablation/28_ip_scale_sweep.py
    python3 scripts/legacy/phase_18_32_generation_ablation/28_ip_scale_sweep.py --n_inputs 3 --n_seeds 2
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
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "processed_multidomain"
CNN_CKPT    = ROOT / "models" / "multidomain_cnn.pt"
ROUTER_CKPT = ROOT / "models" / "dual_head_router.pt"
OUT_DIR     = ROOT / "results" / "ip_scale_sweep"

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

# 5가지 비교 조건 (n_ref=1로 통일, ip_scale 세밀 탐색)
CONDITIONS = {
    "A":  {"ip_scale": 0.0,  "n_ref": 0, "label": "기준선 (IP 없음)"},
    "B1": {"ip_scale": 0.05, "n_ref": 1, "label": "IP 극약 (ip=0.05, ref=1)"},
    "B2": {"ip_scale": 0.10, "n_ref": 1, "label": "IP 약  (ip=0.10, ref=1)"},
    "B3": {"ip_scale": 0.15, "n_ref": 1, "label": "IP 중약 (ip=0.15, ref=1)"},
    "C":  {"ip_scale": 0.20, "n_ref": 1, "label": "IP 중  (ip=0.20, ref=1) — Script 27 B와 동일"},
}

IP_ADAPTER_REPO      = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT    = "ip-adapter_sdxl.bin"


# ── WBCRouter 로드 ────────────────────────────────────────────────────────────
def load_router_module():
    spec = importlib.util.spec_from_file_location(
        "router_inference",
        ROOT / "scripts" / "15_router_inference.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── SSIM 계산 (컬러, Script 25 방식) ─────────────────────────────────────────
def ssim_pair(img_a: Image.Image, img_b: Image.Image) -> float:
    a = np.array(img_a.convert("RGB").resize((256, 256))) / 255.0
    b = np.array(img_b.convert("RGB").resize((256, 256))) / 255.0
    return float(ssim_skimage(a, b, data_range=1.0, channel_axis=2))


# ── CNN 분류기 ────────────────────────────────────────────────────────────────
CNN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_cnn(device):
    base = models.efficientnet_b0(weights=None)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, len(CLASSES))
    ckpt = torch.load(CNN_CKPT, map_location="cpu", weights_only=False)
    base.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return base.eval().to(device)

def cnn_predict(model, img: Image.Image, device):
    t = CNN_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
    idx = probs.argmax().item()
    return CLASSES[idx], probs[idx].item()


# ── FewShotSampler (Script 23에서 재사용) ────────────────────────────────────
class FewShotSampler:
    """같은 클래스 × 다른 도메인에서 참조 이미지를 랜덤 샘플링."""
    def __init__(self, data_dir: Path, domain_list: list, class_list: list, seed: int = SAMPLING_SEED):
        self.rng = random.Random(seed)
        self._pool: dict[str, dict[str, list]] = {}
        for dom in domain_list:
            self._pool[dom] = {}
            for cls in class_list:
                d = data_dir / dom / cls
                if d.exists():
                    self._pool[dom][cls] = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
                else:
                    self._pool[dom][cls] = []

    def sample(self, cls_name: str, exclude_domain: str, n: int) -> list:
        combined = []
        for dom in self._pool:
            if dom != exclude_domain:
                combined.extend(self._pool[dom].get(cls_name, []))
        if not combined:
            return []
        chosen = self.rng.choices(combined, k=n)
        return [Image.open(p).convert("RGB") for p in chosen]


# ── IP-Adapter 장착 ───────────────────────────────────────────────────────────
def load_ip_adapter_on_router(router):
    pipe = router.pipe
    print(f"  IP-Adapter 로드 중 ({IP_ADAPTER_REPO})...")
    pipe.disable_attention_slicing()   # AttnProcessor 충돌 방지
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHT,
    )
    print("  IP-Adapter 로드 완료.")


# ── 생성 함수 ─────────────────────────────────────────────────────────────────
def generate_with_ip(router, input_img: Image.Image, ref_imgs: list,
                     prompt: str, strength: float, ip_scale: float,
                     seed: int, negative_prompt: str) -> Image.Image:
    """
    IP-Adapter 적용 img2img 생성.
    ip_scale=0.0: 기준선 (참조 이미지 무시)
    ip_scale>0.0: 참조 이미지 스타일 주입
    """
    pipe   = router.pipe
    device = router.device

    pipe.set_ip_adapter_scale(ip_scale)
    ref = input_img.convert("RGB").resize((512, 512))
    gen = torch.Generator(device).manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=ref,
        strength=strength,
        guidance_scale=6.0,
        num_inference_steps=25,
        generator=gen,
    )

    # IP-Adapter가 장착된 파이프라인에서는 항상 ip_adapter_image를 전달
    # scale=0.0이면 효과 없음 (입력 이미지를 dummy로 전달)
    if ref_imgs:
        kwargs["ip_adapter_image"] = [ref_imgs] if len(ref_imgs) > 1 else ref_imgs[0]
    else:
        # 조건 A: 참조 없음 → 입력 이미지를 dummy로 사용 (scale=0.0으로 무효화)
        kwargs["ip_adapter_image"] = ref

    with torch.no_grad():
        return pipe(**kwargs).images[0]


# ── Intra / Inter SSIM 계산 ───────────────────────────────────────────────────
def compute_intra_inter_ssim(inp_data_list: list, n_seeds: int,
                              out_images_dir: Path, cond_label: str):
    """
    cond_label: 'A', 'B', 'C'
    파일명 패턴: cond_{cond_label}_seed_{s:02d}.png
    """
    gen_imgs = {}
    for inp_data in inp_data_list:
        inp_idx = inp_data["inp_idx"]
        cls, dom = inp_data["cls"], inp_data["dom"]
        for s in range(n_seeds):
            p = out_images_dir / cls / dom / f"input_{inp_idx:02d}" / f"cond_{cond_label}_seed_{s:02d}.png"
            if p.exists():
                gen_imgs[(inp_idx, s)] = Image.open(p).convert("RGB")

    # intra: 같은 입력 내 seed 간 pairwise SSIM
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

    intra = float(np.mean(intra_vals)) if intra_vals else None
    inter = float(np.mean(inter_vals)) if inter_vals else None
    return intra, inter


# ── 갤러리 마크다운 생성 ──────────────────────────────────────────────────────
def acc_badge(acc: float) -> str:
    if acc >= 0.90: return f"🟩 {acc:.0%}"
    if acc >= 0.67: return f"🟨 {acc:.0%}"
    return f"🟥 {acc:.0%}"


def make_gallery(all_results: list, n_inputs: int, n_seeds: int) -> str:
    cond_keys = list(CONDITIONS.keys())  # ["A", "B1", "B2", "B3", "C"]
    n_conds   = len(cond_keys)
    lines = []

    lines += [
        "# WBC IP-Scale Sweep 갤러리 (ip_scale 저강도 탐색)",
        "",
        f"> **Strength:** {STRENGTH} 고정  ",
        f"> **N_INPUTS:** {n_inputs}  ·  **N_SEEDS:** {n_seeds}  ",
        f"> **총 생성:** {len(CLASSES) * len(DOMAINS) * n_inputs * n_seeds * n_conds}장 ({n_conds}조건 × 입력 × seed)  ",
        "",
        "## 실험 조건",
        "",
        "| 조건 | ip_scale | n_ref | 설명 |",
        "| :--- | :---: | :---: | :--- |",
    ]
    for k, v in CONDITIONS.items():
        lines.append(f"| **{k}** | {v['ip_scale']} | {v['n_ref']} | {v['label']} |")

    lines += [
        "",
        "뱃지: 🟩 CNN ≥ 90% · 🟨 ≥ 67% · 🟥 < 67%",
        "",
        "---",
        "",
    ]

    # ── 클래스별 CNN acc 히트맵 (핵심 분석) ─────────────────────────────────
    lines.append("## 🗺️ 클래스별 CNN acc 히트맵 (ip_scale별)")
    lines.append("")
    lines.append("> Script 27 결과와 비교: C(ip=0.20)가 Script 27 B와 동일하므로 재현성 확인 가능")
    lines.append("")

    # 헤더: 클래스 | A | B1 | B2 | B3 | C
    header_cells = ["**클래스**"] + [f"**{k}** (ip={CONDITIONS[k]['ip_scale']})" for k in cond_keys]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| :--- | " + " | ".join([":---:"] * n_conds) + " |")

    for cls in CLASSES:
        cls_rows = [r for r in all_results if r["cls"] == cls]
        cells = [f"**{cls}**"]
        for k in cond_keys:
            accs = [r[f"cond_{k}"]["overall_cnn_acc"] for r in cls_rows if r.get(f"cond_{k}")]
            avg  = sum(accs) / len(accs) if accs else 0
            cells.append(acc_badge(avg))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Inter SSIM 히트맵 ────────────────────────────────────────────────────
    lines.append("## 📊 클래스별 Inter SSIM 히트맵 (낮을수록 다양)")
    lines.append("")

    header_cells = ["**클래스**"] + [f"**{k}** (ip={CONDITIONS[k]['ip_scale']})" for k in cond_keys]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| :--- | " + " | ".join([":---:"] * n_conds) + " |")

    for cls in CLASSES:
        cls_rows = [r for r in all_results if r["cls"] == cls]
        cells = [f"**{cls}**"]
        for k in cond_keys:
            inters = [r[f"cond_{k}"]["inter_ssim"] for r in cls_rows
                      if r.get(f"cond_{k}") and r[f"cond_{k}"].get("inter_ssim")]
            avg = sum(inters) / len(inters) if inters else None
            cells.append(f"{avg:.4f}" if avg is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("> **Inter SSIM**: 다른 입력 이미지 간 생성 결과 유사도. 낮을수록 입력별 다양성 ↑")
    lines.append("> Script 27 기준값: A=0.4595, B(ip=0.2)=0.1253")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 목차 ─────────────────────────────────────────────────────────────────
    lines.append("## 목차")
    lines.append("")
    for cls in CLASSES:
        lines.append(f"### {cls}")
        for dom in DOMAINS:
            lines.append(f"- [{cls} × {dom}](#{cls.lower()}-{dom.lower()})")
        lines.append("")
    lines.append("---")
    lines.append("")

    # ── 클래스 × 도메인 섹션 ─────────────────────────────────────────────────
    for cls in CLASSES:
        lines.append(f"# {cls.upper()}")
        lines.append("")

        for dom in DOMAINS:
            lines.append(f'<a id="{cls.lower()}-{dom.lower()}"></a>')
            lines.append("")
            lines.append(f"## {cls} × {dom}")
            lines.append("")

            entry = next(
                (r for r in all_results if r["cls"] == cls and r["dom"] == dom), None
            )
            if entry is None:
                lines.append("> ⚠️ 데이터 없음")
                lines.append("")
                continue

            # 조건별 요약 지표
            lines.append("| 조건 | CNN acc | SSIM vs 입력 | Intra SSIM | Inter SSIM |")
            lines.append("| :--- | :---: | :---: | :---: | :---: |")
            for k in cond_keys:
                cd = entry.get(f"cond_{k}", {})
                intra = f"{cd['intra_ssim']:.4f}" if cd.get("intra_ssim") else "—"
                inter = f"{cd['inter_ssim']:.4f}" if cd.get("inter_ssim") else "—"
                lines.append(
                    f"| **{k}: {CONDITIONS[k]['label']}** "
                    f"| {acc_badge(cd.get('overall_cnn_acc', 0))} "
                    f"| {cd.get('overall_ssim_mean', 0):.4f} "
                    f"| {intra} | {inter} |"
                )
            lines.append("")

            # 입력별 블록
            for inp_data in entry["inputs"]:
                inp_idx = inp_data["inp_idx"]
                inp_rel = f"images/{cls}/{dom}/input_{inp_idx:02d}/input.png"
                # n_ref=1인 조건들의 참조 이미지 (B1~C 공통으로 ref_B1.png 형식)
                # 실제 저장 시 조건 k별로 ref_{k}.png로 저장됨
                ref_rel = f"images/{cls}/{dom}/input_{inp_idx:02d}/ref_B1.png"

                lines.append(f"### 입력 {inp_idx}")
                lines.append("")

                # 입력 + 참조 이미지 (A는 참조 없음, B1~C는 ref_B1을 대표로 표시)
                lines.append("| 원본 입력 | 참조 이미지 (B1~C 공통) |")
                lines.append("| :---: | :---: |")
                lines.append(f"| ![inp{inp_idx}]({inp_rel}) | ![ref]({ref_rel}) |")
                lines.append("")

                # 조건별 seed 비교 테이블
                seed_headers = " | ".join(f"seed {s}" for s in range(n_seeds))
                lines.append(f"| 조건 | {seed_headers} | CNN acc | SSIM |")
                sep = " | ".join([":---:"] * (n_seeds + 3))
                lines.append(f"| {sep} |")

                for k in cond_keys:
                    seeds_data = inp_data.get(f"seeds_{k}", [])
                    seed_cells = []
                    for s in range(n_seeds):
                        sd = next((x for x in seeds_data if x["seed_offset"] == s), None)
                        if sd:
                            img_rel = f"images/{cls}/{dom}/input_{inp_idx:02d}/cond_{k}_seed_{s:02d}.png"
                            ok = "✅" if sd["correct"] else "❌"
                            seed_cells.append(f"![{k}s{s}]({img_rel}){ok}")
                        else:
                            seed_cells.append("—")

                    agg = inp_data.get(f"cond_{k}", {})
                    acc_val  = agg.get("cnn_acc", 0)
                    ssim_val = agg.get("ssim_mean", 0)
                    lines.append(
                        f"| **{k}** | "
                        + " | ".join(seed_cells)
                        + f" | {acc_badge(acc_val)} | {ssim_val:.4f} |"
                    )

                lines.append("")

            lines.append("---")
            lines.append("")

    # ── 전체 요약 테이블 ──────────────────────────────────────────────────────
    lines.append("# 전체 결과 요약")
    lines.append("")
    lines.append("| 클래스 | 도메인 | 조건 | CNN acc | SSIM | Intra SSIM | Inter SSIM |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")
    for r in all_results:
        for k in cond_keys:
            cd = r.get(f"cond_{k}", {})
            intra = f"{cd['intra_ssim']:.4f}" if cd.get("intra_ssim") else "—"
            inter = f"{cd['inter_ssim']:.4f}" if cd.get("inter_ssim") else "—"
            lines.append(
                f"| {r['cls']} | {r['dom']} | **{k}** "
                f"| {acc_badge(cd.get('overall_cnn_acc', 0))} "
                f"| {cd.get('overall_ssim_mean', 0):.4f} "
                f"| {intra} | {inter} |"
            )

    lines.append("")

    # ── 조건별 전체 평균 ──────────────────────────────────────────────────────
    lines.append("## 조건별 전체 평균")
    lines.append("")
    lines.append("| 조건 | CNN acc | SSIM vs 입력 | Intra SSIM | Inter SSIM |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for k in cond_keys:
        accs   = [r[f"cond_{k}"]["overall_cnn_acc"]  for r in all_results if r.get(f"cond_{k}")]
        ssims  = [r[f"cond_{k}"]["overall_ssim_mean"] for r in all_results if r.get(f"cond_{k}")]
        intras = [r[f"cond_{k}"]["intra_ssim"] for r in all_results if r.get(f"cond_{k}") and r[f"cond_{k}"].get("intra_ssim")]
        inters = [r[f"cond_{k}"]["inter_ssim"] for r in all_results if r.get(f"cond_{k}") and r[f"cond_{k}"].get("inter_ssim")]
        avg_acc   = sum(accs)   / len(accs)   if accs   else 0
        avg_ssim  = sum(ssims)  / len(ssims)  if ssims  else 0
        avg_intra = sum(intras) / len(intras) if intras else 0
        avg_inter = sum(inters) / len(inters) if inters else 0
        intra_s = f"{avg_intra:.4f}" if intras else "—"
        inter_s = f"{avg_inter:.4f}" if inters else "—"
        lines.append(
            f"| **{k}: {CONDITIONS[k]['label']}** "
            f"| {acc_badge(avg_acc)} | {avg_ssim:.4f} "
            f"| {intra_s} | {inter_s} |"
        )

    lines.append("")
    lines.append("> **읽는 법:**  ")
    lines.append("> - Inter SSIM이 낮을수록 다른 입력에서 더 다양한 이미지가 생성됨  ")
    lines.append("> - Intra SSIM이 낮을수록 같은 입력에서 seed 변화에 따른 다양성이 큼  ")
    lines.append("> - CNN acc ≥ 90% + Inter SSIM < A(0.46) 동시 달성이 목표  ")
    lines.append("> - Script 27 참고: A=99% CNN/0.46 Inter · B(ip=0.2)=59%/0.13  ")

    return "\n".join(lines)


# ── 집계 헬퍼 ─────────────────────────────────────────────────────────────────
def aggregate_seeds(seed_list: list) -> dict:
    if not seed_list:
        return {"cnn_acc": 0.0, "ssim_mean": 0.0}
    return {
        "cnn_acc":   round(float(np.mean([s["correct"] for s in seed_list])), 4),
        "ssim_mean": round(float(np.mean([s["ssim"]    for s in seed_list])), 4),
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inputs",  type=int,   default=5,    help="입력 이미지 수/cls×dom")
    parser.add_argument("--n_seeds",   type=int,   default=3,    help="seed 수/입력")
    parser.add_argument("--seed",      type=int,   default=SAMPLING_SEED)
    parser.add_argument("--strength",  type=float, default=STRENGTH)
    parser.add_argument("--skip_cond", type=str,   default="",   help="건너뛸 조건 (예: C)")
    args = parser.parse_args()

    N_INPUTS = args.n_inputs
    N_SEEDS  = args.n_seeds
    DENOISE  = args.strength
    skip     = set(args.skip_cond.upper().split(",")) if args.skip_cond else set()
    active_conds = [k for k in CONDITIONS if k not in skip]

    n_total = len(CLASSES) * len(DOMAINS) * N_INPUTS * N_SEEDS * len(active_conds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(exist_ok=True)

    print("=" * 65)
    print("Script 28: IP-Adapter ip_scale 세밀 탐색 (저강도 실험)")
    print(f"  N_INPUTS={N_INPUTS}, N_SEEDS={N_SEEDS}, strength={DENOISE}")
    print(f"  활성 조건: {active_conds}")
    print(f"  총 생성 예정: {n_total}장")
    print("=" * 65)

    # ── 1. 모델 로드 ──────────────────────────────────────────────────────────
    print("\n[1/4] WBCRouter 및 CNN 로드...")
    mod = load_router_module()
    WBCRouter           = mod.WBCRouter
    build_prompt        = mod.build_class_domain_prompt
    MULTI_CLASSES_LIST  = mod.MULTI_CLASSES
    DOMAIN_LIST         = mod.DOMAINS          # ["domain_a_pbc", ...]
    NEGATIVE_PROMPT     = mod.NEGATIVE_PROMPT
    get_device          = mod.get_device

    device = get_device()
    router = WBCRouter(router_ckpt=ROUTER_CKPT, cnn_ckpt=CNN_CKPT, device=device)

    # 파이프라인 워밍업 (지연 로딩 트리거)
    print("  파이프라인 워밍업...")
    _sample = next((DATA_DIR / DOMAIN_LIST[0] / CLASSES[0]).glob("*.jpg"), None)
    if _sample:
        router.route(Image.open(_sample).convert("RGB"), generate=True, seed=99)
    print("  완료.")

    # ── 2. IP-Adapter 장착 ────────────────────────────────────────────────────
    print("\n[2/4] IP-Adapter 장착...")
    load_ip_adapter_on_router(router)

    cnn_model = load_cnn(device)
    print("  CNN 로드 완료.")

    # ── 3. 참조 이미지 사전 샘플링 ───────────────────────────────────────────
    print("\n[3/4] 입력 및 참조 이미지 사전 샘플링...")
    rng     = random.Random(args.seed)
    sampler = FewShotSampler(DATA_DIR, DOMAIN_LIST, MULTI_CLASSES_LIST, seed=args.seed)

    # 재현성 보장: 루프 전에 모든 입력/참조를 미리 생성
    # n_ref=1 통일이므로 ref_pool은 단순 1장 리스트
    input_pool = {}   # (cls, dom) → [Path, ...]
    ref_pool   = {}   # (cls, dom_key, inp_idx) → [img]  (n_ref=1, 모든 IP 조건 공유)

    for cls in CLASSES:
        for dom in DOMAINS:
            dom_key = DOMAIN_MAP[dom]
            dom_dir = DATA_DIR / dom_key / cls
            files   = sorted(dom_dir.glob("*.jpg")) + sorted(dom_dir.glob("*.png"))
            if not files:
                print(f"  ⚠️  파일 없음: {dom_dir}")
                input_pool[(cls, dom)] = []
                continue
            inputs = rng.sample(files, k=min(N_INPUTS, len(files)))
            input_pool[(cls, dom)] = inputs

            for inp_idx in range(len(inputs)):
                # 모든 IP-Adapter 조건(B1/B2/B3/C)이 공통으로 사용할 참조 이미지 1장
                ref_pool[(cls, dom_key, inp_idx)] = sampler.sample(
                    cls, exclude_domain=dom_key, n=1
                )

    print(f"  입력 샘플링 완료: {len(input_pool)}개 cls×dom")

    # ── 4. 생성 루프 ──────────────────────────────────────────────────────────
    print("\n[4/4] 이미지 생성 및 평가...")
    all_results = []
    done = 0
    t0   = time.time()

    def _inp_is_complete(inp_dir: Path, conds: list, n_seeds: int) -> bool:
        """입력 폴더의 모든 조건×seed 파일이 완성되었는지 확인."""
        for k in conds:
            for s in range(n_seeds):
                if not (inp_dir / f"cond_{k}_seed_{s:02d}.png").exists():
                    return False
        return True

    for cls in CLASSES:
        for dom in DOMAINS:
            dom_key  = DOMAIN_MAP[dom]
            inputs   = input_pool.get((cls, dom), [])
            if not inputs:
                continue

            dom_idx  = DOMAIN_LIST.index(dom_key) if dom_key in DOMAIN_LIST else 0
            cls_idx  = MULTI_CLASSES_LIST.index(cls) if cls in MULTI_CLASSES_LIST else 0
            prompt   = build_prompt(cls_idx, dom_idx)

            print(f"\n── {cls} × {dom} ({len(inputs)}장 입력) ──")

            inp_data_list = []

            for inp_idx, inp_path in enumerate(inputs):
                input_img = Image.open(inp_path).convert("RGB")

                inp_dir = OUT_DIR / "images" / cls / dom / f"input_{inp_idx:02d}"
                inp_dir.mkdir(parents=True, exist_ok=True)
                input_img.save(inp_dir / "input.png")

                # 참조 이미지 (n_ref=1, 모든 IP 조건 공유)
                shared_refs = ref_pool.get((cls, dom_key, inp_idx), [])

                # 참조 이미지 저장 (조건별 동일 이미지를 각 조건명으로 저장)
                for k in active_conds:
                    if CONDITIONS[k]["n_ref"] > 0 and shared_refs:
                        shared_refs[0].save(inp_dir / f"ref_{k}.png")

                # ── 체크포인트: 완성된 입력 폴더는 파일 재평가로 복원 ──
                if _inp_is_complete(inp_dir, active_conds, N_SEEDS):
                    print(f"  [입력 {inp_idx+1}/{len(inputs)}] ✅ 이미 완료, 파일 로드 중...", flush=True)
                    cond_seeds = {k: [] for k in active_conds}
                    for seed_offset in range(N_SEEDS):
                        seed = inp_idx * 10 + seed_offset
                        for k in active_conds:
                            p = inp_dir / f"cond_{k}_seed_{seed_offset:02d}.png"
                            gen = Image.open(p).convert("RGB")
                            sv = ssim_pair(gen, input_img)
                            pred_cls, conf = cnn_predict(cnn_model, gen, device)
                            ok = (pred_cls == cls)
                            cond_seeds[k].append({
                                "seed_offset": seed_offset,
                                "seed_value":  seed,
                                "ssim":        round(sv, 4),
                                "pred":        pred_cls,
                                "conf":        round(conf, 4),
                                "correct":     ok,
                            })
                            done += 1
                    inp_entry = {
                        "inp_idx":  inp_idx,
                        "inp_path": str(inp_path),
                        "cls":      cls,
                        "dom":      dom,
                    }
                    for k in active_conds:
                        agg = aggregate_seeds(cond_seeds[k])
                        inp_entry[f"cond_{k}"] = agg
                        inp_entry[f"seeds_{k}"] = cond_seeds[k]
                    accs_str = " | ".join(
                        f"{k}={aggregate_seeds(cond_seeds[k])['cnn_acc']:.0%}"
                        for k in active_conds
                    )
                    print(f"    [{done}/{n_total}] {accs_str}")
                    inp_data_list.append(inp_entry)
                    continue

                cond_seeds = {k: [] for k in active_conds}

                print(f"  [입력 {inp_idx+1}/{len(inputs)}] ", end="", flush=True)

                for seed_offset in range(N_SEEDS):
                    seed = inp_idx * 10 + seed_offset

                    for k in active_conds:
                        cfg      = CONDITIONS[k]
                        ip_scale = cfg["ip_scale"]
                        n_ref    = cfg["n_ref"]

                        # 이미 생성된 개별 파일은 스킵
                        save_path = inp_dir / f"cond_{k}_seed_{seed_offset:02d}.png"
                        if save_path.exists():
                            gen = Image.open(save_path).convert("RGB")
                        else:
                            # n_ref=0 → 참조 없음(A), n_ref=1 → shared_refs 사용(B1~C)
                            if n_ref == 0:
                                ref_imgs = []
                            else:
                                ref_imgs = shared_refs[:n_ref]

                            gen = generate_with_ip(
                                router, input_img, ref_imgs,
                                prompt, DENOISE, ip_scale, seed,
                                NEGATIVE_PROMPT,
                            )
                            gen.save(save_path)

                        sv = ssim_pair(gen, input_img)
                        pred_cls, conf = cnn_predict(cnn_model, gen, device)
                        ok = (pred_cls == cls)

                        cond_seeds[k].append({
                            "seed_offset": seed_offset,
                            "seed_value":  seed,
                            "ssim":        round(sv, 4),
                            "pred":        pred_cls,
                            "conf":        round(conf, 4),
                            "correct":     ok,
                        })
                        done += 1

                    print("·", end="", flush=True)

                # inp-level 집계
                inp_entry = {
                    "inp_idx":  inp_idx,
                    "inp_path": str(inp_path),
                    "cls":      cls,
                    "dom":      dom,
                }
                for k in active_conds:
                    agg = aggregate_seeds(cond_seeds[k])
                    inp_entry[f"cond_{k}"] = agg
                    inp_entry[f"seeds_{k}"] = cond_seeds[k]

                # 진행 표시
                accs_str = " | ".join(
                    f"{k}={aggregate_seeds(cond_seeds[k])['cnn_acc']:.0%}"
                    for k in active_conds
                )
                print(f" [{done}/{n_total}] {accs_str}")

                inp_data_list.append(inp_entry)

            # cls×dom 수준 집계
            result_entry = {
                "cls":     cls,
                "dom":     dom,
                "dom_key": dom_key,
                "n_inputs": len(inputs),
                "inputs":  inp_data_list,
            }

            out_images = OUT_DIR / "images"
            for k in active_conds:
                all_ssims   = [sd["ssim"]    for inp in inp_data_list for sd in inp.get(f"seeds_{k}", [])]
                all_correct = [sd["correct"] for inp in inp_data_list for sd in inp.get(f"seeds_{k}", [])]
                intra, inter = compute_intra_inter_ssim(inp_data_list, N_SEEDS, out_images, k)
                result_entry[f"cond_{k}"] = {
                    "overall_cnn_acc":   round(float(np.mean(all_correct)), 4) if all_correct else 0,
                    "overall_ssim_mean": round(float(np.mean(all_ssims)),   4) if all_ssims   else 0,
                    "intra_ssim": round(intra, 4) if intra is not None else None,
                    "inter_ssim": round(inter, 4) if inter is not None else None,
                }

            all_results.append(result_entry)

    elapsed = time.time() - t0
    print(f"\n생성 완료: {done}장, 경과 {elapsed/60:.1f}분")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    summary = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_inputs":     N_INPUTS,
        "n_seeds":      N_SEEDS,
        "strength":     DENOISE,
        "sampling_seed": args.seed,
        "active_conds": active_conds,
        "conditions":   CONDITIONS,
        "n_total":      done,
        "elapsed_min":  round(elapsed / 60, 1),
        "results":      all_results,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  summary.json → {OUT_DIR / 'summary.json'}")

    md = make_gallery(all_results, N_INPUTS, N_SEEDS)
    (OUT_DIR / "gallery.md").write_text(md, encoding="utf-8")
    print(f"  gallery.md   → {OUT_DIR / 'gallery.md'}")

    # 콘솔 요약
    print("\n" + "=" * 65)
    print("조건별 전체 평균")
    print("=" * 65)
    print(f"{'조건':<30} {'CNN acc':>8} {'SSIM':>7} {'Intra':>7} {'Inter':>7}")
    print("-" * 65)
    for k in active_conds:
        accs   = [r[f"cond_{k}"]["overall_cnn_acc"]  for r in all_results if r.get(f"cond_{k}")]
        ssims  = [r[f"cond_{k}"]["overall_ssim_mean"] for r in all_results if r.get(f"cond_{k}")]
        intras = [r[f"cond_{k}"]["intra_ssim"] for r in all_results if r.get(f"cond_{k}") and r[f"cond_{k}"].get("intra_ssim")]
        inters = [r[f"cond_{k}"]["inter_ssim"] for r in all_results if r.get(f"cond_{k}") and r[f"cond_{k}"].get("inter_ssim")]
        avg_acc   = sum(accs)   / len(accs)   if accs   else 0
        avg_ssim  = sum(ssims)  / len(ssims)  if ssims  else 0
        avg_intra = sum(intras) / len(intras) if intras else None
        avg_inter = sum(inters) / len(inters) if inters else None
        intra_s = f"{avg_intra:.4f}" if avg_intra else "  —  "
        inter_s = f"{avg_inter:.4f}" if avg_inter else "  —  "
        label = f"{k}: {CONDITIONS[k]['label']}"
        print(f"  {label:<28} {avg_acc:>7.0%} {avg_ssim:>7.4f} {intra_s:>7} {inter_s:>7}")


if __name__ == "__main__":
    main()
