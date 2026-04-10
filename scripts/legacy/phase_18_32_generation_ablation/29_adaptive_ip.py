"""
Script 29: 클래스별 적응형 IP-Adapter 강도 실험
===============================================
Script 28 결과(ip_scale 0.05~0.20 탐색)를 기반으로,
각 클래스마다 CNN acc ≥ 90% 를 유지하는 최대 ip_scale을 자동 선택.

전략:
  1. Script 28의 summary.json을 읽어 CLASS_IP_MAP 자동 결정
  2. 비교: A 조건 (ip_scale=0.0) vs 적응형 (클래스별 최적 ip_scale)
  3. 목표: 모든 클래스에서 CNN acc ≥ 90% + Inter SSIM < A 조건(0.46) 달성

총 생성: 5cls × 4dom × 5입력 × 3seed × 2조건 = 600장
예상 시간: ~50분

Usage:
    python3 scripts/legacy/phase_18_32_generation_ablation/29_adaptive_ip.py
    python3 scripts/legacy/phase_18_32_generation_ablation/29_adaptive_ip.py --n_inputs 3 --n_seeds 2
    python3 scripts/legacy/phase_18_32_generation_ablation/29_adaptive_ip.py --force_ip_map "basophil:0.20,eosinophil:0.05,lymphocyte:0.05,monocyte:0.10,neutrophil:0.10"
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
SCRIPT28_SUMMARY = ROOT / "results" / "ip_scale_sweep" / "summary.json"
OUT_DIR     = ROOT / "results" / "adaptive_ip"

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

IP_ADAPTER_REPO      = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT    = "ip-adapter_sdxl.bin"

# Script 28 조건 (best ip_scale 탐색에 사용)
SCRIPT28_CONDS = {
    "A":  {"ip_scale": 0.0,  "label": "기준선"},
    "B1": {"ip_scale": 0.05, "label": "IP 극약"},
    "B2": {"ip_scale": 0.10, "label": "IP 약"},
    "B3": {"ip_scale": 0.15, "label": "IP 중약"},
    "C":  {"ip_scale": 0.20, "label": "IP 중"},
}
# 강 → 약 탐색 순서 (가능한 최대 ip_scale 선택)
SEARCH_ORDER = ["C", "B3", "B2", "B1", "A"]

CNN_ACC_THRESHOLD = 0.90  # CNN acc ≥ 90% 유지 기준


# ── Script 28 결과에서 클래스별 최적 ip_scale 자동 결정 ──────────────────────
def find_best_ip_scale(summary: dict, cls: str, threshold: float = CNN_ACC_THRESHOLD) -> float:
    """
    Script 28 summary.json에서 클래스별 'CNN acc ≥ threshold 유지하는 최대 ip_scale' 선택.
    강도 높은 것(C=0.20)부터 내려가며 탐색 → 최초로 기준 만족하는 조건 반환.
    """
    results = summary.get("results", [])
    cls_results = [r for r in results if r["cls"] == cls]

    for cond_key in SEARCH_ORDER:
        if cond_key not in SCRIPT28_CONDS:
            continue
        accs = [
            r[f"cond_{cond_key}"]["overall_cnn_acc"]
            for r in cls_results
            if r.get(f"cond_{cond_key}")
        ]
        if not accs:
            continue
        avg_acc = sum(accs) / len(accs)
        if avg_acc >= threshold:
            return SCRIPT28_CONDS[cond_key]["ip_scale"]

    return 0.0  # fallback: 기준선


def build_class_ip_map(summary: dict, threshold: float = CNN_ACC_THRESHOLD) -> dict:
    """Script 28 결과에서 클래스별 최적 ip_scale 맵을 자동 생성."""
    return {cls: find_best_ip_scale(summary, cls, threshold) for cls in CLASSES}


# ── WBCRouter 로드 ────────────────────────────────────────────────────────────
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


# ── FewShotSampler ────────────────────────────────────────────────────────────
class FewShotSampler:
    """같은 클래스 × 다른 도메인에서 참조 이미지를 랜덤 샘플링."""
    def __init__(self, data_dir: Path, domain_list: list, class_list: list,
                 seed: int = SAMPLING_SEED):
        self.rng = random.Random(seed)
        self._pool: dict = {}
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
    pipe.disable_attention_slicing()
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

    if ref_imgs:
        kwargs["ip_adapter_image"] = [ref_imgs] if len(ref_imgs) > 1 else ref_imgs[0]
    else:
        kwargs["ip_adapter_image"] = ref  # dummy (scale=0.0 이므로 무효)

    with torch.no_grad():
        return pipe(**kwargs).images[0]


# ── Intra / Inter SSIM 계산 ───────────────────────────────────────────────────
def compute_intra_inter_ssim(inp_data_list: list, n_seeds: int,
                              out_images_dir: Path, cond_label: str):
    gen_imgs = {}
    for inp_data in inp_data_list:
        inp_idx = inp_data["inp_idx"]
        cls, dom = inp_data["cls"], inp_data["dom"]
        for s in range(n_seeds):
            p = out_images_dir / cls / dom / f"input_{inp_idx:02d}" / f"cond_{cond_label}_seed_{s:02d}.png"
            if p.exists():
                gen_imgs[(inp_idx, s)] = Image.open(p).convert("RGB")

    intra_vals = []
    for inp_data in inp_data_list:
        inp_idx = inp_data["inp_idx"]
        imgs = [gen_imgs.get((inp_idx, s)) for s in range(n_seeds)]
        imgs = [im for im in imgs if im is not None]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                intra_vals.append(ssim_pair(imgs[i], imgs[j]))

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


def make_gallery(all_results: list, class_ip_map: dict, n_inputs: int, n_seeds: int) -> str:
    lines = []
    cond_keys = ["A", "OPT"]  # A=기준선, OPT=클래스별 최적

    lines += [
        "# WBC 클래스별 적응형 IP-Adapter 실험 갤러리",
        "",
        f"> **Strength:** {STRENGTH} 고정  ",
        f"> **N_INPUTS:** {n_inputs}  ·  **N_SEEDS:** {n_seeds}  ",
        f"> **총 생성:** {len(CLASSES) * len(DOMAINS) * n_inputs * n_seeds * 2}장 (2조건 × 입력 × seed)  ",
        "",
        "## 클래스별 최적 ip_scale 맵",
        "",
        "| 클래스 | 최적 ip_scale | Script 28 근거 |",
        "| :--- | :---: | :--- |",
    ]
    for cls in CLASSES:
        ip = class_ip_map[cls]
        note = "기준선 (IP 없음)" if ip == 0.0 else f"ip={ip} — CNN acc ≥ 90% 유지"
        lines.append(f"| **{cls}** | {ip} | {note} |")

    lines += [
        "",
        "## 비교 조건",
        "",
        "| 조건 | 설명 |",
        "| :--- | :--- |",
        "| **A** | 기준선: ip_scale=0.0 (IP-Adapter 미사용) |",
        "| **OPT** | 클래스별 최적 ip_scale (CLASS_IP_MAP 기반) |",
        "",
        "뱃지: 🟩 CNN ≥ 90% · 🟨 ≥ 67% · 🟥 < 67%",
        "",
        "---",
        "",
    ]

    # ── 클래스별 CNN acc 비교 히트맵 ──────────────────────────────────────────
    lines.append("## 🗺️ 클래스별 CNN acc 비교 (A vs OPT)")
    lines.append("")
    lines.append("| 클래스 | **A** (ip=0.0) | **OPT** (적응형) | OPT ip_scale | Δ CNN acc |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for cls in CLASSES:
        cls_rows = [r for r in all_results if r["cls"] == cls]
        a_accs = [r["cond_A"]["overall_cnn_acc"] for r in cls_rows if r.get("cond_A")]
        o_accs = [r["cond_OPT"]["overall_cnn_acc"] for r in cls_rows if r.get("cond_OPT")]
        avg_a = sum(a_accs) / len(a_accs) if a_accs else 0
        avg_o = sum(o_accs) / len(o_accs) if o_accs else 0
        delta = avg_o - avg_a
        delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
        lines.append(
            f"| **{cls}** | {acc_badge(avg_a)} | {acc_badge(avg_o)} "
            f"| {class_ip_map[cls]} | {delta_str} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Inter SSIM 비교 히트맵 ────────────────────────────────────────────────
    lines.append("## 📊 클래스별 Inter SSIM 비교 (낮을수록 다양)")
    lines.append("")
    lines.append("| 클래스 | **A** (ip=0.0) | **OPT** (적응형) | Δ Inter SSIM |")
    lines.append("| :--- | :---: | :---: | :---: |")

    for cls in CLASSES:
        cls_rows = [r for r in all_results if r["cls"] == cls]
        a_inters = [r["cond_A"]["inter_ssim"] for r in cls_rows
                    if r.get("cond_A") and r["cond_A"].get("inter_ssim")]
        o_inters = [r["cond_OPT"]["inter_ssim"] for r in cls_rows
                    if r.get("cond_OPT") and r["cond_OPT"].get("inter_ssim")]
        avg_a = sum(a_inters) / len(a_inters) if a_inters else None
        avg_o = sum(o_inters) / len(o_inters) if o_inters else None
        if avg_a is not None and avg_o is not None:
            delta = avg_o - avg_a
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            row = f"| **{cls}** | {avg_a:.4f} | {avg_o:.4f} | {delta_str} |"
        else:
            row = f"| **{cls}** | — | — | — |"
        lines.append(row)

    lines.append("")
    lines.append("> **Inter SSIM**: 다른 입력 이미지 간 생성 결과 유사도")
    lines.append("> Script 27/28 기준값: A=0.4595 (ip=0.0)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 목차 ─────────────────────────────────────────────────────────────────
    lines.append("## 목차")
    lines.append("")
    for cls in CLASSES:
        lines.append(f"### {cls} (OPT ip={class_ip_map[cls]})")
        for dom in DOMAINS:
            lines.append(f"- [{cls} × {dom}](#{cls.lower()}-{dom.lower()})")
        lines.append("")
    lines.append("---")
    lines.append("")

    # ── 클래스 × 도메인 상세 섹션 ────────────────────────────────────────────
    for cls in CLASSES:
        opt_ip = class_ip_map[cls]
        lines.append(f"# {cls.upper()}  (OPT ip_scale={opt_ip})")
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
                label = "기준선 (ip=0.0)" if k == "A" else f"적응형 (ip={opt_ip})"
                intra = f"{cd['intra_ssim']:.4f}" if cd.get("intra_ssim") else "—"
                inter = f"{cd['inter_ssim']:.4f}" if cd.get("inter_ssim") else "—"
                lines.append(
                    f"| **{k}: {label}** "
                    f"| {acc_badge(cd.get('overall_cnn_acc', 0))} "
                    f"| {cd.get('overall_ssim_mean', 0):.4f} "
                    f"| {intra} | {inter} |"
                )
            lines.append("")

            # 입력별 블록
            for inp_data in entry["inputs"]:
                inp_idx = inp_data["inp_idx"]
                inp_rel = f"images/{cls}/{dom}/input_{inp_idx:02d}/input.png"
                ref_rel = f"images/{cls}/{dom}/input_{inp_idx:02d}/ref_OPT.png"

                lines.append(f"### 입력 {inp_idx}")
                lines.append("")

                lines.append("| 원본 입력 | 참조 이미지 (OPT 조건) |")
                lines.append("| :---: | :---: |")
                ref_note = "기준선 (IP 없음)" if opt_ip == 0.0 else f"ip={opt_ip} 참조"
                if opt_ip > 0.0:
                    lines.append(f"| ![inp{inp_idx}]({inp_rel}) | ![ref]({ref_rel}) |")
                else:
                    lines.append(f"| ![inp{inp_idx}]({inp_rel}) | *(IP 미사용)* |")
                lines.append("")

                # A vs OPT 비교 테이블
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
                    k_label = "A (기준선)" if k == "A" else f"OPT (ip={opt_ip})"
                    lines.append(
                        f"| **{k_label}** | "
                        + " | ".join(seed_cells)
                        + f" | {acc_badge(acc_val)} | {ssim_val:.4f} |"
                    )

                lines.append("")

            lines.append("---")
            lines.append("")

    # ── 전체 요약 ────────────────────────────────────────────────────────────
    lines.append("# 전체 결과 요약")
    lines.append("")
    lines.append("| 클래스 | 도메인 | OPT ip | A CNN | OPT CNN | A Inter | OPT Inter |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in all_results:
        a_acc  = r.get("cond_A", {}).get("overall_cnn_acc", 0)
        o_acc  = r.get("cond_OPT", {}).get("overall_cnn_acc", 0)
        a_int  = r.get("cond_A", {}).get("inter_ssim")
        o_int  = r.get("cond_OPT", {}).get("inter_ssim")
        opt_ip = class_ip_map.get(r["cls"], 0.0)
        lines.append(
            f"| {r['cls']} | {r['dom']} | {opt_ip} "
            f"| {acc_badge(a_acc)} | {acc_badge(o_acc)} "
            f"| {a_int:.4f if a_int else '—'} "
            f"| {o_int:.4f if o_int else '—'} |"
        )

    lines.append("")
    lines.append("## 조건별 전체 평균")
    lines.append("")
    lines.append("| 조건 | CNN acc | SSIM vs 입력 | Intra SSIM | Inter SSIM |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for k in cond_keys:
        accs   = [r[f"cond_{k}"]["overall_cnn_acc"]  for r in all_results if r.get(f"cond_{k}")]
        ssims  = [r[f"cond_{k}"]["overall_ssim_mean"] for r in all_results if r.get(f"cond_{k}")]
        intras = [r[f"cond_{k}"]["intra_ssim"] for r in all_results
                  if r.get(f"cond_{k}") and r[f"cond_{k}"].get("intra_ssim")]
        inters = [r[f"cond_{k}"]["inter_ssim"] for r in all_results
                  if r.get(f"cond_{k}") and r[f"cond_{k}"].get("inter_ssim")]
        avg_acc   = sum(accs)   / len(accs)   if accs   else 0
        avg_ssim  = sum(ssims)  / len(ssims)  if ssims  else 0
        avg_intra = sum(intras) / len(intras) if intras else None
        avg_inter = sum(inters) / len(inters) if inters else None
        intra_s = f"{avg_intra:.4f}" if avg_intra is not None else "  —  "
        inter_s = f"{avg_inter:.4f}" if avg_inter is not None else "  —  "
        label = "A: 기준선 (ip=0.0)" if k == "A" else "OPT: 클래스별 적응형"
        lines.append(
            f"| **{label}** "
            f"| {acc_badge(avg_acc)} | {avg_ssim:.4f} "
            f"| {intra_s} | {inter_s} |"
        )

    lines.append("")
    lines.append("> **목표**: CNN acc ≥ 90% (A 대비 유지) + Inter SSIM < 0.46 (A 대비 감소)  ")
    lines.append("> **Script 27 기준**: A=99% CNN/0.46 Inter · B(ip=0.2,ref=1)=59%/0.13  ")

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
    parser.add_argument("--n_inputs",    type=int,   default=5,  help="입력 이미지 수/cls×dom")
    parser.add_argument("--n_seeds",     type=int,   default=3,  help="seed 수/입력")
    parser.add_argument("--seed",        type=int,   default=SAMPLING_SEED)
    parser.add_argument("--strength",    type=float, default=STRENGTH)
    parser.add_argument("--threshold",   type=float, default=CNN_ACC_THRESHOLD,
                        help="CLASS_IP_MAP 결정 시 CNN acc 기준값 (기본: 0.90)")
    parser.add_argument("--force_ip_map", type=str, default="",
                        help="CLASS_IP_MAP 수동 설정 (예: 'basophil:0.20,eosinophil:0.05')")
    args = parser.parse_args()

    N_INPUTS = args.n_inputs
    N_SEEDS  = args.n_seeds
    DENOISE  = args.strength
    n_total  = len(CLASSES) * len(DOMAINS) * N_INPUTS * N_SEEDS * 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(exist_ok=True)

    print("=" * 65)
    print("Script 29: 클래스별 적응형 IP-Adapter 실험")
    print(f"  N_INPUTS={N_INPUTS}, N_SEEDS={N_SEEDS}, strength={DENOISE}")
    print(f"  CNN acc 기준: ≥ {args.threshold:.0%}")
    print(f"  총 생성 예정: {n_total}장")
    print("=" * 65)

    # ── CLASS_IP_MAP 결정 ─────────────────────────────────────────────────────
    if args.force_ip_map:
        # 수동 설정: 'basophil:0.20,eosinophil:0.05,...'
        class_ip_map = {}
        for item in args.force_ip_map.split(","):
            cls_name, ip_str = item.strip().split(":")
            class_ip_map[cls_name.strip()] = float(ip_str.strip())
        print("\n[CLASS_IP_MAP] 수동 설정:")
    elif SCRIPT28_SUMMARY.exists():
        with open(SCRIPT28_SUMMARY, "r", encoding="utf-8") as f:
            script28_summary = json.load(f)
        class_ip_map = build_class_ip_map(script28_summary, threshold=args.threshold)
        print("\n[CLASS_IP_MAP] Script 28 결과에서 자동 결정:")
    else:
        print(f"\n⚠️  Script 28 summary.json 없음: {SCRIPT28_SUMMARY}")
        print("   기본값 사용 (basophil=0.20, 나머지=0.05)")
        class_ip_map = {
            "basophil":   0.20,
            "eosinophil": 0.05,
            "lymphocyte": 0.05,
            "monocyte":   0.05,
            "neutrophil": 0.05,
        }

    for cls, ip in class_ip_map.items():
        print(f"  {cls:<12}: ip_scale = {ip}")

    # ── 조건 동적 설정 ────────────────────────────────────────────────────────
    # A: 모든 클래스 ip=0.0 (기준선)
    # OPT: 클래스별 class_ip_map[cls] 사용
    CONDITIONS_29 = {
        "A":   {"ip_scale": 0.0, "n_ref": 0, "label": "기준선 (IP 없음)"},
        "OPT": {"ip_scale": None, "n_ref": 1, "label": "클래스별 적응형"},  # ip_scale은 동적
    }
    active_conds = list(CONDITIONS_29.keys())

    # ── 1. 모델 로드 ──────────────────────────────────────────────────────────
    print("\n[1/4] WBCRouter 및 CNN 로드...")
    mod = load_router_module()
    WBCRouter          = mod.WBCRouter
    build_prompt       = mod.build_class_domain_prompt
    MULTI_CLASSES_LIST = mod.MULTI_CLASSES
    DOMAIN_LIST        = mod.DOMAINS
    NEGATIVE_PROMPT    = mod.NEGATIVE_PROMPT
    get_device         = mod.get_device

    device = get_device()
    router = WBCRouter(router_ckpt=ROUTER_CKPT, cnn_ckpt=CNN_CKPT, device=device)

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

    input_pool = {}
    ref_pool   = {}

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
                opt_ip = class_ip_map.get(cls, 0.0)
                if opt_ip > 0.0:
                    ref_pool[(cls, dom_key, inp_idx)] = sampler.sample(
                        cls, exclude_domain=dom_key, n=1
                    )
                else:
                    ref_pool[(cls, dom_key, inp_idx)] = []

    print(f"  입력 샘플링 완료: {len(input_pool)}개 cls×dom")

    # ── 4. 생성 루프 ──────────────────────────────────────────────────────────
    print("\n[4/4] 이미지 생성 및 평가...")
    all_results = []
    done = 0
    t0   = time.time()

    def _inp_is_complete(inp_dir: Path, n_seeds: int) -> bool:
        for k in active_conds:
            for s in range(n_seeds):
                if not (inp_dir / f"cond_{k}_seed_{s:02d}.png").exists():
                    return False
        return True

    for cls in CLASSES:
        opt_ip = class_ip_map.get(cls, 0.0)

        for dom in DOMAINS:
            dom_key  = DOMAIN_MAP[dom]
            inputs   = input_pool.get((cls, dom), [])
            if not inputs:
                continue

            dom_idx  = DOMAIN_LIST.index(dom_key) if dom_key in DOMAIN_LIST else 0
            cls_idx  = MULTI_CLASSES_LIST.index(cls) if cls in MULTI_CLASSES_LIST else 0
            prompt   = build_prompt(cls_idx, dom_idx)

            print(f"\n── {cls}(OPT ip={opt_ip}) × {dom} ({len(inputs)}장 입력) ──")

            inp_data_list = []

            for inp_idx, inp_path in enumerate(inputs):
                input_img = Image.open(inp_path).convert("RGB")

                inp_dir = OUT_DIR / "images" / cls / dom / f"input_{inp_idx:02d}"
                inp_dir.mkdir(parents=True, exist_ok=True)
                input_img.save(inp_dir / "input.png")

                shared_refs = ref_pool.get((cls, dom_key, inp_idx), [])
                if shared_refs:
                    shared_refs[0].save(inp_dir / "ref_OPT.png")

                # ── 체크포인트 ────────────────────────────────────────────────
                if _inp_is_complete(inp_dir, N_SEEDS):
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
                        "opt_ip":   opt_ip,
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
                        # OPT 조건의 ip_scale은 클래스별 class_ip_map 사용
                        if k == "A":
                            ip_scale = 0.0
                            ref_imgs = []
                        else:  # OPT
                            ip_scale = opt_ip
                            ref_imgs = shared_refs[:1] if opt_ip > 0.0 else []

                        save_path = inp_dir / f"cond_{k}_seed_{seed_offset:02d}.png"
                        if save_path.exists():
                            gen = Image.open(save_path).convert("RGB")
                        else:
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

                inp_entry = {
                    "inp_idx":  inp_idx,
                    "inp_path": str(inp_path),
                    "cls":      cls,
                    "dom":      dom,
                    "opt_ip":   opt_ip,
                }
                for k in active_conds:
                    agg = aggregate_seeds(cond_seeds[k])
                    inp_entry[f"cond_{k}"] = agg
                    inp_entry[f"seeds_{k}"] = cond_seeds[k]

                accs_str = " | ".join(
                    f"{k}={aggregate_seeds(cond_seeds[k])['cnn_acc']:.0%}"
                    for k in active_conds
                )
                print(f" [{done}/{n_total}] {accs_str}")

                inp_data_list.append(inp_entry)

            # cls×dom 수준 집계
            result_entry = {
                "cls":      cls,
                "dom":      dom,
                "dom_key":  dom_key,
                "opt_ip":   opt_ip,
                "n_inputs": len(inputs),
                "inputs":   inp_data_list,
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
        "cnn_acc_threshold": args.threshold,
        "class_ip_map": class_ip_map,
        "script28_summary_used": str(SCRIPT28_SUMMARY),
        "n_total":      done,
        "elapsed_min":  round(elapsed / 60, 1),
        "results":      all_results,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  summary.json → {OUT_DIR / 'summary.json'}")

    md = make_gallery(all_results, class_ip_map, N_INPUTS, N_SEEDS)
    (OUT_DIR / "gallery.md").write_text(md, encoding="utf-8")
    print(f"  gallery.md   → {OUT_DIR / 'gallery.md'}")

    # 콘솔 요약
    print("\n" + "=" * 65)
    print("클래스별 결과 요약 (A vs OPT 비교)")
    print("=" * 65)
    print(f"{'클래스':<14} {'OPT ip':>8} {'A CNN':>8} {'OPT CNN':>8} {'A Inter':>9} {'OPT Inter':>10}")
    print("-" * 65)
    for cls in CLASSES:
        cls_rows = [r for r in all_results if r["cls"] == cls]
        a_accs  = [r["cond_A"]["overall_cnn_acc"] for r in cls_rows if r.get("cond_A")]
        o_accs  = [r["cond_OPT"]["overall_cnn_acc"] for r in cls_rows if r.get("cond_OPT")]
        a_ints  = [r["cond_A"]["inter_ssim"] for r in cls_rows
                   if r.get("cond_A") and r["cond_A"].get("inter_ssim")]
        o_ints  = [r["cond_OPT"]["inter_ssim"] for r in cls_rows
                   if r.get("cond_OPT") and r["cond_OPT"].get("inter_ssim")]
        avg_a_acc = sum(a_accs) / len(a_accs) if a_accs else 0
        avg_o_acc = sum(o_accs) / len(o_accs) if o_accs else 0
        avg_a_int = sum(a_ints) / len(a_ints) if a_ints else None
        avg_o_int = sum(o_ints) / len(o_ints) if o_ints else None
        a_int_s = f"{avg_a_int:.4f}" if avg_a_int is not None else "  —  "
        o_int_s = f"{avg_o_int:.4f}" if avg_o_int is not None else "  —  "
        opt_ip  = class_ip_map.get(cls, 0.0)
        print(f"  {cls:<12} {opt_ip:>8.2f} {avg_a_acc:>8.0%} {avg_o_acc:>8.0%} {a_int_s:>9} {o_int_s:>10}")

    print("\n" + "=" * 65)
    print("조건별 전체 평균")
    print("=" * 65)
    print(f"{'조건':<30} {'CNN acc':>8} {'SSIM':>7} {'Intra':>7} {'Inter':>7}")
    print("-" * 65)
    for k in active_conds:
        accs   = [r[f"cond_{k}"]["overall_cnn_acc"]  for r in all_results if r.get(f"cond_{k}")]
        ssims  = [r[f"cond_{k}"]["overall_ssim_mean"] for r in all_results if r.get(f"cond_{k}")]
        intras = [r[f"cond_{k}"]["intra_ssim"] for r in all_results
                  if r.get(f"cond_{k}") and r[f"cond_{k}"].get("intra_ssim")]
        inters = [r[f"cond_{k}"]["inter_ssim"] for r in all_results
                  if r.get(f"cond_{k}") and r[f"cond_{k}"].get("inter_ssim")]
        avg_acc   = sum(accs)   / len(accs)   if accs   else 0
        avg_ssim  = sum(ssims)  / len(ssims)  if ssims  else 0
        avg_intra = sum(intras) / len(intras) if intras else None
        avg_inter = sum(inters) / len(inters) if inters else None
        intra_s = f"{avg_intra:.4f}" if avg_intra is not None else "  —  "
        inter_s = f"{avg_inter:.4f}" if avg_inter is not None else "  —  "
        label = "A: 기준선 (ip=0.0)" if k == "A" else "OPT: 클래스별 적응형"
        print(f"  {label:<28} {avg_acc:>7.0%} {avg_ssim:>7.4f} {intra_s:>7} {inter_s:>7}")


if __name__ == "__main__":
    main()
