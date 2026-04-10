"""
Script 19: Basophil 2-Input × 10-Gen Gallery
=============================================
두 도메인에서 basophil 이미지 각 1장씩(총 2장)을 입력으로,
WBCRouter를 통해 각 10장씩 img2img 생성(총 20장) 후
마크다운 갤러리 + 상세 품질 분석 리포트를 출력한다.

Usage:
    python scripts/legacy/phase_18_32_generation_ablation/19_basophil_gallery.py \\
        --output_dir results/basophil_gallery/
"""

from __future__ import annotations

import importlib.util
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import models, transforms

# ── 경로 ──────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed_multidomain"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# ── WBCRouter 동적 임포트 ─────────────────────────────────────────────
def _load_script15():
    spec = importlib.util.spec_from_file_location(
        "router_inference",
        ROOT / "scripts" / "15_router_inference.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── CNN 평가 모델 ─────────────────────────────────────────────────────
class MultidomainCNN(nn.Module):
    def __init__(self, ckpt_path: Path, device_str: str):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location=device_str, weights_only=False)
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, 5)
        base.load_state_dict(ckpt["model_state_dict"])
        self.model = base
        self.class_names = ckpt.get(
            "class_names",
            ["basophil","eosinophil","lymphocyte","monocyte","neutrophil"],
        )
        self.eval()

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def predict_batch(self, imgs: list[Image.Image], device: str):
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        tensors = torch.stack([tf(im.convert("RGB")) for im in imgs]).to(device)
        logits  = self.forward(tensors.to(next(self.model.parameters()).device))
        probs   = F.softmax(logits, dim=1).cpu()
        preds   = probs.argmax(dim=1).tolist()
        confs   = probs.max(dim=1).values.tolist()
        all_probs = probs.tolist()
        return preds, confs, all_probs


# ── Embedding 추출 ────────────────────────────────────────────────────
class EmbExtractor(nn.Module):
    def __init__(self, ckpt_path: Path, device_str: str):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location=device_str, weights_only=False)
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, 5)
        base.load_state_dict(ckpt["model_state_dict"])
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.eval()

    def forward(self, x):
        return self.backbone(x).flatten(1)

    @torch.no_grad()
    def embed_images(self, imgs: list[Image.Image], device: str, batch_size=32) -> np.ndarray:
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        all_embs = []
        for i in range(0, len(imgs), batch_size):
            batch = torch.stack([tf(im.convert("RGB")) for im in imgs[i:i+batch_size]]).to(device)
            all_embs.append(self.forward(batch).cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    @torch.no_grad()
    def embed_paths(self, paths: list[Path], device: str) -> np.ndarray:
        imgs = [Image.open(p).convert("RGB") for p in paths]
        return self.embed_images(imgs, device)


# ── FrechetDistance ───────────────────────────────────────────────────
def frechet_distance(real_embs: np.ndarray, gen_embs: np.ndarray) -> float:
    if real_embs.shape[0] < 2 or gen_embs.shape[0] < 2:
        return float("nan")
    mu_r, mu_g = real_embs.mean(0), gen_embs.mean(0)
    cov_r = np.cov(real_embs, rowvar=False) + np.eye(real_embs.shape[1]) * 1e-6
    cov_g = np.cov(gen_embs,  rowvar=False) + np.eye(gen_embs.shape[1]) * 1e-6
    diff  = mu_r - mu_g
    covmean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return round(float(diff @ diff + np.trace(cov_r + cov_g - 2.0 * covmean)), 4)


def nn_cosine(gen_embs: np.ndarray, real_embs: np.ndarray) -> tuple[np.ndarray, float]:
    gn = gen_embs  / (np.linalg.norm(gen_embs,  axis=1, keepdims=True) + 1e-8)
    rn = real_embs / (np.linalg.norm(real_embs, axis=1, keepdims=True) + 1e-8)
    sims = gn @ rn.T  # (n_gen, n_real)
    max_sims = sims.max(axis=1)
    return max_sims, round(float(max_sims.mean()), 4)


# ── 색상/텍스처 분석 ─────────────────────────────────────────────────
def color_stats(img: Image.Image) -> dict:
    """RGB 채널별 mean/std + HSV Value mean."""
    arr = np.array(img.resize((224, 224)).convert("RGB")).astype(float)
    stats = {}
    for i, ch in enumerate(["R", "G", "B"]):
        stats[f"{ch}_mean"] = round(float(arr[:,:,i].mean()), 2)
        stats[f"{ch}_std"]  = round(float(arr[:,:,i].std()),  2)
    # Brightness (V channel of HSV)
    hsv_v = arr.max(axis=2) / 255.0
    stats["brightness_mean"] = round(float(hsv_v.mean()), 4)
    stats["brightness_std"]  = round(float(hsv_v.std()),  4)
    return stats


def sharpness_laplacian(img: Image.Image) -> float:
    """Laplacian variance → 선명도 (높을수록 선명)."""
    gray = np.array(img.resize((224, 224)).convert("L")).astype(float)
    # 3×3 Laplacian kernel
    lap = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
    from scipy.ndimage import convolve
    filtered = convolve(gray, lap)
    return round(float(filtered.var()), 2)


def compute_psnr(img_a: Image.Image, img_b: Image.Image, size=224) -> float:
    """PSNR (dB) between two images (higher is more similar)."""
    a = np.array(img_a.resize((size, size))).astype(float)
    b = np.array(img_b.resize((size, size))).astype(float)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return 100.0
    return round(float(20 * np.log10(255.0 / np.sqrt(mse))), 2)


# ── 마크다운 갤러리 생성 ─────────────────────────────────────────────
DOMAIN_LABELS = {
    "domain_a_pbc":    "PBC · Spain · May-Grünwald Giemsa / CellaVision",
    "domain_b_raabin": "Raabin · Iran · Giemsa / 스마트폰 현미경",
    "domain_c_mll23":  "MLL23 · Germany · Pappenheim / Metafer",
    "domain_e_amc":    "AMC · Korea · Romanowsky / miLab",
}
DOMAIN_SHORT = {
    "domain_a_pbc": "PBC",
    "domain_b_raabin": "Raabin",
    "domain_c_mll23": "MLL23",
    "domain_e_amc": "AMC",
}


def make_gallery_md(
    out_dir: Path,
    inputs: list[dict],
    timestamp: str,
) -> str:
    """
    inputs: list of {
      domain, domain_label, input_path, input_rel,
      pred_class, class_conf, domain_conf, routing_mode, prompt,
      generated: list of {
        path_rel, ssim, psnr, sharpness, color_stats,
        cnn_pred, cnn_conf, cnn_probs, nn_cos
      }
    }
    """
    lines = [
        "# Basophil 생성 이미지 갤러리 리포트",
        "",
        f"> **생성 일시:** {timestamp}  ",
        "> **대상 클래스:** `basophil`  ",
        "> **입력:** 2개 도메인 각 1장 (총 2장)  ",
        "> **생성:** 각 입력당 10장 img2img (총 20장)  ",
        "> **라우터:** `dual_head_router.pt` (DualHeadRouter, TTA 활성)  ",
        "> **생성 모델:** SDXL img2img + 멀티도메인 전문가 LoRA (`multidomain_basophil`)  ",
        "> **img2img strength:** 0.35 (원본 스타일 ~65% 보존)",
        "",
        "---",
        "",
        "## 목차",
        "1. [입력 이미지 및 라우터 분류 결과](#1-입력-이미지-및-라우터-분류-결과)",
        "2. [생성 이미지 갤러리](#2-생성-이미지-갤러리)",
        "3. [품질 지표 상세 분석](#3-품질-지표-상세-분석)",
        "4. [도메인별 비교](#4-도메인별-비교)",
        "5. [생성 프롬프트](#5-생성-프롬프트)",
        "",
        "---",
        "",
        "## 1. 입력 이미지 및 라우터 분류 결과",
        "",
    ]

    for inp in inputs:
        dom_short = DOMAIN_SHORT.get(inp["domain"], inp["domain"])
        lines += [
            f"### 입력 [{inp['idx']+1}] — {DOMAIN_LABELS.get(inp['domain'], inp['domain'])}",
            "",
            f"| 항목 | 값 |",
            f"|------|-----|",
            f"| 원본 파일 | `{Path(inp['input_path']).name}` |",
            f"| 예측 클래스 | **{inp['pred_class']}** (conf: {inp['class_conf']:.4f}) |",
            f"| 예측 도메인 | **{dom_short}** (conf: {inp['domain_conf']:.4f}) |",
            f"| Routing mode | `{inp['routing_mode']}` |",
            "",
            f"<img src=\"{inp['input_rel']}\" width=\"256\" alt=\"Input {inp['idx']+1}\">",
            "",
        ]

    lines += [
        "---",
        "",
        "## 2. 생성 이미지 갤러리",
        "",
    ]

    for inp in inputs:
        dom_short = DOMAIN_SHORT.get(inp["domain"], inp["domain"])
        lines += [
            f"### 입력 [{inp['idx']+1}] → 생성 10장 ({dom_short} 도메인 basophil)",
            "",
            "#### 원본 입력",
            f"<img src=\"{inp['input_rel']}\" width=\"180\" alt=\"원본\">",
            "",
            "#### 생성 이미지 (seed 0–9)",
            "",
        ]

        # 5열 × 2행
        gen_list = inp["generated"]
        row1 = gen_list[:5]
        row2 = gen_list[5:]

        def make_row(gens):
            cells = []
            for g in gens:
                cells.append(
                    f'<img src="{g["path_rel"]}" width="160" '
                    f'title="seed={g["seed"]} SSIM={g["ssim"]:.4f}">'
                )
            return "| " + " | ".join(cells) + " |"

        def make_sep(n):
            return "| " + " | ".join(["---"] * n) + " |"

        def make_label_row(gens):
            cells = [f"seed={g['seed']}" for g in gens]
            return "| " + " | ".join(cells) + " |"

        lines += [
            make_row(row1),
            make_sep(5),
            make_label_row(row1),
            "",
            make_row(row2),
            make_sep(5),
            make_label_row(row2),
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. 품질 지표 상세 분석",
        "",
        "### 3-1. 구조적 유사도 & PSNR",
        "",
        "> **SSIM** (0~1): 구조·밝기·대비 종합 유사도. 1에 가까울수록 원본 보존.  ",
        "> **PSNR** (dB): 픽셀 레벨 신호 대 노이즈 비율. 높을수록 원본과 유사.  ",
        "> strength=0.35이므로 SSIM≈0.97~0.99, PSNR≈26~32dB 예상.",
        "",
    ]

    for inp in inputs:
        dom_short = DOMAIN_SHORT.get(inp["domain"], inp["domain"])
        ssims = [g["ssim"] for g in inp["generated"]]
        psnrs = [g["psnr"] for g in inp["generated"]]
        lines += [
            f"#### 입력 [{inp['idx']+1}] ({dom_short})",
            "",
            "| Seed | SSIM | PSNR (dB) | 평가 |",
            "|------|------|-----------|------|",
        ]
        for g in inp["generated"]:
            quality = "🟢 우수" if g["ssim"] >= 0.985 else ("🟡 양호" if g["ssim"] >= 0.975 else "🔴 보통")
            lines.append(
                f"| {g['seed']} | {g['ssim']:.4f} | {g['psnr']:.2f} | {quality} |"
            )
        lines += [
            f"| **평균** | **{np.mean(ssims):.4f}** | **{np.mean(psnrs):.2f}** | — |",
            f"| **std** | {np.std(ssims):.4f} | {np.std(psnrs):.2f} | — |",
            "",
        ]

    lines += [
        "### 3-2. CNN 분류 신뢰도 (multidomain_cnn.pt)",
        "",
        "> 생성 이미지를 독립 CNN으로 재분류. **basophil**로 예측되면 성공.  ",
        "> confidence가 높을수록 클래스 특성이 잘 보존됨.",
        "",
    ]

    CLASS_NAMES = ["basophil","eosinophil","lymphocyte","monocyte","neutrophil"]
    for inp in inputs:
        dom_short = DOMAIN_SHORT.get(inp["domain"], inp["domain"])
        lines += [
            f"#### 입력 [{inp['idx']+1}] ({dom_short})",
            "",
            "| Seed | CNN 예측 | conf | basophil prob | 판정 |",
            "|------|---------|------|--------------|------|",
        ]
        for g in inp["generated"]:
            ok = "✅" if g["cnn_pred"] == 0 else "❌"  # 0=basophil
            bp = g["cnn_probs"][0] if g["cnn_probs"] else 0.0
            lines.append(
                f"| {g['seed']} | {CLASS_NAMES[g['cnn_pred']]} "
                f"| {g['cnn_conf']:.4f} | {bp:.4f} | {ok} |"
            )
        acc_10 = sum(1 for g in inp["generated"] if g["cnn_pred"] == 0)
        lines += [
            f"| **집계** | **{acc_10}/10 correct** | — | — | — |",
            "",
        ]

    lines += [
        "### 3-3. NN Cosine Similarity (real basophil pool vs 생성)",
        "",
        "> 생성 이미지 embedding과 real basophil pool의 최근접 이웃 cosine 유사도.  ",
        "> 1에 가까울수록 실제 basophil 분포에 근접.",
        "",
        "| 입력 | NN cosine mean | min | max |",
        "|------|---------------|-----|-----|",
    ]
    for inp in inputs:
        dom_short = DOMAIN_SHORT.get(inp["domain"], inp["domain"])
        nn_vals = [g["nn_cos"] for g in inp["generated"]]
        lines.append(
            f"| 입력 [{inp['idx']+1}] ({dom_short}) "
            f"| {np.mean(nn_vals):.4f} | {np.min(nn_vals):.4f} | {np.max(nn_vals):.4f} |"
        )
    lines.append("")

    lines += [
        "### 3-4. FrechetDistance (real basophil vs 생성 전체)",
        "",
        "> 두 분포(real vs gen)의 통계적 거리. 낮을수록 실제 데이터 분포에 가깝다.",
        "",
        "| 비교 | FD |",
        "|------|-----|",
    ]
    for inp in inputs:
        dom_short = DOMAIN_SHORT.get(inp["domain"], inp["domain"])
        lines.append(f"| 입력 [{inp['idx']+1}] ({dom_short}) 생성 10장 vs real 200장 | {inp['fd']:.4f} |")
    # 전체 20장
    lines += [
        f"| 전체 20장 vs real 200장 | {inputs[0].get('fd_total', 'N/A')} |",
        "",
    ]

    lines += [
        "### 3-5. 색상 및 선명도 분석",
        "",
        "> **Sharpness** (Laplacian variance): 높을수록 선명한 이미지.  ",
        "> **Brightness**: HSV Value 채널 평균 (0~1).",
        "",
    ]

    for inp in inputs:
        dom_short = DOMAIN_SHORT.get(inp["domain"], inp["domain"])
        lines += [
            f"#### 입력 [{inp['idx']+1}] ({dom_short})",
            "",
            "| Seed | Sharpness | Brightness | R mean | G mean | B mean |",
            "|------|-----------|-----------|--------|--------|--------|",
        ]
        # 원본 먼저
        ic = inp["input_color"]
        isp = inp["input_sharpness"]
        lines.append(
            f"| **원본** | **{isp:.1f}** | **{ic['brightness_mean']:.4f}** "
            f"| {ic['R_mean']:.1f} | {ic['G_mean']:.1f} | {ic['B_mean']:.1f} |"
        )
        for g in inp["generated"]:
            c = g["color_stats"]
            lines.append(
                f"| {g['seed']} | {g['sharpness']:.1f} | {c['brightness_mean']:.4f} "
                f"| {c['R_mean']:.1f} | {c['G_mean']:.1f} | {c['B_mean']:.1f} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 4. 도메인별 비교",
        "",
        "두 입력 도메인의 생성 품질 차이를 비교한다.",
        "",
        "| 지표 | "
        + " | ".join(
            f"입력 [{inp['idx']+1}] ({DOMAIN_SHORT.get(inp['domain'], inp['domain'])})"
            for inp in inputs
        ) + " |",
        "|------|" + "|".join(["---"] * len(inputs)) + "|",
    ]

    def inp_stat(key_fn):
        return " | ".join(f"{key_fn(inp)}" for inp in inputs)

    ssim_means = [f"{np.mean([g['ssim'] for g in inp['generated']]):.4f}" for inp in inputs]
    psnr_means = [f"{np.mean([g['psnr'] for g in inp['generated']]):.2f} dB" for inp in inputs]
    cnn_accs   = [f"{sum(1 for g in inp['generated'] if g['cnn_pred']==0)}/10" for inp in inputs]
    cnn_confs  = [f"{np.mean([g['cnn_conf'] for g in inp['generated']]):.4f}" for inp in inputs]
    nn_means   = [f"{np.mean([g['nn_cos'] for g in inp['generated']]):.4f}" for inp in inputs]
    sharp_m    = [f"{np.mean([g['sharpness'] for g in inp['generated']]):.1f}" for inp in inputs]
    fds        = [f"{inp['fd']:.4f}" for inp in inputs]

    rows = [
        ("SSIM mean", ssim_means),
        ("PSNR mean", psnr_means),
        ("CNN accuracy", cnn_accs),
        ("CNN confidence mean", cnn_confs),
        ("NN cosine mean", nn_means),
        ("Sharpness mean", sharp_m),
        ("FrechetDistance", fds),
    ]
    for label, vals in rows:
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "---",
        "",
        "## 5. 생성 프롬프트",
        "",
    ]
    for inp in inputs:
        dom_short = DOMAIN_SHORT.get(inp["domain"], inp["domain"])
        lines += [
            f"### 입력 [{inp['idx']+1}] ({dom_short})",
            "",
            f"```",
            inp["prompt"],
            f"```",
            "",
        ]

    lines += [
        "---",
        "",
        "## 6. 종합 평가 요약",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        f"| 총 생성 이미지 수 | 20장 (입력 2×10) |",
    ]
    all_ssims = [g["ssim"] for inp in inputs for g in inp["generated"]]
    all_psnrs = [g["psnr"] for inp in inputs for g in inp["generated"]]
    all_confs = [g["cnn_conf"] for inp in inputs for g in inp["generated"]]
    all_nn    = [g["nn_cos"] for inp in inputs for g in inp["generated"]]
    all_sharp = [g["sharpness"] for inp in inputs for g in inp["generated"]]
    total_acc = sum(1 for inp in inputs for g in inp["generated"] if g["cnn_pred"] == 0)
    lines += [
        f"| SSIM mean (전체) | {np.mean(all_ssims):.4f} ± {np.std(all_ssims):.4f} |",
        f"| PSNR mean (전체) | {np.mean(all_psnrs):.2f} ± {np.std(all_psnrs):.2f} dB |",
        f"| CNN accuracy (전체) | {total_acc}/20 ({total_acc/20:.0%}) |",
        f"| CNN confidence mean | {np.mean(all_confs):.4f} |",
        f"| NN cosine mean | {np.mean(all_nn):.4f} |",
        f"| Sharpness mean | {np.mean(all_sharp):.1f} |",
        "",
        "### 종합 해석",
        "",
    ]

    # 자동 해석
    interp = []
    ssim_m = np.mean(all_ssims)
    if ssim_m >= 0.985:
        interp.append(f"- **SSIM {ssim_m:.4f}**: 매우 높음 → strength=0.35 설정으로 원본 도말 스타일(색조·배경)이 98.5% 이상 보존됨.")
    elif ssim_m >= 0.97:
        interp.append(f"- **SSIM {ssim_m:.4f}**: 높음 → 원본 구조를 잘 유지하면서 세포 형태가 강화됨.")
    else:
        interp.append(f"- **SSIM {ssim_m:.4f}**: 보통 → 생성 과정에서 스타일 변화가 발생함.")

    if total_acc == 20:
        interp.append(f"- **CNN accuracy 100%**: 생성된 20장 전부 `basophil`로 올바르게 분류 → LoRA 전문가 클래스 정체성 완벽 유지.")
    elif total_acc >= 18:
        interp.append(f"- **CNN accuracy {total_acc}/20**: 대부분의 생성 이미지가 basophil로 분류됨.")
    else:
        interp.append(f"- **CNN accuracy {total_acc}/20**: 일부 이미지에서 클래스 정체성 손실 발생.")

    nn_m = np.mean(all_nn)
    if nn_m >= 0.90:
        interp.append(f"- **NN cosine {nn_m:.4f}**: 생성 이미지가 실제 basophil 분포와 매우 근접. 데이터 증강 적합성 우수.")
    elif nn_m >= 0.80:
        interp.append(f"- **NN cosine {nn_m:.4f}**: 실제 데이터 분포와 유사하나 일부 다양성 존재.")
    else:
        interp.append(f"- **NN cosine {nn_m:.4f}**: 생성 이미지가 실제 분포와 다소 거리가 있음.")

    lines += interp
    lines += [
        "",
        "---",
        "",
        f"> 생성된 이미지: `results/basophil_gallery/`  ",
        f"> 평가 모델: `models/multidomain_cnn.pt` (F1=0.9917), `models/dual_head_router.pt` (class_acc=99.15%)",
    ]

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",  type=Path,  default=ROOT / "results" / "basophil_gallery")
    p.add_argument("--router_ckpt", type=Path,  default=ROOT / "models" / "dual_head_router.pt")
    p.add_argument("--cnn_ckpt",    type=Path,  default=ROOT / "models" / "multidomain_cnn.pt")
    p.add_argument("--denoise",     type=float, default=0.35)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--n_gen",       type=int,   default=10)
    # 고정 입력: PBC 도메인과 AMC 도메인 각 1장
    p.add_argument("--domain_a",    type=str,   default="domain_a_pbc",
                   help="첫 번째 입력 도메인")
    p.add_argument("--domain_b",    type=str,   default="domain_e_amc",
                   help="두 번째 입력 도메인")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*65}")
    print(f"  Script 19 — Basophil Gallery [{ts}]")
    print(f"{'='*65}")

    # ── 입력 이미지 선택 ──────────────────────────────────────────────
    rng = random.Random(args.seed)
    selected_inputs = []
    for domain in [args.domain_a, args.domain_b]:
        cls_dir = DATA_DIR / domain / "basophil"
        all_imgs = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        chosen   = rng.choice(all_imgs)
        selected_inputs.append((chosen, domain))
        print(f"  선택: {domain}/basophil/{chosen.name}")

    # ── WBCRouter 초기화 ──────────────────────────────────────────────
    print("\n[Step 1] WBCRouter 초기화...")
    mod15   = _load_script15()
    WBCRouter = mod15.WBCRouter
    ssim_pair = mod15.ssim_pair
    device  = mod15.get_device()
    router  = WBCRouter(
        router_ckpt=args.router_ckpt if args.router_ckpt.exists() else None,
        cnn_ckpt=args.cnn_ckpt,
        device=device,
    )

    # ── CNN & Embedding 모델 로드 ─────────────────────────────────────
    print("\n[Step 2] CNN/Embedding 모델 로드...")
    cnn_model = MultidomainCNN(args.cnn_ckpt, device).to(device)
    emb_model = EmbExtractor(args.cnn_ckpt, device).to(device)

    # Real basophil pool (FD용, 200장 max)
    real_pool_paths = []
    for d in sorted(DATA_DIR.iterdir()):
        cls_dir = d / "basophil"
        if cls_dir.exists():
            ps = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            real_pool_paths.extend(ps)
    rng2 = random.Random(args.seed + 99)
    if len(real_pool_paths) > 200:
        real_pool_paths = rng2.sample(real_pool_paths, 200)
    print(f"  Real basophil pool: {len(real_pool_paths)}장")
    real_pool_imgs = [Image.open(p).convert("RGB") for p in real_pool_paths]
    real_embs = emb_model.embed_images(real_pool_imgs, device)
    print(f"  real_embs: {real_embs.shape}")

    # ── 생성 루프 ─────────────────────────────────────────────────────
    print(f"\n[Step 3] 생성 루프 (2×{args.n_gen}장)...")
    all_gen_imgs: list[Image.Image] = []
    inp_data_list = []

    for idx, (img_path, domain) in enumerate(selected_inputs):
        print(f"\n  ── 입력 [{idx+1}] {domain}/{img_path.name} ──")
        img = Image.open(img_path).convert("RGB")

        sub_dir = args.output_dir / f"input_{idx+1:02d}_{domain.split('_')[1]}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        # 원본 저장 (정사각형 256 리사이즈)
        img_save = img.resize((256, 256))
        img_save.save(sub_dir / "input.png")

        # 분류 + 첫 생성
        route_result = router.route(
            img,
            conf_threshold=0.7,
            top_k=2,
            use_tta=True,
            n_gen_candidates=1,
            denoise=args.denoise,
            seed=0,
            generate=True,
        )
        pred_class   = route_result["class_name"]
        class_conf   = route_result["class_conf"]
        domain_conf  = route_result["domain_conf"]
        routing_mode = route_result["routing_mode"]
        prompt       = route_result.get("prompt", "")
        print(f"    pred_class={pred_class} conf={class_conf:.4f}, "
              f"domain_conf={domain_conf:.4f}, routing={routing_mode}")

        # 10장 생성 (seed 0 포함)
        gen_imgs_raw = [route_result["generated"]]
        for s in range(1, args.n_gen):
            g = router._generate_once(img, prompt, denoise=args.denoise, seed=s)
            gen_imgs_raw.append(g)

        # 저장 + 지표 계산
        # CNN 일괄 예측
        cnn_preds, cnn_confs_list, cnn_probs_list = cnn_model.predict_batch(gen_imgs_raw, device)

        # Embedding (NN cosine용)
        gen_embs_per_inp = emb_model.embed_images(gen_imgs_raw, device)
        nn_per_img, nn_mean_inp = nn_cosine(gen_embs_per_inp, real_embs)

        # FD (10장 vs real)
        fd_inp = frechet_distance(real_embs, gen_embs_per_inp)

        # 입력 이미지 색상/선명도
        input_color  = color_stats(img)
        input_sharp  = sharpness_laplacian(img)

        gen_data = []
        for s, (gen_img, cpred, cconf, cprobs, nn_val) in enumerate(
            zip(gen_imgs_raw, cnn_preds, cnn_confs_list, cnn_probs_list, nn_per_img)
        ):
            # 저장
            gen_path = sub_dir / f"gen_{s:02d}.png"
            gen_img.save(gen_path)

            ssim_v = ssim_pair(gen_img, img)
            psnr_v = compute_psnr(gen_img, img)
            sharp_v = sharpness_laplacian(gen_img)
            col_v   = color_stats(gen_img)

            print(f"    seed={s}: SSIM={ssim_v:.4f}, PSNR={psnr_v:.2f}dB, "
                  f"CNN={['bas','eos','lym','mon','neu'][cpred]}({cconf:.3f}), "
                  f"NN_cos={nn_val:.4f}")

            gen_data.append({
                "seed":       s,
                "path_rel":   f"input_{idx+1:02d}_{domain.split('_')[1]}/gen_{s:02d}.png",
                "ssim":       round(ssim_v, 4),
                "psnr":       round(psnr_v, 2),
                "sharpness":  round(sharp_v, 2),
                "color_stats": col_v,
                "cnn_pred":   cpred,
                "cnn_conf":   round(cconf, 4),
                "cnn_probs":  [round(x, 4) for x in cprobs],
                "nn_cos":     round(float(nn_val), 4),
            })

        all_gen_imgs.extend(gen_imgs_raw)

        inp_data_list.append({
            "idx":          idx,
            "domain":       domain,
            "domain_label": DOMAIN_LABELS.get(domain, domain),
            "input_path":   str(img_path),
            "input_rel":    f"input_{idx+1:02d}_{domain.split('_')[1]}/input.png",
            "pred_class":   pred_class,
            "class_conf":   class_conf,
            "domain_conf":  domain_conf,
            "routing_mode": routing_mode,
            "prompt":       prompt,
            "input_color":  input_color,
            "input_sharpness": input_sharp,
            "generated":    gen_data,
            "fd":           fd_inp,
        })

    # 전체 20장 FD
    all_gen_embs = emb_model.embed_images(all_gen_imgs, device)
    fd_total = frechet_distance(real_embs, all_gen_embs)
    inp_data_list[0]["fd_total"] = fd_total

    # ── 요약 JSON ─────────────────────────────────────────────────────
    print(f"\n[Step 4] 결과 저장...")
    summary = {
        "timestamp": ts,
        "seed": args.seed,
        "denoise": args.denoise,
        "n_gen_per_input": args.n_gen,
        "n_total": len(all_gen_imgs),
        "fd_total_20_vs_real200": fd_total,
        "inputs": inp_data_list,
        "overall": {
            "ssim_mean":   round(float(np.mean([g["ssim"] for inp in inp_data_list for g in inp["generated"]])), 4),
            "ssim_std":    round(float(np.std( [g["ssim"] for inp in inp_data_list for g in inp["generated"]])), 4),
            "psnr_mean":   round(float(np.mean([g["psnr"] for inp in inp_data_list for g in inp["generated"]])), 2),
            "cnn_acc":     round(sum(1 for inp in inp_data_list for g in inp["generated"] if g["cnn_pred"]==0) / 20, 4),
            "cnn_conf_mean": round(float(np.mean([g["cnn_conf"] for inp in inp_data_list for g in inp["generated"]])), 4),
            "nn_cosine_mean": round(float(np.mean([g["nn_cos"] for inp in inp_data_list for g in inp["generated"]])), 4),
            "sharpness_mean": round(float(np.mean([g["sharpness"] for inp in inp_data_list for g in inp["generated"]])), 2),
            "fd_total": fd_total,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # ── 마크다운 갤러리 ───────────────────────────────────────────────
    gallery_md = make_gallery_md(args.output_dir, inp_data_list, ts)
    (args.output_dir / "gallery_report.md").write_text(gallery_md, encoding="utf-8")

    # ── 최종 요약 ─────────────────────────────────────────────────────
    ov = summary["overall"]
    print(f"\n{'='*65}")
    print(f"  ✅ Script 19 완료!")
    print(f"{'='*65}")
    print(f"  총 생성: {len(all_gen_imgs)}장  (2입력 × {args.n_gen})")
    print(f"  SSIM mean : {ov['ssim_mean']:.4f} ± {ov['ssim_std']:.4f}")
    print(f"  PSNR mean : {ov['psnr_mean']:.2f} dB")
    print(f"  CNN acc   : {ov['cnn_acc']:.0%} ({int(ov['cnn_acc']*20)}/20)")
    print(f"  CNN conf  : {ov['cnn_conf_mean']:.4f}")
    print(f"  NN cosine : {ov['nn_cosine_mean']:.4f}")
    print(f"  FD (총20장vs real200장): {ov['fd_total']}")
    print(f"  결과: {args.output_dir}")
    print(f"  리포트: {args.output_dir / 'gallery_report.md'}")
    print(f"{'='*65}\n")

    router.cleanup()


if __name__ == "__main__":
    main()
