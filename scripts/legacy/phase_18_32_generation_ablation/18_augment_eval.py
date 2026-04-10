"""
Script 18: Random Augmentation & Evaluation
============================================
데이터셋에서 랜덤 10장을 선택 → 라우터로 class+domain 분류 →
img2img로 10장씩 생성(총 100장) → 생성 품질 종합 평가.

평가 지표:
  - SSIM      : 생성 이미지 vs 원본 입력 (ssim_pair)
  - CNN acc   : multidomain_cnn.pt로 생성 이미지 분류 → 라우터 예측과 일치 여부
  - FD        : FrechetDistance (EfficientNet-B0 1280-dim, real vs gen)
  - NN cosine : 생성 embed vs real pool embed의 mean-max cosine similarity

Usage:
    python scripts/legacy/phase_18_32_generation_ablation/18_augment_eval.py \\
        --router_ckpt models/dual_head_router.pt \\
        --cnn_ckpt    models/multidomain_cnn.pt \\
        --n_inputs 10 --n_gen 10 --seed 42 \\
        --output_dir results/augment_eval/
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import models, transforms
from tqdm import tqdm

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "processed_multidomain"
IMG_EXTS  = {".jpg", ".jpeg", ".png"}

# ── script 15 동적 임포트 (숫자 시작 모듈명 대응) ─────────────────────
def _load_script15():
    spec = importlib.util.spec_from_file_location(
        "router_inference",
        ROOT / "scripts" / "15_router_inference.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 데이터 수집 ────────────────────────────────────────────────────────
def collect_all_paths(data_dir: Path) -> list[tuple[Path, str, str]]:
    """
    processed_multidomain/{domain}/{class}/*.{jpg,png} 전체 수집.
    반환: [(path, class_name, domain_name), ...]
    """
    records = []
    for domain_dir in sorted(data_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain_name = domain_dir.name
        for class_dir in sorted(domain_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for p in class_dir.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    records.append((p, class_name, domain_name))
    return records


def sample_random_inputs(
    all_paths: list[tuple[Path, str, str]],
    n: int,
    seed: int,
) -> list[tuple[Path, str, str]]:
    rng = random.Random(seed)
    return rng.sample(all_paths, min(n, len(all_paths)))


# ── CNN 평가용 모델 (multidomain_cnn.pt) ─────────────────────────────
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
            ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"],
        )
        self.class_idx = {c: i for i, c in enumerate(self.class_names)}
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def predict_batch(
        self, imgs: list[Image.Image], device: str
    ) -> tuple[list[int], list[float]]:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensors = torch.stack([transform(img.convert("RGB")) for img in imgs]).to(device)
        logits = self.forward(tensors)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1).cpu().tolist()
        confs  = probs.max(dim=1).values.cpu().tolist()
        return preds, confs


# ── 임베딩 추출 (FD, NN cosine용) ────────────────────────────────────
class EmbExtractor(nn.Module):
    """EfficientNet-B0 backbone → 1280-dim flatten embedding."""

    def __init__(self, ckpt_path: Path, device_str: str):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location=device_str, weights_only=False)
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, 5)
        base.load_state_dict(ckpt["model_state_dict"])
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)

    @torch.no_grad()
    def embed_images(
        self, imgs: list[Image.Image], device: str, batch_size: int = 32
    ) -> np.ndarray:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        all_embs = []
        for i in range(0, len(imgs), batch_size):
            batch = torch.stack(
                [transform(img.convert("RGB")) for img in imgs[i : i + batch_size]]
            ).to(device)
            embs = self.forward(batch).cpu().numpy()
            all_embs.append(embs)
        return np.concatenate(all_embs, axis=0)

    @torch.no_grad()
    def embed_paths(
        self, paths: list[Path], device: str, batch_size: int = 32
    ) -> np.ndarray:
        imgs = [Image.open(p).convert("RGB") for p in paths]
        return self.embed_images(imgs, device, batch_size)


# ── FrechetDistance ───────────────────────────────────────────────────
def frechet_distance(real_embs: np.ndarray, gen_embs: np.ndarray) -> float:
    """FD between two embedding sets (each row = 1 image)."""
    mu_r, mu_g = real_embs.mean(0), gen_embs.mean(0)
    # 극소 샘플 대비 정규화
    if real_embs.shape[0] < 2 or gen_embs.shape[0] < 2:
        return float("nan")
    cov_r = np.cov(real_embs, rowvar=False) + np.eye(real_embs.shape[1]) * 1e-6
    cov_g = np.cov(gen_embs,  rowvar=False) + np.eye(gen_embs.shape[1]) * 1e-6
    diff  = mu_r - mu_g
    covmean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = float(diff @ diff + np.trace(cov_r + cov_g - 2.0 * covmean))
    return round(fd, 4)


# ── NN cosine similarity ──────────────────────────────────────────────
def nn_cosine_similarity(gen_embs: np.ndarray, real_embs: np.ndarray) -> float:
    """mean over gen of max cosine similarity to any real image."""
    gen_n  = gen_embs  / (np.linalg.norm(gen_embs,  axis=1, keepdims=True) + 1e-8)
    real_n = real_embs / (np.linalg.norm(real_embs, axis=1, keepdims=True) + 1e-8)
    sims   = gen_n @ real_n.T  # (n_gen, n_real)
    max_sims = sims.max(axis=1)  # (n_gen,)
    return round(float(max_sims.mean()), 4)


# ── 마크다운 리포트 ───────────────────────────────────────────────────
def make_report(summary: dict) -> str:
    ov = summary["overall"]
    lines = [
        "# Script 18 — WBC Augmentation & Evaluation Report",
        f"\n**실행일시:** {summary.get('timestamp', '')}",
        f"**Seed:** {summary['seed']}  |  입력 {summary['n_inputs']}장 × {summary['n_gen_per_input']}장 생성 = **총 {summary['n_generated_total']}장**",
        "",
        "---",
        "",
        "## Overall 평가 결과",
        "",
        "| 지표 | 값 |",
        "|------|----|",
        f"| SSIM mean ± std | {ov['ssim_mean']:.4f} ± {ov['ssim_std']:.4f} |",
        f"| CNN accuracy (multidomain_cnn.pt) | {ov['cnn_accuracy']:.2%} ({int(ov['cnn_accuracy']*summary['n_generated_total'])}/{summary['n_generated_total']}) |",
        f"| CNN confidence mean | {ov['cnn_conf_mean']:.4f} |",
        f"| FrechetDistance (real vs gen) | {ov.get('frechet_distance', 'N/A')} |",
        f"| NN cosine similarity mean | {ov.get('nn_cosine_mean', 'N/A')} |",
        "",
        "### Routing mode 분포",
        "",
        "| Routing Mode | Count |",
        "|--------------|-------|",
    ]
    for mode, cnt in sorted(ov.get("routing_mode_counts", {}).items(), key=lambda x: -x[1]):
        lines.append(f"| {mode} | {cnt} |")

    lines += [
        "",
        "---",
        "",
        "## Per-Input 결과",
        "",
        "| # | 파일 | 실제 클래스 | 예측 클래스 | cls conf | 도메인 예측 | SSIM mean | CNN acc/10 | routing |",
        "|---|------|------------|------------|----------|------------|-----------|-----------|---------|",
    ]
    for inp in summary["per_input"]:
        fname = Path(inp["input_path"]).name
        lines.append(
            f"| {inp['idx']:02d} | `{fname}` | {inp['true_class']} | **{inp['pred_class']}** "
            f"| {inp['class_conf']:.3f} | {inp['pred_domain'].split('_')[1]} "
            f"| {inp['ssim_mean']:.4f} | {inp['cnn_acc_10']:.0%} ({int(inp['cnn_acc_10']*10)}/10) "
            f"| {inp['routing_mode']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 클래스별 집계",
        "",
        "| 클래스 | 입력 수 | gen SSIM mean | CNN acc |",
        "|--------|--------|--------------|---------|",
    ]
    class_stats = defaultdict(lambda: {"ssims": [], "cnn_correct": [], "n": 0})
    for inp in summary["per_input"]:
        cls = inp["pred_class"]
        class_stats[cls]["ssims"].extend(inp["ssim_list"])
        class_stats[cls]["n"] += 1
        # cnn_acc_10은 비율이므로 correct 수 역산
        class_stats[cls]["cnn_correct"].extend(
            [1] * round(inp["cnn_acc_10"] * 10) + [0] * (10 - round(inp["cnn_acc_10"] * 10))
        )
    for cls, st in sorted(class_stats.items()):
        ssim_m = np.mean(st["ssims"]) if st["ssims"] else float("nan")
        cnn_m  = np.mean(st["cnn_correct"]) if st["cnn_correct"] else float("nan")
        lines.append(f"| {cls} | {st['n']} | {ssim_m:.4f} | {cnn_m:.2%} |")

    lines += [
        "",
        "---",
        "",
        "> 생성된 이미지는 `results/augment_eval/` 에 저장됨.",
        "> 평가 모델: `models/multidomain_cnn.pt` (EfficientNet-B0, 5class, F1=0.9917)",
    ]
    return "\n".join(lines)


# ── argparse ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Script 18: Random 10장 → 라우터 분류 → 10장×생성(총 100장) → 평가"
    )
    p.add_argument("--router_ckpt", type=Path,
                   default=ROOT / "models" / "dual_head_router.pt")
    p.add_argument("--cnn_ckpt",    type=Path,
                   default=ROOT / "models" / "multidomain_cnn.pt")
    p.add_argument("--data_dir",    type=Path,  default=DATA_DIR)
    p.add_argument("--output_dir",  type=Path,
                   default=ROOT / "results" / "augment_eval")
    p.add_argument("--n_inputs",    type=int,   default=10,
                   help="랜덤 선택 입력 이미지 수 (기본 10)")
    p.add_argument("--n_gen",       type=int,   default=10,
                   help="입력 1장당 생성 이미지 수 (기본 10)")
    p.add_argument("--denoise",     type=float, default=0.35,
                   help="img2img strength (기본 0.35)")
    p.add_argument("--real_pool_max", type=int, default=200,
                   help="FD/NN cosine용 real pool 최대 장수 (클래스당)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--no_tta",      action="store_true", help="TTA 비활성화")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*65}")
    print(f"  Script 18 — WBC Augmentation & Evaluation  [{ts}]")
    print(f"{'='*65}")
    print(f"  n_inputs={args.n_inputs}  n_gen={args.n_gen}  seed={args.seed}")
    print(f"  router_ckpt : {args.router_ckpt}")
    print(f"  cnn_ckpt    : {args.cnn_ckpt}")
    print(f"  output_dir  : {args.output_dir}")
    print()

    # ── 1. script 15 동적 임포트 ──────────────────────────────────────
    print("[Step 1] script 15 (WBCRouter) 임포트...")
    mod15 = _load_script15()
    WBCRouter     = mod15.WBCRouter
    ssim_pair     = mod15.ssim_pair
    MULTI_CLASSES = mod15.MULTI_CLASSES
    DOMAINS       = mod15.DOMAINS
    print(f"  MULTI_CLASSES: {MULTI_CLASSES}")
    print(f"  DOMAINS      : {DOMAINS}")

    device = mod15.get_device()
    print(f"  device: {device}")

    # ── 2. 랜덤 10장 샘플링 ───────────────────────────────────────────
    print("\n[Step 2] 데이터 수집 및 랜덤 샘플링...")
    all_paths = collect_all_paths(args.data_dir)
    print(f"  전체 이미지: {len(all_paths):,}장")
    inputs = sample_random_inputs(all_paths, args.n_inputs, args.seed)
    print(f"  선택된 {len(inputs)}장:")
    for i, (p, cls, dom) in enumerate(inputs):
        print(f"    [{i:02d}] {dom}/{cls}/{p.name}")

    # ── 3. WBCRouter 초기화 ───────────────────────────────────────────
    print("\n[Step 3] WBCRouter 초기화 (파이프라인은 첫 생성 시 지연 로드)...")
    router = WBCRouter(
        router_ckpt=args.router_ckpt if args.router_ckpt.exists() else None,
        cnn_ckpt=args.cnn_ckpt,
        device=device,
    )

    # ── 4. 생성 루프 ─────────────────────────────────────────────────
    print(f"\n[Step 4] 생성 루프 ({len(inputs)}×{args.n_gen}={len(inputs)*args.n_gen}장)...")
    per_input_results = []
    all_generated_imgs: list[Image.Image] = []
    all_gen_class_idxs: list[int] = []

    for i, (img_path, true_class, true_domain) in enumerate(inputs):
        print(f"\n  ── Input [{i:02d}] {true_class}/{img_path.name} ──")
        img = Image.open(img_path).convert("RGB")

        # 서브 디렉토리 준비
        safe_pred = f"{i:02d}_{true_class}"  # 임시; pred_class로 나중에 rename 가능
        sub_dir = args.output_dir / safe_pred
        sub_dir.mkdir(parents=True, exist_ok=True)
        img.save(sub_dir / "input.png")

        # route() 1회 호출 (분류 + seed=0 생성)
        route_result = router.route(
            img,
            conf_threshold=0.7,
            top_k=2,
            use_tta=not args.no_tta,
            n_gen_candidates=1,   # 여기서는 1장만; 10장은 아래서 루프
            denoise=args.denoise,
            seed=0,
            generate=True,
        )

        pred_class   = route_result["class_name"]
        pred_class_i = route_result["class_idx"]
        pred_domain  = route_result["domain_name"]
        class_conf   = route_result["class_conf"]
        domain_conf  = route_result["domain_conf"]
        routing_mode = route_result["routing_mode"]
        prompt       = route_result.get("prompt", "")

        print(f"    pred_class={pred_class} (conf={class_conf:.3f}), "
              f"pred_domain={pred_domain} (conf={domain_conf:.3f}), "
              f"routing={routing_mode}")

        # 생성 이미지 수집: seed=0 결과 + seed=1~(n_gen-1) 추가 생성
        generated_imgs: list[Image.Image] = [route_result["generated"]]
        for s in range(1, args.n_gen):
            gen = router._generate_once(img, prompt, denoise=args.denoise, seed=s)
            generated_imgs.append(gen)

        # SSIM 계산 및 저장
        ssim_list = []
        for j, gen_img in enumerate(generated_imgs):
            ssim_val = ssim_pair(gen_img, img)
            ssim_list.append(round(ssim_val, 4))
            gen_img.save(sub_dir / f"gen_{j:02d}.png")

        ssim_mean = round(float(np.mean(ssim_list)), 4)
        print(f"    SSIM: mean={ssim_mean:.4f}, list={ssim_list}")

        all_generated_imgs.extend(generated_imgs)
        all_gen_class_idxs.extend([pred_class_i] * args.n_gen)

        # per-input meta.json
        meta = {
            "idx":         i,
            "input_path":  str(img_path),
            "true_class":  true_class,
            "true_domain": true_domain,
            "pred_class":  pred_class,
            "pred_class_idx": pred_class_i,
            "pred_domain": pred_domain,
            "class_conf":  class_conf,
            "domain_conf": domain_conf,
            "routing_mode": routing_mode,
            "prompt":      prompt,
            "ssim_list":   ssim_list,
            "ssim_mean":   ssim_mean,
        }
        (sub_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        per_input_results.append(meta)

    # ── 5. CNN 평가 ──────────────────────────────────────────────────
    print(f"\n[Step 5] CNN 평가 (multidomain_cnn.pt, {len(all_generated_imgs)}장)...")
    cnn_model = MultidomainCNN(args.cnn_ckpt, device).to(device)
    cnn_preds, cnn_confs = cnn_model.predict_batch(all_generated_imgs, device)

    # per-input CNN acc 집계
    for i, inp_meta in enumerate(per_input_results):
        start = i * args.n_gen
        end   = start + args.n_gen
        preds_i = cnn_preds[start:end]
        confs_i = cnn_confs[start:end]
        expected_idx = inp_meta["pred_class_idx"]
        correct_i = [1 if p == expected_idx else 0 for p in preds_i]
        inp_meta["cnn_acc_10"]    = round(sum(correct_i) / len(correct_i), 4)
        inp_meta["cnn_conf_mean"] = round(float(np.mean(confs_i)), 4)
        print(f"  [{i:02d}] {inp_meta['pred_class']}: "
              f"CNN acc={inp_meta['cnn_acc_10']:.2%}, conf={inp_meta['cnn_conf_mean']:.4f}")

    # overall CNN
    all_correct = [1 if p == e else 0
                   for p, e in zip(cnn_preds, all_gen_class_idxs)]
    overall_cnn_acc  = round(float(np.mean(all_correct)), 4)
    overall_cnn_conf = round(float(np.mean(cnn_confs)), 4)
    print(f"\n  Overall CNN accuracy: {overall_cnn_acc:.2%} ({sum(all_correct)}/{len(all_correct)})")

    # ── 6. Embedding 추출 (FD + NN cosine) ──────────────────────────
    print(f"\n[Step 6] Embedding 추출 (EfficientNet-B0, 1280-dim)...")
    emb_extractor = EmbExtractor(args.cnn_ckpt, device).to(device)

    # 생성 이미지 전체 embed
    gen_embs = emb_extractor.embed_images(all_generated_imgs, device)
    print(f"  gen_embs shape: {gen_embs.shape}")

    # real pool: 모든 클래스 통합 (= 예측 클래스별 real 이미지)
    # 각 클래스별로 real pool 수집 후 concat
    real_pool_all: list[Path] = []
    MULTI_CLASSES_list = MULTI_CLASSES  # ['basophil', ...]
    rng_pool = random.Random(args.seed + 1)
    for cls in MULTI_CLASSES_list:
        cls_paths: list[Path] = []
        for domain_dir in sorted(args.data_dir.iterdir()):
            cls_dir = domain_dir / cls
            if cls_dir.exists():
                cls_paths.extend(p for p in cls_dir.iterdir()
                                 if p.suffix.lower() in IMG_EXTS)
        if len(cls_paths) > args.real_pool_max:
            cls_paths = rng_pool.sample(cls_paths, args.real_pool_max)
        real_pool_all.extend(cls_paths)

    print(f"  real pool: {len(real_pool_all)}장 (최대 {args.real_pool_max}×{len(MULTI_CLASSES_list)})")
    real_embs = emb_extractor.embed_paths(real_pool_all, device)
    print(f"  real_embs shape: {real_embs.shape}")

    # FD (전체 통합)
    fd_val = frechet_distance(real_embs, gen_embs)
    print(f"  FrechetDistance: {fd_val}")

    # NN cosine (전체 통합)
    nn_cos = nn_cosine_similarity(gen_embs, real_embs)
    print(f"  NN cosine similarity: {nn_cos}")

    # ── 7. Summary 집계 ──────────────────────────────────────────────
    print("\n[Step 7] Summary 집계...")
    all_ssims = [s for inp in per_input_results for s in inp["ssim_list"]]
    routing_counts = Counter(inp["routing_mode"] for inp in per_input_results)

    summary = {
        "timestamp":        ts,
        "seed":             args.seed,
        "n_inputs":         len(inputs),
        "n_gen_per_input":  args.n_gen,
        "n_generated_total": len(all_generated_imgs),
        "router_ckpt":      str(args.router_ckpt),
        "cnn_ckpt":         str(args.cnn_ckpt),
        "per_input":        per_input_results,
        "overall": {
            "ssim_mean":          round(float(np.mean(all_ssims)), 4),
            "ssim_std":           round(float(np.std(all_ssims)), 4),
            "ssim_min":           round(float(np.min(all_ssims)), 4),
            "ssim_max":           round(float(np.max(all_ssims)), 4),
            "cnn_accuracy":       overall_cnn_acc,
            "cnn_conf_mean":      overall_cnn_conf,
            "frechet_distance":   fd_val,
            "nn_cosine_mean":     nn_cos,
            "routing_mode_counts": dict(routing_counts),
        },
    }

    # ── 8. 저장 ──────────────────────────────────────────────────────
    print("\n[Step 8] 결과 저장...")
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  → {summary_path}")

    report_md = make_report(summary)
    report_path = args.output_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  → {report_path}")

    # ── 9. 최종 출력 ─────────────────────────────────────────────────
    ov = summary["overall"]
    print(f"\n{'='*65}")
    print(f"  ✅ Script 18 완료!")
    print(f"{'='*65}")
    print(f"  총 생성: {summary['n_generated_total']}장  (입력 {summary['n_inputs']}×{summary['n_gen_per_input']})")
    print(f"  SSIM mean   : {ov['ssim_mean']:.4f} ± {ov['ssim_std']:.4f}")
    print(f"  CNN accuracy: {ov['cnn_accuracy']:.2%}")
    print(f"  CNN conf    : {ov['cnn_conf_mean']:.4f}")
    print(f"  FD          : {ov['frechet_distance']}")
    print(f"  NN cosine   : {ov['nn_cosine_mean']}")
    print(f"  결과 디렉토리: {args.output_dir}")
    print(f"{'='*65}\n")

    # 메모리 해제
    router.cleanup()
    del cnn_model, emb_extractor
    gc.collect()


if __name__ == "__main__":
    main()
