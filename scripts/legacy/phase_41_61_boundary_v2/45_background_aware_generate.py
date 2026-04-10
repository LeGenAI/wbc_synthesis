"""
Script 45: Background-aware contextual generation V2.

Outputs:
  data/generated_boundary_v2/{class}/...
  results/boundary_v2_generation/{class}/report.json
  results/boundary_v2_generation/{class}/summary.md
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from boundary_aware_utils import (
    ROOT,
    CLASSES,
    DOMAINS,
    DOMAIN_SHORT,
    build_background_prompt,
    build_contextual_prompt,
    boundary_score,
    cnn_prob_vector,
    entropy_margin_target,
    extract_cell_mask,
    get_device,
    load_cnn,
    masked_similarity,
)


DATA_DIR = ROOT / "data" / "processed_contextual_multidomain"
MASK_DIR = ROOT / "data" / "processed_contextual_masks"
LORA_DIR = ROOT / "lora" / "weights"
GEN_DIR = ROOT / "data" / "generated_boundary_v2"
OUT_DIR = ROOT / "results" / "boundary_v2_generation"
GEN_RUNS_DIR = ROOT / "data" / "generated_boundary_v2_runs"
OUT_RUNS_DIR = ROOT / "results" / "boundary_v2_generation_runs"

DEFAULT_BG_STRENGTHS = [0.65]
DEFAULT_REFINE_STRENGTHS = [0.20]
NEG_BG = "multiple cells, text, watermark, cell distortion, unrealistic nucleus, blurry foreground"
NEG_FULL = "multiple cells, text, watermark, unrealistic colors, deformed nucleus"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def get_target_domains(ref_domain: str, mode: str, allowed_targets: list[str] | None = None) -> list[str]:
    if mode == "same_domain":
        domains = [ref_domain]
    elif mode == "cross_only":
        domains = [d for d in DOMAINS if d != ref_domain]
    elif mode == "all_pairs":
        domains = list(DOMAINS)
    else:
        raise ValueError(mode)
    if allowed_targets is not None:
        domains = [d for d in domains if d in allowed_targets]
    return domains


def load_pipelines(lora_dir: Path, device: torch.device):
    from diffusers import (
        DPMSolverMultistepScheduler,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
    )

    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    dtype = torch.float32
    bg_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(base_model, torch_dtype=dtype, use_safetensors=True)
    bg_pipe.scheduler = DPMSolverMultistepScheduler.from_config(bg_pipe.scheduler.config, use_karras_sigmas=True)
    bg_pipe.load_lora_weights(str(lora_dir))
    bg_pipe = bg_pipe.to(device)
    bg_pipe.enable_attention_slicing()

    refine_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(base_model, torch_dtype=dtype, use_safetensors=True)
    refine_pipe.scheduler = DPMSolverMultistepScheduler.from_config(refine_pipe.scheduler.config, use_karras_sigmas=True)
    refine_pipe.load_lora_weights(str(lora_dir))
    refine_pipe = refine_pipe.to(device)
    refine_pipe.enable_attention_slicing()
    return bg_pipe, refine_pipe


def sample_inputs(
    class_name: str,
    n_per_domain: int,
    seed: int,
    allowed_ref_domains: list[str] | None = None,
) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    sampled = {}
    ref_domains = allowed_ref_domains or list(DOMAINS)
    for domain in ref_domains:
        cls_dir = DATA_DIR / domain / class_name
        paths = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS) if cls_dir.exists() else []
        if len(paths) > n_per_domain:
            paths = rng.sample(paths, n_per_domain)
        sampled[domain] = paths
    return sampled


def parse_domain_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    invalid = [v for v in values if v not in DOMAINS]
    if invalid:
        raise ValueError(f"invalid domains: {invalid}")
    return list(dict.fromkeys(values))


def find_mask_path(img_path: Path) -> Path:
    rel = img_path.relative_to(DATA_DIR).with_suffix(".png")
    return MASK_DIR / rel


def resize_mask(mask: np.ndarray, size: int = 512, invert: bool = False) -> Image.Image:
    arr = mask.copy()
    if invert:
        arr = np.where(arr > 0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.NEAREST)


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}
    cell_mean = float(np.mean([r["cell_ssim"] for r in records]))
    bg_mean = float(np.mean([r["background_ssim"] for r in records]))
    return {
        "n": len(records),
        "cnn_accuracy": round(float(np.mean([1.0 if r["cnn_correct"] else 0.0 for r in records])), 4),
        "cell_ssim_mean": round(cell_mean, 4),
        "background_ssim_mean": round(bg_mean, 4),
        "region_gap": round(cell_mean - bg_mean, 4),
        "entropy_mean": round(float(np.mean([r["cnn_entropy"] for r in records])), 4),
        "margin_mean": round(float(np.mean([r["target_margin"] for r in records])), 4),
        "near_boundary_rate": round(float(np.mean([1.0 if r["near_boundary"] else 0.0 for r in records])), 4),
        "variation_score_mean": round(float(np.mean([r["variation_score"] for r in records])), 4),
    }


def save_summary(class_name: str, report: dict, out_dir: Path) -> None:
    agg = report["aggregate"]
    lines = [
        f"# Boundary V2 Generation Summary: {class_name}",
        "",
        f"- timestamp: `{report['timestamp']}`",
        f"- total generated: `{report['n_generated']}`",
        f"- bg strengths: `{report['config']['background_strengths']}`",
        f"- refine strengths: `{report['config']['refine_strengths']}`",
        f"- cross_domain_mode: `{report['config']['cross_domain_mode']}`",
        "",
        "## Aggregate",
        "",
        f"- CNN accuracy: `{agg['cnn_accuracy']}`",
        f"- Mean cell SSIM: `{agg['cell_ssim_mean']}`",
        f"- Mean background SSIM: `{agg['background_ssim_mean']}`",
        f"- Region gap: `{agg['region_gap']}`",
        f"- Mean entropy: `{agg['entropy_mean']}`",
        f"- Mean margin: `{agg['margin_mean']}`",
        f"- Near-boundary rate: `{agg['near_boundary_rate']}`",
        f"- Mean variation score: `{agg['variation_score_mean']}`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_float_list(values: list[str] | None, default: list[float]) -> list[float]:
    if not values:
        return list(default)
    return [round(float(v), 2) for v in values]


def resolve_output_roots(run_tag: str | None) -> tuple[Path, Path]:
    if not run_tag:
        return GEN_DIR, OUT_DIR
    return GEN_RUNS_DIR / run_tag, OUT_RUNS_DIR / run_tag


def main():
    parser = argparse.ArgumentParser(description="Background-aware contextual generation V2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--class_name", choices=CLASSES)
    group.add_argument("--all_classes", action="store_true")
    parser.add_argument("--n_per_domain", type=int, default=2)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross_domain_mode", choices=["same_domain", "cross_only", "all_pairs"], default="cross_only")
    parser.add_argument("--ref_domains", nargs="+", choices=DOMAINS)
    parser.add_argument("--target_domains", nargs="+", choices=DOMAINS)
    parser.add_argument("--background_strengths", nargs="+")
    parser.add_argument("--refine_strengths", nargs="+")
    parser.add_argument("--disable_refine", action="store_true")
    parser.add_argument("--run_tag", help="Optional tag to write outputs under separate run-specific roots.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    bg_strengths = parse_float_list(args.background_strengths, DEFAULT_BG_STRENGTHS)
    refine_strengths = [0.0] if args.disable_refine else parse_float_list(args.refine_strengths, DEFAULT_REFINE_STRENGTHS)
    ref_domains = parse_domain_list(args.ref_domains)
    target_domains = parse_domain_list(args.target_domains)
    gen_root, out_root = resolve_output_roots(args.run_tag)

    device = get_device()
    cnn = load_cnn(device)

    class_names = CLASSES if args.all_classes else [args.class_name]
    for class_name in class_names:
        print(f"\n{'=' * 68}")
        print(f"Boundary V2 generation: {class_name}")
        print(f"device={device} bg_strengths={bg_strengths} refine_strengths={refine_strengths}")
        print(f"{'=' * 68}")

        sampled = sample_inputs(class_name, args.n_per_domain, args.seed, allowed_ref_domains=ref_domains)
        total_refs = sum(len(v) for v in sampled.values())
        if total_refs == 0:
            print(f"[skip] no contextual images found for {class_name}")
            continue
        active_ref_domains = [d for d, paths in sampled.items() if paths]
        target_count_lookup = {
            d: len(get_target_domains(d, args.cross_domain_mode, target_domains))
            for d in active_ref_domains
        }
        n_total = (
            sum(target_count_lookup.values())
            * args.n_per_domain
            * len(bg_strengths)
            * len(refine_strengths)
            * args.n_seeds
        )
        if args.dry_run:
            print(f"[dry_run] planned images: {n_total}")
            continue

        bg_pipe, refine_pipe = load_pipelines(LORA_DIR / f"contextual_multidomain_{class_name}", device)
        out_class_dir = out_root / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        records = []
        target_idx = CLASSES.index(class_name)
        t0 = time.time()

        for ref_domain, ref_paths in sampled.items():
            for inp_idx, ref_path in enumerate(ref_paths):
                ref_img = Image.open(ref_path).convert("RGB")
                mask_path = find_mask_path(ref_path)
                if mask_path.exists():
                    cell_mask = np.array(Image.open(mask_path).convert("L"))
                else:
                    cell_mask = extract_cell_mask(ref_img)
                bg_mask_img = resize_mask(cell_mask, size=512, invert=True)
                cell_mask_arr = np.array(resize_mask(cell_mask, size=512, invert=False))
                ref_512 = ref_img.resize((512, 512), Image.LANCZOS)

                for target_domain in get_target_domains(ref_domain, args.cross_domain_mode, target_domains):
                    bg_prompt = build_background_prompt(class_name, target_domain)
                    full_prompt = build_contextual_prompt(class_name, target_domain)
                    for bg_strength in bg_strengths:
                        for refine_strength in refine_strengths:
                            for s in range(args.n_seeds):
                                seed = args.seed + inp_idx * 1000 + int(bg_strength * 100) * 10 + s
                                out_img_dir = (
                                    gen_root / class_name / f"ref_{ref_domain}" / f"to_{target_domain}"
                                    / f"bg{int(bg_strength * 100):03d}" / f"rf{int(refine_strength * 100):03d}"
                                )
                                out_img_dir.mkdir(parents=True, exist_ok=True)
                                out_img = out_img_dir / f"{inp_idx:04d}_s{s}.png"

                                if out_img.exists() and not args.force:
                                    gen_img = Image.open(out_img).convert("RGB")
                                else:
                                    gen = torch.Generator(device).manual_seed(seed)
                                    stage1 = bg_pipe(
                                        prompt=bg_prompt,
                                        negative_prompt=NEG_BG,
                                        image=ref_512,
                                        mask_image=bg_mask_img,
                                        strength=bg_strength,
                                        guidance_scale=6.0,
                                        num_inference_steps=25,
                                        generator=gen,
                                    ).images[0]
                                    if refine_strength > 0:
                                        gen = torch.Generator(device).manual_seed(seed + 17)
                                        gen_img = refine_pipe(
                                            prompt=full_prompt,
                                            negative_prompt=NEG_FULL,
                                            image=stage1,
                                            strength=refine_strength,
                                            guidance_scale=6.0,
                                            num_inference_steps=25,
                                            generator=gen,
                                        ).images[0]
                                    else:
                                        gen_img = stage1
                                    gen_img.save(out_img)

                                probs = cnn_prob_vector(cnn, gen_img, device)
                                entropy, margin, target_prob, pred_idx = entropy_margin_target(probs, target_idx)
                                pred_name = CLASSES[pred_idx]
                                correct = pred_idx == target_idx
                                cell_ssim = masked_similarity(ref_512, gen_img, cell_mask_arr)
                                background_ssim = masked_similarity(ref_512, gen_img, np.where(cell_mask_arr > 0, 0, 255).astype(np.uint8))
                                near_boundary = correct and (0.02 <= margin <= 0.20)
                                score = boundary_score(cell_ssim, background_ssim, entropy, margin)
                                records.append({
                                    "file": str(out_img.relative_to(ROOT)),
                                    "input_file": str(ref_path.relative_to(ROOT)),
                                    "mask_file": str(mask_path.relative_to(ROOT)) if mask_path.exists() else None,
                                    "class_name": class_name,
                                    "domain": target_domain,
                                    "domain_short": DOMAIN_SHORT[target_domain],
                                    "ref_domain": ref_domain,
                                    "ref_domain_short": DOMAIN_SHORT[ref_domain],
                                    "prompt_domain": target_domain,
                                    "prompt_domain_short": DOMAIN_SHORT[target_domain],
                                    "is_cross_domain": ref_domain != target_domain,
                                    "inp_idx": inp_idx,
                                    "seed": seed,
                                    "background_strength": bg_strength,
                                    "refine_strength": refine_strength,
                                    "cnn_pred": pred_name,
                                    "cnn_correct": correct,
                                    "cnn_probs": [round(float(x), 4) for x in probs.tolist()],
                                    "cnn_entropy": round(entropy, 4),
                                    "target_margin": round(margin, 4),
                                    "target_prob": round(target_prob, 4),
                                    "cell_ssim": round(cell_ssim, 4),
                                    "background_ssim": round(background_ssim, 4),
                                    "near_boundary": near_boundary,
                                    "variation_score": round(score, 4),
                                })

        report = {
            "class": class_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_generated": len(records),
            "elapsed_min": round((time.time() - t0) / 60, 1),
            "config": {
                "background_strengths": bg_strengths,
                "refine_strengths": refine_strengths,
                "cross_domain_mode": args.cross_domain_mode,
                "ref_domains": ref_domains or list(DOMAINS),
                "target_domains": target_domains or list(DOMAINS),
                "n_per_domain": args.n_per_domain,
                "n_seeds": args.n_seeds,
                "run_tag": args.run_tag,
            },
            "aggregate": aggregate(records),
            "per_image": records,
        }
        (out_class_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        save_summary(class_name, report, out_class_dir)
        print(f"[done] {class_name} -> {out_class_dir}")


if __name__ == "__main__":
    main()
