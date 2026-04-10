"""
Step 3: Image generation pipeline (img2img, SDXL base + class LoRA).

For each class:
  - Loads the class-specific LoRA weights
  - Randomly samples `n_generate` real images from data/processed/train/{class}/
  - Runs img2img at denoise_strength ∈ [0.25, 0.35, 0.45] (configurable)
  - Saves generated images to data/generated/{class}/ds{denoise_str}/

Usage:
    # Generate with default settings (all classes, 3 denoise strengths)
    python 03_generate.py --all

    # Single class, single denoise
    python 03_generate.py --class_name neutrophil --denoise 0.35

    # Control generation multiplier
    python 03_generate.py --all --multiplier 2
    #   → generates 2× the train-set size per class
"""
import os
import argparse
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler


def _get_device() -> str:
    """Return the best available torch device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _get_device()

# ── Paths ──────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
TRAIN_DIR  = ROOT / "data" / "processed" / "train"
LORA_DIR   = ROOT / "lora" / "weights"
GEN_DIR    = ROOT / "data" / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# ── Prompt templates (same as LoRA training) ───────────────────────
CLASS_PROMPTS = {
    "basophil":    "microscopy image of a single basophil white blood cell, peripheral blood smear, clinical lab imaging, clear bilobed nucleus with dark granules, sharp focus, realistic",
    "eosinophil":  "microscopy image of a single eosinophil white blood cell, peripheral blood smear, clinical lab imaging, bilobed nucleus with orange-red granules, sharp focus, realistic",
    "erythroblast":"microscopy image of a single erythroblast cell, peripheral blood smear, clinical lab imaging, large round nucleus, basophilic cytoplasm, sharp focus, realistic",
    "ig":          "microscopy image of a single immature granulocyte white blood cell, peripheral blood smear, clinical lab imaging, band or metamyelocyte nucleus, sharp focus, realistic",
    "lymphocyte":  "microscopy image of a single lymphocyte white blood cell, peripheral blood smear, clinical lab imaging, large round nucleus with scant cytoplasm, sharp focus, realistic",
    "monocyte":    "microscopy image of a single monocyte white blood cell, peripheral blood smear, clinical lab imaging, kidney-shaped nucleus with grey cytoplasm, sharp focus, realistic",
    "neutrophil":  "microscopy image of a single neutrophil white blood cell, peripheral blood smear, clinical lab imaging, multilobed nucleus with pale granules, sharp focus, realistic",
    "platelet":    "microscopy image of a single platelet thrombocyte, peripheral blood smear, clinical lab imaging, small anucleate cell, sharp focus, realistic",
}

NEGATIVE_PROMPT = (
    "cartoon, illustration, text, watermark, multiple cells, "
    "heavy artifacts, unrealistic colors, deformed nucleus, blurry"
)

# ── Generation defaults ────────────────────────────────────────────
DENOISE_STRENGTHS = [0.25, 0.35, 0.45]
GUIDANCE_SCALE    = 6.0
NUM_STEPS         = 25
SEED              = 42


def load_pipeline(lora_path: Path | None = None) -> StableDiffusionXLImg2ImgPipeline:
    # Use float16 on GPU/MPS (unified memory on Apple Silicon); float32 on CPU only
    dtype   = torch.float16 if DEVICE != "cpu" else torch.float32
    variant = "fp16"        if DEVICE != "cpu" else None

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        variant=variant,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    if lora_path and lora_path.exists():
        pipe.load_lora_weights(str(lora_path))
        print(f"  Loaded LoRA from {lora_path}")
    else:
        print(f"  [WARN] LoRA not found at {lora_path}, using base model")

    pipe = pipe.to(DEVICE)

    if DEVICE == "cuda":
        # CUDA: offload model components to CPU between steps to save VRAM
        pipe.enable_model_cpu_offload()
        # xformers memory-efficient attention (CUDA-only)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        # MPS / CPU: Apple Silicon uses unified memory — no offload needed.
        # enable_attention_slicing reduces peak memory on MPS.
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    print(f"  Pipeline loaded on {DEVICE}")
    return pipe


def get_real_images(class_name: str) -> list[Path]:
    d = TRAIN_DIR / class_name
    if not d.exists():
        return []
    return [p for p in d.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def generate_for_class(
    class_name: str,
    pipe: StableDiffusionXLImg2ImgPipeline,
    n_generate: int,
    denoise_strengths: list[float],
    seed: int = SEED,
):
    real_imgs = get_real_images(class_name)
    if not real_imgs:
        print(f"  [SKIP] no train images for {class_name}")
        return

    prompt  = CLASS_PROMPTS.get(class_name, f"microscopy image of a single {class_name} white blood cell, peripheral blood smear, realistic")
    rng     = random.Random(seed)
    gen     = torch.Generator(DEVICE).manual_seed(seed)

    for ds in denoise_strengths:
        ds_tag = f"ds{int(ds*100):03d}"
        out_dir = GEN_DIR / class_name / ds_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # How many already generated?
        existing = len(list(out_dir.glob("*.png")))
        remaining = max(0, n_generate - existing)
        if remaining == 0:
            print(f"  [{class_name}|{ds_tag}] already {existing} images, skipping")
            continue

        print(f"  [{class_name}|{ds_tag}] generating {remaining} images "
              f"(denoise={ds}, guidance={GUIDANCE_SCALE})")

        src_paths = rng.choices(real_imgs, k=remaining)

        for i, src_path in enumerate(tqdm(src_paths, desc=f"  {class_name}|{ds_tag}")):
            ref = Image.open(src_path).convert("RGB").resize((512, 512))
            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=ref,
                strength=ds,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_STEPS,
                generator=torch.Generator(DEVICE).manual_seed(seed + i),
            ).images[0]
            fname = f"gen_{class_name}_{ds_tag}_{existing+i:06d}.png"
            result.save(out_dir / fname)

    print(f"  [{class_name}] generation done → {GEN_DIR / class_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name",  type=str, default=None)
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--denoise",     type=float, nargs="+",
                        default=DENOISE_STRENGTHS,
                        help="Denoise strength(s), e.g. --denoise 0.25 0.35 0.45")
    parser.add_argument("--multiplier",  type=int, default=1,
                        help="Generate multiplier × train-set size per class")
    parser.add_argument("--n_generate",  type=int, default=None,
                        help="Fixed number to generate (overrides --multiplier)")
    parser.add_argument("--seed",        type=int, default=SEED)
    args = parser.parse_args()

    if args.all:
        classes = [d.name for d in sorted(TRAIN_DIR.iterdir()) if d.is_dir()]
    elif args.class_name:
        classes = [args.class_name]
    else:
        parser.print_help()
        return

    print(f"Classes to generate: {classes}")
    print(f"Denoise strengths: {args.denoise}")

    for cls in classes:
        print(f"\n{'='*60}\nClass: {cls}")
        lora_path = LORA_DIR / cls
        pipe = load_pipeline(lora_path)

        real_count = len(get_real_images(cls))
        n_gen = args.n_generate if args.n_generate else real_count * args.multiplier
        print(f"  Real train images: {real_count} → generating: {n_gen} per denoise level")

        generate_for_class(cls, pipe, n_gen, args.denoise, args.seed)

        # Free GPU memory before loading next class's LoRA
        del pipe
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()

    print("\nAll generation done.")


if __name__ == "__main__":
    main()
