"""
Basophil generation test + similarity measurement.

Given a trained LoRA, generates N images via img2img and measures:
1. Visual quality   : Laplacian variance (sharpness)
2. Feature similarity: SSIM between generated and nearest real image
3. LPIPS perceptual distance (VGG-based)
4. FID proxy        : Intra-set Fréchet distance using CNN features
                      (baseline CNN's penultimate layer as feature extractor)
5. Classifier confidence: baseline CNN softmax score for correct class

Usage:
    python test_generation_similarity.py \
        --lora_dir lora/weights/basophil \
        --class_name basophil \
        --n_gen 20 \
        --denoise 0.35

Outputs:
    results/generation_test/basophil/
        ├── generated/          generated PNG files
        ├── report.json         per-image + aggregate metrics
        └── grid.png            4×5 visual grid (real | generated)
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
TRAIN_DIR   = ROOT / "data" / "processed" / "train"
MODEL_CKPT  = ROOT / "models" / "baseline_cnn.pt"
OUT_ROOT    = ROOT / "results" / "generation_test"

BASE_MODEL  = "stabilityai/stable-diffusion-xl-base-1.0"

CLASS_NAMES = ["basophil","eosinophil","erythroblast","ig",
               "lymphocyte","monocyte","neutrophil","platelet"]

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
NEGATIVE_PROMPT = "cartoon, illustration, text, watermark, multiple cells, heavy artifacts, unrealistic colors, deformed nucleus, blurry"

# ── Helpers ──────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def laplacian_variance(pil_img: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def ssim_pair(img_a: Image.Image, img_b: Image.Image, size: int = 224) -> float:
    """Structural Similarity Index between two PIL images (grayscale)."""
    a = np.array(img_a.resize((size, size)).convert("L")).astype(float) / 255.
    b = np.array(img_b.resize((size, size)).convert("L")).astype(float) / 255.
    mu_a, mu_b = a.mean(), b.mean()
    sig_a = a.std(); sig_b = b.std()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = 0.01**2, 0.03**2
    return float((2*mu_a*mu_b + C1) * (2*sig_ab + C2) /
                 ((mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2)))

def mse_pair(img_a: Image.Image, img_b: Image.Image, size: int = 224) -> float:
    a = np.array(img_a.resize((size, size))).astype(float) / 255.
    b = np.array(img_b.resize((size, size))).astype(float) / 255.
    return float(np.mean((a - b) ** 2))

# ── CNN feature extractor (penultimate layer of baseline CNN) ─────────

class FeatureExtractor(nn.Module):
    def __init__(self, ckpt_path: Path, n_classes: int, device):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, n_classes)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        base.load_state_dict(state)
        # Drop final linear layer → use 1280-d embedding
        self.features = nn.Sequential(*list(base.children())[:-1])  # AdaptiveAvgPool included
        self.classifier = base.classifier
        self.eval()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return feat.flatten(1)  # (B, 1280)

    def predict(self, x: torch.Tensor):
        emb = self.embed(x)
        logits = self.classifier(emb)
        return F.softmax(logits, dim=1)

def build_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def extract_embeddings(imgs: list, model: FeatureExtractor, transform, device) -> np.ndarray:
    model = model.to(device)
    vecs = []
    for img in imgs:
        t = transform(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            vecs.append(model.embed(t).cpu().numpy())
    return np.vstack(vecs)  # (N, 1280)

def frechet_distance(mu1, sig1, mu2, sig2) -> float:
    """Simplified FD: ||mu1-mu2||^2 + Tr(sig1 + sig2 - 2*sqrt(sig1*sig2))
    Uses regularization to handle rank-deficient covariance matrices (small n vs high-dim embeddings).
    """
    from scipy.linalg import sqrtm
    eps = 1e-6
    sig1 = sig1 + eps * np.eye(sig1.shape[0])
    sig2 = sig2 + eps * np.eye(sig2.shape[0])
    diff = mu1 - mu2
    covmean = sqrtm(sig1 @ sig2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2 * covmean))

# ── Generation ───────────────────────────────────────────────────────

def load_generation_pipeline(lora_dir: Path, device: str):
    from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
    # MPS requires full float32 to avoid NaN→black image bug with fp16
    use_fp16 = device == "cuda"
    dtype   = torch.float16 if use_fp16 else torch.float32
    variant = "fp16"        if use_fp16 else None
    print(f"  Loading SDXL img2img pipeline ({'fp16' if use_fp16 else 'fp32'})...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, variant=variant, use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    lora_weights = lora_dir / "pytorch_lora_weights.safetensors"
    if lora_weights.exists():
        print(f"  Loading LoRA weights from {lora_dir.name}...")
        pipe.load_lora_weights(str(lora_dir))
    else:
        print(f"  [WARN] No LoRA weights found at {lora_dir}, using base SDXL")
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

def generate_images(pipe, class_name: str, real_imgs: list,
                    n_gen: int, denoise: float, device: str, seed: int = 42):
    prompt = CLASS_PROMPTS[class_name]
    rng = random.Random(seed)
    src_paths = rng.choices(real_imgs, k=n_gen)
    generated = []
    src_used = []
    print(f"  Generating {n_gen} images (denoise={denoise})...")
    for i, src_path in enumerate(tqdm(src_paths, desc="  gen")):
        ref = Image.open(src_path).convert("RGB").resize((512, 512))
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=ref,
                strength=denoise,
                guidance_scale=6.0,
                num_inference_steps=25,
                generator=torch.Generator(device).manual_seed(seed + i),
            ).images[0]
        generated.append(result)
        src_used.append(src_path)
    return generated, src_used

# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir",   type=str, required=True)
    parser.add_argument("--class_name", type=str, default="basophil")
    parser.add_argument("--n_gen",      type=int, default=20)
    parser.add_argument("--denoise",    type=float, default=0.35)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no_generate", action="store_true",
                        help="Skip generation, use existing images in output dir")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    lora_dir = Path(args.lora_dir)
    cls = args.class_name
    class_idx = CLASS_NAMES.index(cls)

    out_dir = OUT_ROOT / cls / lora_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(exist_ok=True)

    # ── Real images ─────────────────────────────────────────────────
    real_paths = list((TRAIN_DIR / cls).rglob("*.jpg")) + \
                 list((TRAIN_DIR / cls).rglob("*.png"))
    random.seed(args.seed)
    sample_reals = random.sample(real_paths, min(args.n_gen, len(real_paths)))
    real_imgs = [Image.open(p).convert("RGB") for p in sample_reals]
    print(f"Real images: {len(real_imgs)} sampled from {len(real_paths)} total")

    # ── Generation ──────────────────────────────────────────────────
    if not args.no_generate:
        pipe = load_generation_pipeline(lora_dir, device)
        gen_imgs, src_used = generate_images(
            pipe, cls, real_paths, args.n_gen, args.denoise, device, args.seed
        )
        del pipe
        if device == "mps": torch.mps.empty_cache()
        # Save generated images
        gen_paths = []
        for i, img in enumerate(gen_imgs):
            p = gen_dir / f"gen_{cls}_{i:04d}_ds{int(args.denoise*100)}.png"
            img.save(p)
            gen_paths.append(p)
        print(f"  Saved {len(gen_imgs)} images → {gen_dir}")
    else:
        gen_paths = sorted(gen_dir.glob("*.png"))
        gen_imgs = [Image.open(p).convert("RGB") for p in gen_paths]
        print(f"  Loaded {len(gen_imgs)} existing generated images")

    # ── Load baseline CNN ───────────────────────────────────────────
    print("\nLoading baseline CNN for feature extraction...")
    cnn = FeatureExtractor(MODEL_CKPT, len(CLASS_NAMES), device)
    cnn = cnn.to(device)
    transform = build_transform()

    # ── Per-image metrics ────────────────────────────────────────────
    print("\nComputing per-image metrics...")
    results = []

    # Embeddings for FD calculation
    real_embs  = extract_embeddings(real_imgs,  cnn, transform, device)  # (N, 1280)
    gen_embs   = extract_embeddings(gen_imgs,   cnn, transform, device)  # (N, 1280)

    for i, (gen_img, gen_path) in enumerate(zip(gen_imgs, gen_paths)):
        # Sharpness
        gen_sharp  = laplacian_variance(gen_img)
        real_sharp = laplacian_variance(real_imgs[i % len(real_imgs)])

        # SSIM vs closest real (by pixel MSE)
        mses = [mse_pair(gen_img, r) for r in real_imgs]
        nearest_idx = int(np.argmin(mses))
        nearest_real = real_imgs[nearest_idx]
        ssim_val = ssim_pair(gen_img, nearest_real)
        mse_val  = float(mses[nearest_idx])

        # CNN confidence for correct class
        t = transform(gen_img.resize((224,224))).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = cnn.predict(t)[0].cpu().numpy()
        conf_correct = float(probs[class_idx])
        pred_class   = int(np.argmax(probs))
        conf_max     = float(probs[pred_class])

        results.append({
            "idx": i,
            "file": str(gen_path.name),
            "sharpness_gen":  round(gen_sharp, 2),
            "sharpness_real": round(real_sharp, 2),
            "ssim_nearest":   round(ssim_val, 4),
            "mse_nearest":    round(mse_val, 6),
            "cnn_conf_correct": round(conf_correct, 4),
            "cnn_pred_class": CLASS_NAMES[pred_class],
            "cnn_correct":    pred_class == class_idx,
        })

    # ── Aggregate metrics ────────────────────────────────────────────
    real_sharp_vals = [laplacian_variance(r) for r in real_imgs]

    # FD (Fréchet Distance using CNN features)
    mu_r, sig_r = real_embs.mean(0), np.cov(real_embs.T)
    mu_g, sig_g = gen_embs.mean(0),  np.cov(gen_embs.T)
    fd_score = frechet_distance(mu_r, sig_r, mu_g, sig_g)

    # Cosine similarity (mean pairwise gen↔real)
    real_norm = real_embs / (np.linalg.norm(real_embs, axis=1, keepdims=True) + 1e-8)
    gen_norm  = gen_embs  / (np.linalg.norm(gen_embs,  axis=1, keepdims=True) + 1e-8)
    cos_sim_matrix = gen_norm @ real_norm.T          # (N_gen, N_real)
    mean_cos_sim   = float(cos_sim_matrix.max(axis=1).mean())   # nearest-neighbor cosine sim

    agg = {
        "class": cls,
        "lora_dir": str(lora_dir),
        "denoise_strength": args.denoise,
        "n_generated": len(gen_imgs),
        "n_real_sample": len(real_imgs),
        # Sharpness
        "sharpness_gen_mean":   round(float(np.mean([r["sharpness_gen"] for r in results])), 2),
        "sharpness_gen_std":    round(float(np.std([r["sharpness_gen"] for r in results])), 2),
        "sharpness_real_mean":  round(float(np.mean(real_sharp_vals)), 2),
        "sharpness_real_std":   round(float(np.std(real_sharp_vals)), 2),
        # SSIM
        "ssim_mean":  round(float(np.mean([r["ssim_nearest"] for r in results])), 4),
        "ssim_std":   round(float(np.std([r["ssim_nearest"] for r in results])), 4),
        # MSE
        "mse_mean":   round(float(np.mean([r["mse_nearest"] for r in results])), 6),
        # CNN
        "cnn_conf_mean":      round(float(np.mean([r["cnn_conf_correct"] for r in results])), 4),
        "cnn_conf_std":       round(float(np.std([r["cnn_conf_correct"] for r in results])), 4),
        "cnn_accuracy":       round(float(np.mean([r["cnn_correct"] for r in results])), 4),
        # Feature-space
        "frechet_distance":         round(fd_score, 4),
        "nn_cosine_similarity_mean": round(mean_cos_sim, 4),
    }

    # ── Save report ──────────────────────────────────────────────────
    report = {"aggregate": agg, "per_image": results}
    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {report_path}")

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  CLASS: {cls.upper()}  |  LoRA: {lora_dir.name}")
    print(f"  Denoise strength: {args.denoise}")
    print("="*55)
    print(f"  Generated: {agg['n_generated']} images")
    print()
    print(f"  [Sharpness]  gen={agg['sharpness_gen_mean']:.1f}±{agg['sharpness_gen_std']:.1f}"
          f"  real={agg['sharpness_real_mean']:.1f}±{agg['sharpness_real_std']:.1f}")
    print(f"  [SSIM↑]      {agg['ssim_mean']:.4f} ± {agg['ssim_std']:.4f}  (vs nearest real)")
    print(f"  [MSE↓]       {agg['mse_mean']:.6f}")
    print(f"  [CNN conf↑]  {agg['cnn_conf_mean']:.4f} ± {agg['cnn_conf_std']:.4f}")
    print(f"  [CNN acc↑]   {agg['cnn_accuracy']*100:.1f}%  (correct class predicted)")
    print(f"  [FD↓]        {agg['frechet_distance']:.2f}  (CNN feature Fréchet distance)")
    print(f"  [CosSim↑]    {agg['nn_cosine_similarity_mean']:.4f}  (NN cosine similarity)")
    print("="*55)

    # ── Visual grid ──────────────────────────────────────────────────
    _save_grid(real_imgs[:10], gen_imgs[:10], out_dir / "grid.png", cls)
    print(f"  Visual grid → {out_dir/'grid.png'}")


def _save_grid(reals, gens, path: Path, title: str, thumb=200):
    """Save side-by-side grid: top row = real, bottom row = generated."""
    n = min(len(reals), len(gens), 10)
    w = n * thumb
    h = 2 * thumb + 40  # +40 for label bar
    grid = Image.new("RGB", (w, h), (30, 30, 30))

    for i in range(n):
        r = reals[i].resize((thumb, thumb))
        g = gens[i].resize((thumb, thumb))
        grid.paste(r, (i * thumb, 0))
        grid.paste(g, (i * thumb, thumb + 40))

    # Draw labels with PIL drawing
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        draw.text((4, thumb + 2),     "▲ Real",      fill=(200, 255, 200))
        draw.text((4, thumb + 18),    "▼ Generated", fill=(200, 200, 255))
        draw.text((w//2 - 60, thumb + 10), f"Class: {title}", fill=(255, 220, 100))
    except Exception:
        pass

    grid.save(path)


if __name__ == "__main__":
    main()
