"""
T2I LoRA 생성 이미지 평가 스크립트
====================================
train_text_to_image_lora_sdxl.py로 학습한 T2I LoRA를 평가.

기존 test_generation_similarity.py (img2img)와 달리:
  - StableDiffusionXLPipeline (text-to-image) 사용
  - 도메인별 조건부 프롬프트로 생성
  - 동일 지표: FD, CNN Accuracy, CosSim으로 비교

Usage:
    python scripts/legacy/phase_08_17_domain_gap_multidomain/12_t2i_lora_eval.py --class_name basophil
    python scripts/legacy/phase_08_17_domain_gap_multidomain/12_t2i_lora_eval.py --class_name basophil --n_gen 20
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
TRAIN_DIR  = ROOT / "data" / "processed" / "train"
MODEL_CKPT = ROOT / "models" / "baseline_cnn.pt"
OUT_ROOT   = ROOT / "results" / "generation_test"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

CLASS_NAMES = ["basophil","eosinophil","erythroblast","ig",
               "lymphocyte","monocyte","neutrophil","platelet"]

# T2I 도메인별 프롬프트 (학습 시와 동일)
DOMAIN_PROMPTS = {
    "domain_a_pbc":    "May-Grünwald Giemsa stain, CellaVision automated analyzer, Barcelona Spain",
    "domain_b_raabin": "Giemsa stain, smartphone microscope camera, Iran hospital",
    "domain_c_mll23":  "Pappenheim stain, Metafer scanner, Germany clinical lab",
    "domain_e_amc":    "Romanowsky stain, miLab automated analyzer, South Korea AMC",
}
CLASS_MORPHOLOGY = {
    "basophil":    "bilobed nucleus with dark purple-black granules covering nucleus",
    "eosinophil":  "bilobed nucleus with large orange-red eosinophilic granules",
    "lymphocyte":  "large round nucleus with scant pale-blue cytoplasm",
    "monocyte":    "kidney-shaped or horseshoe nucleus with grey-blue cytoplasm",
    "neutrophil":  "multilobed nucleus with pale pink cytoplasmic granules",
}
NEGATIVE_PROMPT = "cartoon, illustration, text, watermark, multiple cells, heavy artifacts, unrealistic colors, deformed nucleus, blurry"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_t2i_prompt(class_name: str, domain: str) -> str:
    morph = CLASS_MORPHOLOGY.get(class_name, class_name)
    domain_desc = DOMAIN_PROMPTS.get(domain, domain)
    return (
        f"microscopy image of {class_name} white blood cell, "
        f"{morph}, "
        f"{domain_desc}, "
        f"peripheral blood smear, clinical lab imaging"
    )


# ── FeatureExtractor (test_generation_similarity.py에서 재사용) ────────
import torch.nn as nn
from torchvision import models, transforms


class FeatureExtractor(nn.Module):
    def __init__(self, ckpt_path: Path, n_classes: int, device):
        super().__init__()
        self.device = device
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(1280, n_classes)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.model.to(device)
        # features = avg pooling output (before classifier)
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        self.features.eval()
        self.classifier = self.model.classifier

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(self.device)
            feat = self.features(x)
            feat = feat.squeeze(-1).squeeze(-1)
        return feat.cpu().numpy()

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.to(self.device)
            feat = self.features(x)
            feat = feat.squeeze(-1).squeeze(-1)
            logits = self.classifier(feat)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()


def build_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])


def frechet_distance(mu1, sig1, mu2, sig2) -> float:
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    cov_mean, _ = sqrtm(sig1 @ sig2, disp=False)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fd = float(np.dot(diff, diff) + np.trace(sig1 + sig2 - 2 * cov_mean))
    return fd


def extract_embeddings(imgs: list, extractor: FeatureExtractor,
                       transform, batch_size: int = 8) -> np.ndarray:
    all_embs = []
    for i in range(0, len(imgs), batch_size):
        batch = [transform(img) for img in imgs[i:i+batch_size]]
        batch_t = torch.stack(batch)
        embs = extractor.embed(batch_t)
        all_embs.append(embs)
    return np.concatenate(all_embs, axis=0)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> float:
    """평균 NN cosine similarity: 각 generated 이미지에서 가장 가까운 real 이미지까지."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    sim = a_n @ b_n.T  # (N_gen, N_real)
    return float(sim.max(axis=1).mean())


# ── 파이프라인 로드 ───────────────────────────────────────────────────

def load_t2i_pipeline(lora_dir: Path, device: str):
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    use_fp16 = device == "cuda"
    dtype    = torch.float16 if use_fp16 else torch.float32
    variant  = "fp16"        if use_fp16 else None
    print(f"  Loading SDXL text-to-image pipeline ({'fp16' if use_fp16 else 'fp32'})...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, variant=variant, use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    lora_weights = lora_dir / "pytorch_lora_weights.safetensors"
    if lora_weights.exists():
        print(f"  Loading T2I LoRA weights from {lora_dir.name}...")
        pipe.load_lora_weights(str(lora_dir))
    else:
        print(f"  [WARN] No LoRA weights at {lora_dir}, using base SDXL")
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_t2i_images(
    pipe, class_name: str, n_gen: int,
    device: str, seed: int = 42,
    n_per_domain: int = None,
) -> list:
    """각 도메인 프롬프트로 균등하게 이미지 생성."""
    domains = list(DOMAIN_PROMPTS.keys())
    if n_per_domain is None:
        n_per_domain = max(1, n_gen // len(domains))

    generated = []
    for domain in domains:
        prompt = build_t2i_prompt(class_name, domain)
        print(f"  [{domain}] {n_per_domain}장 생성 중...")
        for i in range(n_per_domain):
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    height=512,
                    width=512,
                    generator=torch.Generator(device).manual_seed(seed + i),
                ).images[0]
            generated.append((result, domain))
            if len(generated) >= n_gen:
                break
        if len(generated) >= n_gen:
            break

    print(f"  총 {len(generated)}장 생성 완료")
    return generated  # list of (PIL Image, domain str)


# ── 평가 ─────────────────────────────────────────────────────────────

def evaluate_t2i(
    class_name: str,
    gen_imgs_with_domain: list,
    real_imgs: list,
    extractor: FeatureExtractor,
    transform,
    class_idx: int,
) -> dict:
    gen_imgs = [img for img, _ in gen_imgs_with_domain]

    print("\n  [평가] 임베딩 추출 중...")
    gen_embs  = extract_embeddings(gen_imgs, extractor, transform)
    real_embs = extract_embeddings(real_imgs, extractor, transform)

    # Fréchet Distance
    mu_g, sig_g = gen_embs.mean(0), np.cov(gen_embs, rowvar=False)
    mu_r, sig_r = real_embs.mean(0), np.cov(real_embs, rowvar=False)
    fd = frechet_distance(mu_g, sig_g, mu_r, sig_r)

    # NN Cosine Similarity
    cos_sim = cosine_sim_matrix(gen_embs, real_embs)

    # CNN Accuracy & Confidence
    correct = 0
    confidences = []
    for i in range(0, len(gen_imgs), 8):
        batch = [transform(img) for img in gen_imgs[i:i+8]]
        batch_t = torch.stack(batch)
        probs = extractor.predict(batch_t)
        for p in probs:
            pred = int(np.argmax(p))
            conf = float(p[class_idx])
            confidences.append(conf)
            if pred == class_idx:
                correct += 1

    accuracy   = correct / len(gen_imgs)
    avg_conf   = float(np.mean(confidences))

    return {
        "frechet_distance": round(fd, 4),
        "nn_cosine_sim":    round(cos_sim, 4),
        "cnn_accuracy":     round(accuracy, 4),
        "cnn_confidence":   round(avg_conf, 4),
        "n_generated":      len(gen_imgs),
        "n_real_reference": len(real_imgs),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="T2I LoRA 멀티도메인 생성 이미지 평가"
    )
    parser.add_argument("--class_name",  type=str, required=True,
                        choices=["basophil","eosinophil","lymphocyte","monocyte","neutrophil"])
    parser.add_argument("--n_gen",       type=int, default=20,
                        help="생성할 총 이미지 수 (도메인별 균등 분배)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no_generate", action="store_true",
                        help="기존 생성 이미지 재사용")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    cls       = args.class_name
    class_idx = CLASS_NAMES.index(cls)

    lora_dir  = ROOT / "lora" / "weights" / f"t2i_multidomain_{cls}"
    out_dir   = OUT_ROOT / cls / f"t2i_multidomain_{cls}"
    gen_dir   = out_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(exist_ok=True)

    # ── Real 이미지 로드 ─────────────────────────────────────────────
    real_paths = list((TRAIN_DIR / cls).rglob("*.jpg")) + \
                 list((TRAIN_DIR / cls).rglob("*.png"))
    random.seed(args.seed)
    n_real = min(100, len(real_paths))
    sample_real_paths = random.sample(real_paths, n_real)
    real_imgs = [Image.open(p).convert("RGB") for p in sample_real_paths]
    print(f"Real images: {n_real}장 로드 완료")

    # ── 생성 ────────────────────────────────────────────────────────
    if not args.no_generate:
        pipe = load_t2i_pipeline(lora_dir, device)
        gen_imgs_with_domain = generate_t2i_images(
            pipe, cls, args.n_gen, device, args.seed
        )
        del pipe
        if device == "mps":
            torch.mps.empty_cache()
        # 저장
        saved = []
        for i, (img, domain) in enumerate(gen_imgs_with_domain):
            fname = gen_dir / f"t2i_{cls}_{domain}_{i:04d}.png"
            img.save(fname)
            saved.append((img, domain))
        gen_imgs_with_domain = saved
        print(f"  저장: {gen_dir} ({len(saved)}장)")
    else:
        gen_paths = sorted(gen_dir.glob("*.png"))
        gen_imgs_with_domain = []
        for p in gen_paths:
            domain = "unknown"
            for d in DOMAIN_PROMPTS:
                if d in p.name:
                    domain = d
                    break
            gen_imgs_with_domain.append((Image.open(p).convert("RGB"), domain))
        print(f"  기존 생성 이미지 로드: {len(gen_imgs_with_domain)}장")

    if not gen_imgs_with_domain:
        print("[ERROR] 생성된 이미지 없음")
        return

    # ── 평가 ────────────────────────────────────────────────────────
    print("\n[평가 시작]")
    extractor = FeatureExtractor(MODEL_CKPT, n_classes=8, device=device)
    transform = build_transform()

    metrics = evaluate_t2i(cls, gen_imgs_with_domain, real_imgs,
                           extractor, transform, class_idx)

    # ── 기존 결과와 비교 ─────────────────────────────────────────────
    comparison = {}
    for model_name in ["basophil", "multidomain_basophil"]:
        report_path = OUT_ROOT / cls / model_name / "report.json"
        if report_path.exists():
            with open(report_path) as f:
                prev = json.load(f)
            agg = prev.get("aggregate", {})
            comparison[model_name] = {
                "frechet_distance": agg.get("frechet_distance", "N/A"),
                "nn_cosine_sim":    agg.get("nn_cosine_sim", "N/A"),
                "cnn_accuracy":     agg.get("cnn_accuracy_rate", "N/A"),
                "cnn_confidence":   agg.get("cnn_confidence_mean", "N/A"),
            }

    # ── 결과 출력 ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  T2I LoRA 멀티도메인 평가 결과: {cls.upper()}")
    print("="*70)
    print(f"  생성 이미지:    {metrics['n_generated']}장")
    print(f"  실제 이미지:    {metrics['n_real_reference']}장 (참조용)")
    print()
    print(f"  Fréchet Distance:  {metrics['frechet_distance']:>10.2f}")
    print(f"  NN Cosine Sim:     {metrics['nn_cosine_sim']:>10.4f}")
    print(f"  CNN Accuracy:      {metrics['cnn_accuracy']*100:>9.1f}%")
    print(f"  CNN Confidence:    {metrics['cnn_confidence']:>10.4f}")

    if comparison:
        print("\n" + "-"*70)
        print("  모델 비교:")
        print(f"  {'모델':<30} {'FD':>12} {'CosSim':>8} {'Acc':>8} {'Conf':>8}")
        print("  " + "-"*66)
        for model, vals in comparison.items():
            fd_val  = f"{vals['frechet_distance']:.2f}" if isinstance(vals['frechet_distance'], float) else str(vals['frechet_distance'])
            cos_val = f"{vals['nn_cosine_sim']:.4f}"    if isinstance(vals['nn_cosine_sim'], float) else str(vals['nn_cosine_sim'])
            acc_val = f"{vals['cnn_accuracy']*100:.1f}%" if isinstance(vals['cnn_accuracy'], float) else str(vals['cnn_accuracy'])
            conf_val= f"{vals['cnn_confidence']:.4f}"   if isinstance(vals['cnn_confidence'], float) else str(vals['cnn_confidence'])
            label = "단일도메인 DreamBooth" if model == cls else "멀티도메인 DreamBooth"
            print(f"  {label:<30} {fd_val:>12} {cos_val:>8} {acc_val:>8} {conf_val:>8}")

        # T2I 결과도 같은 줄에
        fd_v  = f"{metrics['frechet_distance']:.2f}"
        cos_v = f"{metrics['nn_cosine_sim']:.4f}"
        acc_v = f"{metrics['cnn_accuracy']*100:.1f}%"
        conf_v= f"{metrics['cnn_confidence']:.4f}"
        print(f"  {'T2I LoRA (이번)':<30} {fd_v:>12} {cos_v:>8} {acc_v:>8} {conf_v:>8}")

    print("="*70)

    # ── JSON 저장 ────────────────────────────────────────────────────
    result = {
        "model": f"t2i_multidomain_{cls}",
        "class": cls,
        "aggregate": {
            "frechet_distance":    metrics["frechet_distance"],
            "nn_cosine_sim":       metrics["nn_cosine_sim"],
            "cnn_accuracy_rate":   metrics["cnn_accuracy"],
            "cnn_confidence_mean": metrics["cnn_confidence"],
        },
        "n_generated": metrics["n_generated"],
        "comparison":  comparison,
    }
    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  결과 저장: {report_path}")


if __name__ == "__main__":
    main()
