"""
Script 15: WBC Router Inference System
=======================================
5가지 최신 트렌드를 반영한 강건한 라우터 + img2img 생성 파이프라인.

Trend 1. Domain-conditioned generation  — 예측된 도메인 캡션을 LoRA 프롬프트에 주입
Trend 2. Soft Routing (MoE)             — top-k 전문가 LoRA를 확률 가중으로 blend
Trend 3. Dual-head router               — 단일 forward pass로 클래스+도메인 동시 예측
Trend 4. TTA (FiveCrop 앙상블)           — 5-crop softmax 평균으로 경계 이미지 강건화
Trend 5. Confidence-aware generation    — 저신뢰도 → 복수 후보 생성 → SSIM 최고 선택

Usage:
    # 단일 이미지
    python scripts/legacy/phase_08_17_domain_gap_multidomain/15_router_inference.py \\
        --router_ckpt models/dual_head_router.pt \\
        --image path/to/cell.jpg \\
        --output_dir results/router_test/

    # 배치 (디렉토리, 클래스별 n장)
    python scripts/legacy/phase_08_17_domain_gap_multidomain/15_router_inference.py \\
        --router_ckpt models/dual_head_router.pt \\
        --image_dir data/processed_multidomain/domain_e_amc/ \\
        --n_per_class 5 --output_dir results/router_test/

    # 분류만 (생성 없이)
    python scripts/legacy/phase_08_17_domain_gap_multidomain/15_router_inference.py \\
        --router_ckpt models/dual_head_router.pt \\
        --image_dir ... --classify_only

    # DualHeadRouter 없이 (baseline_cnn fallback, domain 항상 pbc)
    python scripts/legacy/phase_08_17_domain_gap_multidomain/15_router_inference.py \\
        --cnn_ckpt models/baseline_cnn.pt \\
        --image path/to/cell.jpg
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "processed_multidomain"
LORA_ROOT = ROOT / "lora" / "weights"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# ── 메타데이터 ─────────────────────────────────────────────────────────
MULTI_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_IDX     = {c: i for i, c in enumerate(MULTI_CLASSES)}

DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_IDX = {d: i for i, d in enumerate(DOMAINS)}

# multidomain_cnn.pt는 5클래스 직접 학습 → 슬라이싱 불필요
# (구 baseline_cnn.pt 8클래스 슬라이싱 방식 폐기)

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# ── 형태학적 설명 (10_multidomain_lora_train.py에서 재사용) ──────────
CLASS_MORPHOLOGY = {
    "basophil":   "bilobed nucleus with dark purple-black granules filling cytoplasm",
    "eosinophil": "bilobed nucleus with bright orange-red granules",
    "lymphocyte": "large round nucleus with scant agranular cytoplasm",
    "monocyte":   "kidney-shaped or folded nucleus with grey-blue cytoplasm",
    "neutrophil": "multilobed nucleus with pale pink granules",
}

# ── 도메인 캡션 (학습 시와 동일한 프롬프트 — 학습-추론 일관성) ────────
DOMAIN_PROMPTS_INFERENCE = {
    0: "May-Grünwald Giemsa stain, CellaVision automated analyzer, Barcelona Spain",
    1: "Giemsa stain, smartphone microscope camera, Iran hospital",
    2: "Pappenheim stain, Metafer scanner, Germany clinical lab",
    3: "Romanowsky stain, miLab automated analyzer, South Korea AMC",
}

NEGATIVE_PROMPT = (
    "cartoon, illustration, text, watermark, multiple cells, "
    "heavy artifacts, unrealistic colors, deformed nucleus, blurry"
)

# ── Confidence 임계값 (Trend 5) ───────────────────────────────────────
HIGH_CONF = 0.85  # 단일 생성으로 충분
LOW_CONF  = 0.55  # 이 이하는 복수 생성 후 SSIM 최고 선택


# ── 디바이스 ──────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── SSIM (13_gallery_report.py L44에서 재사용) ───────────────────────
def ssim_pair(img_a: Image.Image, img_b: Image.Image, size: int = 224) -> float:
    import numpy as np
    a = np.array(img_a.resize((size, size)).convert("L")).astype(float) / 255.
    b = np.array(img_b.resize((size, size)).convert("L")).astype(float) / 255.
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(), b.std()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = 0.01**2, 0.03**2
    return float((2*mu_a*mu_b + C1) * (2*sig_ab + C2) /
                 ((mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2)))


# ── DualHeadRouter (14_train_router.py와 동일 구조) ───────────────────
class DualHeadRouter(nn.Module):
    """
    EfficientNet-B0 기반 Dual-Head 라우터 (추론 전용).
    forward(x) → (class_logits[B,5], domain_logits[B,4])
    """

    def __init__(self, ckpt_path: Path, device_str: str):
        super().__init__()

        ckpt  = torch.load(ckpt_path, map_location=device_str, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        # 백본 (EfficientNet-B0 features + avgpool)
        # classifier[1] 크기는 backbone 추출에 무관하나 명시적으로 5로 설정
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, 5)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # class_head (5클래스, raw embedding 사용)
        self.class_head = nn.Linear(1280, 5, bias=True)

        # domain_head (MLP 1280→512→4, L2-norm embedding 사용)
        # 14_train_router.py와 동일 구조 필수
        self.domain_head = nn.Sequential(
            nn.Linear(1280, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 4, bias=True),
        )

        # 가중치 로드
        self.load_state_dict(state)
        self.eval()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) → (B, 1280) raw embedding"""
        return self.backbone(x).flatten(1)

    def forward(self, x: torch.Tensor):
        raw_emb  = self.embed(x)
        norm_emb = F.normalize(raw_emb, p=2, dim=1)  # domain_head용 L2-norm
        return self.class_head(raw_emb), self.domain_head(norm_emb)


class _FallbackRouter(nn.Module):
    """
    dual_head_router.pt가 없을 때 사용하는 fallback.
    multidomain_cnn.pt (5클래스 직접 학습) → 클래스 분류, domain은 항상 0(pbc).
    """

    def __init__(self, ckpt_path: Path, device_str: str):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location=device_str, weights_only=False)

        # multidomain_cnn.pt: 5클래스 직접 학습 (슬라이싱 불필요)
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, 5)
        base.load_state_dict(ckpt["model_state_dict"])
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # class_head 직접 로드 (5클래스, 슬라이싱 없음)
        w = ckpt["model_state_dict"]["classifier.1.weight"]  # [5, 1280]
        b = ckpt["model_state_dict"]["classifier.1.bias"]    # [5]
        self.class_head = nn.Linear(1280, 5, bias=True)
        self.class_head.weight = nn.Parameter(w.clone())
        self.class_head.bias   = nn.Parameter(b.clone())

        # 더미 domain_head (항상 pbc=0 반환)
        self.domain_head = nn.Linear(1280, 4, bias=True)
        nn.init.zeros_(self.domain_head.weight)
        nn.init.constant_(self.domain_head.bias, 0.0)
        self.domain_head.bias.data[0] = 10.0  # index 0 (pbc) 항상 선택
        self.eval()

    def forward(self, x: torch.Tensor):
        emb = self.backbone(x).flatten(1)
        return self.class_head(emb), self.domain_head(emb)


# ── 프롬프트 빌더 ─────────────────────────────────────────────────────
def build_class_domain_prompt(class_idx: int, domain_idx: int) -> str:
    """Trend 1: 클래스 형태 + 도메인 캡션 합성 프롬프트."""
    cls_name   = MULTI_CLASSES[class_idx]
    morphology = CLASS_MORPHOLOGY[cls_name]
    domain_desc = DOMAIN_PROMPTS_INFERENCE[domain_idx]
    return (
        f"microscopy image of a single {cls_name} white blood cell, "
        f"peripheral blood smear, {morphology}, "
        f"{domain_desc}, "
        f"sharp focus, realistic, clinical lab imaging"
    )


# ── WBCRouter ─────────────────────────────────────────────────────────
class WBCRouter:
    """
    5가지 트렌드를 통합한 WBC 라우터 + 생성 시스템.

    router_ckpt: dual_head_router.pt (DualHeadRouter)
    cnn_ckpt:    baseline_cnn.pt (fallback용)
    """

    def __init__(
        self,
        router_ckpt: Optional[Path],
        cnn_ckpt: Path,
        device: Optional[str] = None,
    ):
        self.device = device or get_device()
        self.pipe   = None
        self._loaded_adapters: set[str] = set()

        print(f"  WBCRouter 초기화: device={self.device}")

        # ── Trend 3: Dual-head router 로드 ───────────────────────────
        if router_ckpt is not None and Path(router_ckpt).exists():
            print(f"  DualHeadRouter 로드: {router_ckpt}")
            self.router = DualHeadRouter(router_ckpt, self.device).to(self.device)
            self._has_domain_head = True
        else:
            warnings.warn(
                f"  [WARN] router_ckpt 없음 ({router_ckpt}). "
                f"baseline_cnn fallback 사용 (domain=pbc 고정)."
            )
            self.router = _FallbackRouter(cnn_ckpt, self.device).to(self.device)
            self._has_domain_head = False

        self.router.eval()

        # ── TTA 변환 ─────────────────────────────────────────────────
        _normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])
        self._tta_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.FiveCrop(224),
            # FiveCrop → 5개 PIL 이미지 튜플 → 각각 ToTensor+Normalize → stack
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Compose([transforms.ToTensor(), _normalize])(c)
                for c in crops
            ])),  # → (5, 3, 224, 224)
        ])
        self._std_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            _normalize,
        ])

    # ── Trend 4: TTA 예측 ────────────────────────────────────────────
    @torch.no_grad()
    def tta_predict(
        self, img: Image.Image, n_crops: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        FiveCrop TTA: 5-crop softmax 평균.
        반환: (class_probs[5], domain_probs[4])
        """
        crops_tensor = self._tta_transform(img.convert("RGB"))  # (5,3,224,224)
        cls_list, dom_list = [], []
        for i in range(n_crops):
            inp = crops_tensor[i].unsqueeze(0).to(self.device)
            cl, dl = self.router(inp)
            cls_list.append(F.softmax(cl, dim=1))
            dom_list.append(F.softmax(dl, dim=1))
        cls_probs = torch.stack(cls_list).mean(0).squeeze(0).cpu()
        dom_probs = torch.stack(dom_list).mean(0).squeeze(0).cpu()
        return cls_probs, dom_probs

    @torch.no_grad()
    def _single_predict(
        self, img: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TTA 없는 단일 예측."""
        inp = self._std_transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        cl, dl = self.router(inp)
        return F.softmax(cl, dim=1).squeeze(0).cpu(), F.softmax(dl, dim=1).squeeze(0).cpu()

    # ── 파이프라인 지연 로딩 ─────────────────────────────────────────
    def _load_lora_pipeline(self) -> None:
        """
        SDXL img2img 파이프라인 + 존재하는 multidomain LoRA 전체 로드.
        각 LoRA는 별도 adapter_name으로 등록.
        """
        from diffusers import (StableDiffusionXLImg2ImgPipeline,
                                DPMSolverMultistepScheduler)

        use_fp16 = self.device == "cuda"
        dtype    = torch.float16 if use_fp16 else torch.float32
        variant  = "fp16" if use_fp16 else None

        print(f"  SDXL 파이프라인 로드 ({'fp16' if use_fp16 else 'fp32'})...")
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            variant=variant,
            use_safetensors=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        pipe = pipe.to(self.device)
        pipe.enable_attention_slicing()

        # Trend 2: 각 LoRA를 adapter_name으로 개별 등록
        self._loaded_adapters = set()
        for cls in MULTI_CLASSES:
            lora_dir = LORA_ROOT / f"multidomain_{cls}"
            weights  = lora_dir / "pytorch_lora_weights.safetensors"
            if weights.exists():
                adapter_name = f"lora_{cls}"
                print(f"  LoRA 로드: {adapter_name}")
                pipe.load_lora_weights(str(lora_dir), adapter_name=adapter_name)
                self._loaded_adapters.add(adapter_name)
            else:
                print(f"  [SKIP] LoRA 없음: multidomain_{cls}")

        self.pipe = pipe
        print(f"  파이프라인 준비 완료 (adapter: {self._loaded_adapters})")

    # ── Trend 2: Soft Routing ─────────────────────────────────────────
    def _apply_routing(
        self,
        class_probs: torch.Tensor,
        class_idx: int,
        conf: float,
        conf_threshold: float,
        top_k: int,
    ) -> str:
        """
        conf ≥ threshold → 단일 전문가(hard routing)
        conf < threshold → top-k LoRA 확률 가중 blend(soft routing)
        반환: routing_mode 문자열
        """
        available = [c for c in MULTI_CLASSES
                     if f"lora_{c}" in self._loaded_adapters]

        if not available:
            # LoRA 없음: base SDXL
            self.pipe.disable_lora()
            return "base_sdxl"

        cls_name = MULTI_CLASSES[class_idx]

        if conf >= conf_threshold:
            # Hard routing: 단일 전문가
            adapter = f"lora_{cls_name}"
            if adapter in self._loaded_adapters:
                self.pipe.set_adapters([adapter], adapter_weights=[1.0])
                return "single"
            else:
                self.pipe.disable_lora()
                return "base_sdxl_no_lora"

        # Soft routing: 존재하는 어댑터 중 top-k 가중 blend
        avail_idx    = [MULTI_CLASSES.index(c) for c in available]
        avail_probs  = class_probs[avail_idx]
        k            = min(top_k, len(available))
        topk_vals, topk_local = torch.topk(avail_probs, k)
        weights      = (topk_vals / topk_vals.sum()).tolist()
        selected_cls = [available[i] for i in topk_local.tolist()]
        adapter_names = [f"lora_{c}" for c in selected_cls]

        self.pipe.set_adapters(
            adapter_names=adapter_names,
            adapter_weights=weights,
        )
        return f"soft_k{k}({'+'.join(selected_cls)})"

    # ── 단일 생성 ─────────────────────────────────────────────────────
    def _generate_once(
        self, img: Image.Image, prompt: str, denoise: float, seed: int
    ) -> Image.Image:
        ref = img.convert("RGB").resize((512, 512))
        gen = torch.Generator(self.device).manual_seed(seed)
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=ref,
                strength=denoise,
                guidance_scale=6.0,
                num_inference_steps=25,
                generator=gen,
            ).images[0]
        return result

    # ── Trend 5: Confidence-aware 생성 ───────────────────────────────
    def _confidence_aware_generate(
        self,
        img: Image.Image,
        prompt: str,
        conf: float,
        denoise: float,
        n_candidates: int,
        base_seed: int,
    ) -> tuple[Image.Image, int]:
        """
        신뢰도에 따라 생성 후보 수를 조절하고 SSIM 최고 이미지 반환.
        반환: (best_image, n_generated)
        """
        if conf >= HIGH_CONF:
            n = 1
        elif conf >= LOW_CONF:
            n = n_candidates
        else:
            n = n_candidates * 2

        candidates = [
            self._generate_once(img, prompt, denoise, seed=base_seed + i)
            for i in range(n)
        ]
        if len(candidates) == 1:
            return candidates[0], 1

        best = max(candidates, key=lambda gen: ssim_pair(gen, img))
        return best, n

    # ── 메인 라우팅 API ───────────────────────────────────────────────
    def route(
        self,
        img: Image.Image,
        conf_threshold: float = 0.7,
        top_k: int = 2,
        use_tta: bool = True,
        n_gen_candidates: int = 3,
        denoise: float = 0.35,
        seed: int = 42,
        generate: bool = True,
    ) -> dict:
        """
        분류 → 라우팅 → 생성 end-to-end 파이프라인.

        반환 dict:
          class_name, class_idx, class_conf, class_probs,
          domain_name, domain_idx, domain_conf, domain_probs,
          prompt, routing_mode,
          generated (PIL.Image | None),
          n_gen_candidates, best_ssim
        """
        # Trend 3+4: Dual-head + TTA 예측
        if use_tta:
            cls_probs, dom_probs = self.tta_predict(img)
        else:
            cls_probs, dom_probs = self._single_predict(img)

        class_idx  = cls_probs.argmax().item()
        domain_idx = dom_probs.argmax().item()
        class_conf = cls_probs.max().item()
        domain_conf = dom_probs.max().item()

        result = {
            "class_name":   MULTI_CLASSES[class_idx],
            "class_idx":    class_idx,
            "class_conf":   round(class_conf, 4),
            "class_probs":  {c: round(cls_probs[i].item(), 4)
                             for i, c in enumerate(MULTI_CLASSES)},
            "domain_name":  DOMAINS[domain_idx],
            "domain_idx":   domain_idx,
            "domain_conf":  round(domain_conf, 4),
            "domain_probs": {d: round(dom_probs[i].item(), 4)
                             for i, d in enumerate(DOMAINS)},
            "generated":    None,
            "routing_mode": "classify_only",
            "n_gen_candidates": 0,
            "best_ssim":    None,
        }

        if not generate:
            return result

        # 파이프라인 지연 로딩
        if self.pipe is None:
            self._load_lora_pipeline()

        # Trend 2: Soft/Hard routing
        routing_mode = self._apply_routing(
            cls_probs, class_idx, class_conf, conf_threshold, top_k
        )

        # Trend 1: Domain-conditioned prompt
        prompt = build_class_domain_prompt(class_idx, domain_idx)

        # Trend 5: Confidence-aware 복수 생성
        gen_img, n_gen = self._confidence_aware_generate(
            img, prompt, class_conf, denoise, n_gen_candidates, seed
        )
        best_ssim = ssim_pair(gen_img, img)

        result.update({
            "prompt":           prompt,
            "routing_mode":     routing_mode,
            "generated":        gen_img,
            "n_gen_candidates": n_gen,
            "best_ssim":        round(best_ssim, 4),
        })
        return result

    def cleanup(self):
        """파이프라인 메모리 해제."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()


# ── 배치 처리 헬퍼 ────────────────────────────────────────────────────
def collect_image_paths(image_dir: Path, n_per_class: int, seed: int = 42) -> list:
    """
    image_dir/{class}/ 구조 또는 이미지 파일 직접.
    n_per_class: 클래스별 최대 샘플 수 (0이면 전체).
    """
    rng = random.Random(seed)
    paths = []

    # 클래스 서브디렉토리 구조인지 확인
    subdirs = [d for d in image_dir.iterdir()
               if d.is_dir() and d.name in MULTI_CLASSES]
    if subdirs:
        for sub in subdirs:
            imgs = [p for p in sub.iterdir() if p.suffix.lower() in IMG_EXTS]
            if n_per_class > 0 and len(imgs) > n_per_class:
                imgs = rng.sample(imgs, n_per_class)
            paths.extend((p, sub.name) for p in imgs)
    else:
        # 단층 디렉토리: 모든 이미지 수집
        imgs = [p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        if n_per_class > 0 and len(imgs) > n_per_class:
            imgs = rng.sample(imgs, n_per_class)
        paths = [(p, "unknown") for p in imgs]

    return paths


# ── argparse ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="WBC Router Inference: 분류 + 전문가 LoRA img2img 생성"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",     type=Path, help="단일 이미지 경로")
    src.add_argument("--image_dir", type=Path, help="이미지 디렉토리")

    p.add_argument("--router_ckpt", type=Path, default=None,
                   help="dual_head_router.pt 경로 (없으면 baseline_cnn fallback)")
    p.add_argument("--cnn_ckpt",    type=Path,
                   default=ROOT / "models" / "baseline_cnn.pt",
                   help="baseline_cnn.pt 경로 (fallback용)")

    p.add_argument("--output_dir",  type=Path,
                   default=ROOT / "results" / "router_test")
    p.add_argument("--n_per_class", type=int, default=5,
                   help="클래스별 처리 이미지 수 (0=전체)")
    p.add_argument("--classify_only", action="store_true",
                   help="분류만 수행, 이미지 생성 없음")
    p.add_argument("--conf_threshold", type=float, default=0.7,
                   help="Hard/Soft routing 전환 신뢰도 임계값")
    p.add_argument("--top_k",         type=int,   default=2,
                   help="Soft routing 전문가 수")
    p.add_argument("--no_tta",        action="store_true",
                   help="TTA 비활성화 (빠른 추론)")
    p.add_argument("--n_candidates",  type=int,   default=3,
                   help="Confidence-aware 생성 후보 수 (기본)")
    p.add_argument("--denoise",       type=float, default=0.35,
                   help="img2img strength")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Script 15 — WBC Router Inference")
    print(f"  classify_only={args.classify_only}, tta={not args.no_tta}")
    print(f"  conf_threshold={args.conf_threshold}, top_k={args.top_k}")
    print(f"  denoise={args.denoise}, n_candidates={args.n_candidates}")
    print(f"{'='*60}")

    router = WBCRouter(
        router_ckpt=args.router_ckpt,
        cnn_ckpt=args.cnn_ckpt,
    )

    # ── 이미지 목록 수집 ────────────────────────────────────────────
    if args.image:
        img_list = [(args.image, "unknown")]
    else:
        img_list = collect_image_paths(args.image_dir, args.n_per_class, args.seed)

    print(f"\n  처리 이미지: {len(img_list)}장\n")

    all_results = []
    for i, (img_path, true_cls) in enumerate(tqdm(img_list, desc="  routing")):
        img = Image.open(img_path).convert("RGB")

        res = router.route(
            img,
            conf_threshold=args.conf_threshold,
            top_k=args.top_k,
            use_tta=(not args.no_tta),
            n_gen_candidates=args.n_candidates,
            denoise=args.denoise,
            seed=args.seed + i,
            generate=(not args.classify_only),
        )

        # 생성 이미지 저장
        if res["generated"] is not None:
            cls_dir = args.output_dir / res["class_name"]
            cls_dir.mkdir(parents=True, exist_ok=True)

            stem = img_path.stem
            img.save(cls_dir / f"input_{stem}.png")
            res["generated"].save(cls_dir / f"generated_{stem}.png")

        # summary용 직렬화 (PIL 제거)
        summary_entry = {k: v for k, v in res.items() if k != "generated"}
        summary_entry["input_path"]  = str(img_path)
        summary_entry["true_class"]  = true_cls
        summary_entry["correct"]     = (res["class_name"] == true_cls
                                        if true_cls in MULTI_CLASSES else None)
        all_results.append(summary_entry)

        # 진행 로그
        correct_str = ""
        if summary_entry["correct"] is not None:
            correct_str = " ✅" if summary_entry["correct"] else " ❌"
        print(f"  [{i+1:3d}/{len(img_list)}] "
              f"{img_path.name} → "
              f"class={res['class_name']}({res['class_conf']:.2f}){correct_str} "
              f"domain={res['domain_name']}({res['domain_conf']:.2f}) "
              f"mode={res['routing_mode']}"
              + (f" ssim={res['best_ssim']}" if res['best_ssim'] else ""))

    # ── 결과 저장 ─────────────────────────────────────────────────────
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_images":    len(all_results),
            "args": {k: str(v) for k, v in vars(args).items()},
            "results":     all_results,
        }, f, indent=2, ensure_ascii=False)

    # ── 집계 통계 ─────────────────────────────────────────────────────
    correct_list = [r["correct"] for r in all_results if r["correct"] is not None]
    ssim_list    = [r["best_ssim"] for r in all_results if r["best_ssim"] is not None]
    mode_counts  = {}
    for r in all_results:
        m = r["routing_mode"].split("(")[0]  # soft_k2(xxx) → soft_k2
        mode_counts[m] = mode_counts.get(m, 0) + 1

    print(f"\n{'='*60}")
    print(f"  요약")
    print(f"{'='*60}")
    if correct_list:
        print(f"  분류 정확도: {sum(correct_list)/len(correct_list)*100:.1f}% "
              f"({sum(correct_list)}/{len(correct_list)})")
    if ssim_list:
        print(f"  생성 SSIM:   avg={np.mean(ssim_list):.4f}, "
              f"min={min(ssim_list):.4f}, max={max(ssim_list):.4f}")
    print(f"  라우팅 분포: {mode_counts}")
    print(f"  결과 저장:   {summary_path}")

    router.cleanup()
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
