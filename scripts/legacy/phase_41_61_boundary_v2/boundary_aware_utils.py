from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent.parent

CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_MORPHOLOGY = {
    "basophil": "bilobed nucleus with dark purple-black granules filling cytoplasm",
    "eosinophil": "bilobed nucleus with bright orange-red granules",
    "lymphocyte": "large round nucleus with scant agranular cytoplasm",
    "monocyte": "kidney-shaped or folded nucleus with grey-blue cytoplasm",
    "neutrophil": "multilobed nucleus with pale pink granules",
}

DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_SHORT = {
    "domain_a_pbc": "PBC",
    "domain_b_raabin": "Raabin",
    "domain_c_mll23": "MLL23",
    "domain_e_amc": "AMC",
}
DOMAIN_TOKENS = {
    "domain_a_pbc": "[DOM_PBC]",
    "domain_b_raabin": "[DOM_RAABIN]",
    "domain_c_mll23": "[DOM_MLL23]",
    "domain_e_amc": "[DOM_AMC]",
}
DOMAIN_STYLE_TEXT = {
    "domain_a_pbc": "May-Grunwald Giemsa stain, CellaVision analyzer, pale smear background",
    "domain_b_raabin": "Giemsa stain, smartphone microscope texture, vivid smear background",
    "domain_c_mll23": "Pappenheim stain, Metafer scanner texture, laboratory smear background",
    "domain_e_amc": "Romanowsky stain, miLab analyzer texture, clean smear background",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CNN_CKPT = ROOT / "models" / "multidomain_cnn.pt"
CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_contextual_prompt(class_name: str, domain: str) -> str:
    return (
        f"microscopy image of a single {class_name} white blood cell, "
        f"{CLASS_MORPHOLOGY[class_name]}, "
        f"{DOMAIN_TOKENS[domain]}, {DOMAIN_STYLE_TEXT[domain]}, "
        "peripheral blood smear background, realistic hematology imaging"
    )


def build_background_prompt(class_name: str, domain: str) -> str:
    return (
        f"{DOMAIN_TOKENS[domain]}, {DOMAIN_STYLE_TEXT[domain]}, "
        "blood smear background, stain texture, scanner appearance, realistic microscopy background"
    )


def stable_int_seed(text: str, mod: int = 10_000) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % mod


def resize_with_padding(img: Image.Image, canvas: int = 384, fill: tuple[int, int, int] = (245, 245, 245)) -> tuple[Image.Image, dict]:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(canvas / w, canvas / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    out = Image.new("RGB", (canvas, canvas), fill)
    left = (canvas - new_w) // 2
    top = (canvas - new_h) // 2
    out.paste(resized, (left, top))
    return out, {
        "original_size": [w, h],
        "resized_size": [new_w, new_h],
        "pad_left": left,
        "pad_top": top,
        "policy": "resize_with_padding",
    }


def bounded_center_jitter_crop(img: Image.Image, crop_size: int = 448, output_size: int = 384, key: str = "") -> tuple[Image.Image, dict]:
    img = img.convert("RGB")
    w, h = img.size
    crop = min(crop_size, w, h)
    max_dx = max(0, int(round(0.08 * crop)))
    max_dy = max(0, int(round(0.08 * crop)))
    seed = stable_int_seed(key or "crop")
    dx = (seed % (2 * max_dx + 1)) - max_dx
    dy = ((seed // 97) % (2 * max_dy + 1)) - max_dy
    cx = w // 2 + dx
    cy = h // 2 + dy
    left = min(max(0, cx - crop // 2), w - crop)
    top = min(max(0, cy - crop // 2), h - crop)
    cropped = img.crop((left, top, left + crop, top + crop)).resize((output_size, output_size), Image.LANCZOS)
    return cropped, {
        "original_size": [w, h],
        "crop_size": crop,
        "crop_left": int(left),
        "crop_top": int(top),
        "jitter_dx": int(dx),
        "jitter_dy": int(dy),
        "policy": "bounded_center_jitter_crop",
    }


def extract_cell_mask(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    val = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)

    sat_thr = max(20, int(np.percentile(sat, 60)))
    val_thr = int(np.percentile(val, 85))
    mask = ((sat > sat_thr) & (val < val_thr)).astype(np.uint8) * 255

    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    center_weight = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.22 * w) ** 2)))
    mask = np.where(center_weight > 0.15, mask, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return fallback_center_mask(mask.shape)

    best_idx = None
    best_score = -1.0
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        comp_cx = x + ww / 2.0
        comp_cy = y + hh / 2.0
        dist = np.hypot(comp_cx - cx, comp_cy - cy)
        score = area - 2.5 * dist
        if score > best_score:
            best_score = score
            best_idx = i
    mask = (labels == best_idx).astype(np.uint8) * 255
    mask = cv2.dilate(mask, kernel, iterations=2)
    if mask.sum() < 0.02 * h * w * 255:
        return fallback_center_mask(mask.shape)
    return mask


def fallback_center_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    ellipse = (((xx - cx) / (0.22 * w)) ** 2 + ((yy - cy) / (0.18 * h)) ** 2) <= 1.0
    return ellipse.astype(np.uint8) * 255


def mask_qc(mask: np.ndarray) -> dict:
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {
            "area_ratio": 0.0,
            "center_overlap": 0.0,
            "bbox_width": 0,
            "bbox_height": 0,
            "fallback_required": True,
        }
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    area_ratio = float((mask > 0).mean())

    center_mask = fallback_center_mask(mask.shape) > 0
    center_overlap = float(((mask > 0) & center_mask).sum() / max(1, center_mask.sum()))
    return {
        "area_ratio": round(area_ratio, 4),
        "center_overlap": round(center_overlap, 4),
        "bbox_width": int(x1 - x0 + 1),
        "bbox_height": int(y1 - y0 + 1),
        "fallback_required": False,
    }


def masked_similarity(img_a: Image.Image, img_b: Image.Image, mask: np.ndarray, size: int = 224) -> float:
    a = np.array(img_a.resize((size, size)).convert("RGB")).astype(np.float32) / 255.0
    b = np.array(img_b.resize((size, size)).convert("RGB")).astype(np.float32) / 255.0
    m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 0
    if m.sum() < 32:
        return 0.0
    mse = ((a - b) ** 2)[m].mean()
    return float(max(0.0, 1.0 - mse * 3.0))


def load_cnn(device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    ckpt = torch.load(CNN_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(device)


@torch.no_grad()
def cnn_prob_vector(model: nn.Module, img: Image.Image, device: torch.device) -> np.ndarray:
    x = CNN_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    return probs


def entropy_margin_target(probs: np.ndarray, target_idx: int) -> tuple[float, float, float, int]:
    eps = 1e-8
    entropy = float(-(probs * np.log(probs + eps)).sum())
    target_prob = float(probs[target_idx])
    max_other = float(np.max(np.delete(probs, target_idx)))
    margin = target_prob - max_other
    pred_idx = int(np.argmax(probs))
    return entropy, margin, target_prob, pred_idx


def boundary_score(cell_ssim: float, background_ssim: float, entropy: float, margin: float) -> float:
    return float((1.0 - background_ssim) + 0.5 * entropy - 0.25 * max(margin, 0.0) + 0.2 * cell_ssim)


def ensure_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
