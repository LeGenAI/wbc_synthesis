"""
Step 6: Robustness / corruption evaluation.

Applies 5 corruption types × 3 severity levels to the clean test set,
then evaluates each trained model checkpoint.

Corruptions:
  1. gaussian_blur      (sigma: 1, 2, 3)
  2. gaussian_noise     (std: 0.05, 0.1, 0.2)
  3. jpeg_compression   (quality: 60, 40, 20)
  4. brightness_contrast (alpha/beta: mild/moderate/strong)
  5. stain_shift        (HSV hue/sat jitter: mild/moderate/strong)

Output:
  results/corrupted/{ckpt_name}_corruption_results.json
    → per-corruption, per-severity:  accuracy, macro-F1
    → summary: mean performance drop vs clean

Usage:
    python 06_robustness_eval.py --ckpt models/real_only_efficientnet_b0_seed42_best.pt
    python 06_robustness_eval.py --all_ckpts   # evaluates all .pt files in models/
"""
import argparse
import json
import io
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
TEST_DIR   = ROOT / "data" / "processed" / "test"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "corrupted"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_WORKERS = 4


# ── Corruption functions (operate on numpy uint8 HWC BGR) ──────────
def gaussian_blur(img: np.ndarray, severity: int) -> np.ndarray:
    sigma = [1, 2, 3][severity - 1]
    k = int(6 * sigma + 1) | 1   # make odd
    return cv2.GaussianBlur(img, (k, k), sigma)


def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    std = [0.05, 0.10, 0.20][severity - 1]
    noise = (np.random.randn(*img.shape) * std * 255).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    quality = [60, 40, 20][severity - 1]
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def brightness_contrast(img: np.ndarray, severity: int) -> np.ndarray:
    # alpha: contrast, beta: brightness
    params = [(1.2, 15), (1.4, 30), (1.7, 50)][severity - 1]
    alpha, beta = params
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def stain_shift(img: np.ndarray, severity: int) -> np.ndarray:
    """Simulate staining variation by jittering HSV hue/saturation."""
    h_shift, s_scale = [(5, 0.9), (12, 0.75), (20, 0.60)][severity - 1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


CORRUPTIONS: dict[str, Callable] = {
    "gaussian_blur":      gaussian_blur,
    "gaussian_noise":     gaussian_noise,
    "jpeg_compression":   jpeg_compression,
    "brightness_contrast":brightness_contrast,
    "stain_shift":        stain_shift,
}
SEVERITIES = [1, 2, 3]


# ── Dataset ────────────────────────────────────────────────────────
class CorruptedTestDataset(Dataset):
    def __init__(self, test_dir: Path, class_to_idx: dict,
                 corrupt_fn: Callable | None, severity: int,
                 transform):
        self.samples    = []
        self.corrupt_fn = corrupt_fn
        self.severity   = severity
        self.transform  = transform
        for cls, idx in class_to_idx.items():
            cls_dir = test_dir / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.rglob("*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(str(path))
        if self.corrupt_fn is not None:
            img = self.corrupt_fn(img, self.severity)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return self.transform(pil), label


# ── Model loading ──────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device):
    ckpt        = torch.load(ckpt_path, map_location=device)
    model_name  = ckpt.get("model_name", "efficientnet_b0")
    class_names = ckpt["class_names"]
    n_cls       = len(class_names)

    if model_name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, n_cls)
    elif model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, n_cls)
    elif model_name == "efficientnet_b2":
        m = models.efficientnet_b2(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, n_cls)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m.to(device), class_names


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def run_eval(model, loader, device):
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out  = model(imgs)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, macro_f1


def evaluate_checkpoint(ckpt_path: Path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model, class_names = load_model(ckpt_path, device)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    transform    = get_transform()

    print(f"\n{'='*65}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Classes:    {class_names}")

    results = {}

    # ── Clean baseline ─────────────────────────────────────────────
    clean_ds     = CorruptedTestDataset(TEST_DIR, class_to_idx, None, 0, transform)
    clean_loader = DataLoader(clean_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    clean_acc, clean_f1 = run_eval(model, clean_loader, device)
    print(f"  Clean → acc={clean_acc:.4f}  macro_f1={clean_f1:.4f}")
    results["clean"] = {"accuracy": clean_acc, "macro_f1": clean_f1}

    # ── Corrupted evaluations ──────────────────────────────────────
    drops_f1 = []
    for corr_name, corr_fn in CORRUPTIONS.items():
        results[corr_name] = {}
        for sev in SEVERITIES:
            ds     = CorruptedTestDataset(TEST_DIR, class_to_idx, corr_fn, sev, transform)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS)
            acc, f1 = run_eval(model, loader, device)
            drop_f1 = clean_f1 - f1
            drops_f1.append(drop_f1)
            results[corr_name][f"sev{sev}"] = {
                "accuracy": acc,
                "macro_f1": f1,
                "f1_drop_vs_clean": drop_f1,
            }
            print(f"  {corr_name:25s} sev{sev}  "
                  f"acc={acc:.4f}  f1={f1:.4f}  drop={drop_f1:+.4f}")

    results["summary"] = {
        "mean_f1_drop":   float(np.mean(drops_f1)),
        "max_f1_drop":    float(np.max(drops_f1)),
        "n_corruptions":  len(drops_f1),
    }
    print(f"\n  Mean F1 drop across corruptions: {np.mean(drops_f1):+.4f}")
    print(f"  Max  F1 drop across corruptions: {np.max(drops_f1):+.4f}")

    out_path = RESULTS_DIR / f"{ckpt_path.stem}_corruption.json"
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": str(ckpt_path),
            "class_names": class_names,
            "results": results,
        }, f, indent=2)
    print(f"  Saved → {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       type=str, default=None,
                        help="Path to a specific .pt checkpoint")
    parser.add_argument("--all_ckpts",  action="store_true",
                        help="Evaluate all .pt files in models/")
    args = parser.parse_args()

    if args.all_ckpts:
        ckpts = sorted(MODELS_DIR.glob("*.pt"))
        if not ckpts:
            print(f"No .pt files found in {MODELS_DIR}")
            return
        for ckpt in ckpts:
            evaluate_checkpoint(ckpt)
    elif args.ckpt:
        evaluate_checkpoint(Path(args.ckpt))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
