"""
Step 5: CNN training and evaluation.

Modes:
  --mode real_only          : Train on real images only (Baseline 1)
  --mode real_augmented     : Train on real + traditional augmentation (Baseline 2)
  --mode real_generated     : Train on real + SDXL-filtered generated images (Exp)

Models: resnet50 | efficientnet_b0 | efficientnet_b2

Usage:
    # Baseline 1
    python 05_train_cnn.py --mode real_only --model efficientnet_b0

    # Baseline 2
    python 05_train_cnn.py --mode real_augmented --model efficientnet_b0

    # Experiment: real + 1× generated (denoise 0.35)
    python 05_train_cnn.py --mode real_generated --model efficientnet_b0 \
        --gen_multiplier 1 --denoise_tag ds035

    # Run all three modes in sequence
    python 05_train_cnn.py --run_all --model efficientnet_b0
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)
from tqdm import tqdm
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
PROC_DIR     = ROOT / "data" / "processed"
FILTERED_DIR = ROOT / "data" / "filtered"
MODELS_DIR   = ROOT / "models"
RESULTS_DIR  = ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Training config ────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-4
LR_STEP_SIZE = 10
LR_GAMMA     = 0.5
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
SEED         = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Transforms ─────────────────────────────────────────────────────
def get_transforms(mode: str, augment: bool = False):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


# ── Dataset helpers ────────────────────────────────────────────────
class LabeledImageFolder(Dataset):
    """Flat image folder with a fixed class-to-idx mapping."""
    def __init__(self, root: Path, class_to_idx: dict, transform=None):
        self.samples = []
        self.transform = transform
        for cls, idx in class_to_idx.items():
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.rglob("*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataset(mode: str, gen_multiplier: int, denoise_tag: str,
                  class_to_idx: dict, train_tf, val_tf):
    """Build train/val/test datasets depending on mode."""
    train_real = LabeledImageFolder(PROC_DIR / "train", class_to_idx, train_tf)
    val_ds     = LabeledImageFolder(PROC_DIR / "val",   class_to_idx, val_tf)
    test_ds    = LabeledImageFolder(PROC_DIR / "test",  class_to_idx, val_tf)

    if mode == "real_only" or mode == "real_augmented":
        train_ds = train_real
    elif mode == "real_generated":
        gen_ds = LabeledImageFolder(
            FILTERED_DIR,
            class_to_idx,
            train_tf,
        )
        # Subsample by multiplier
        real_size = len(train_real)
        gen_cap   = real_size * gen_multiplier
        if len(gen_ds) > gen_cap:
            indices = random.sample(range(len(gen_ds)), gen_cap)
            from torch.utils.data import Subset
            gen_ds = Subset(gen_ds, indices)
        print(f"  real={len(train_real)}  gen={len(gen_ds)}  total={len(train_real)+len(gen_ds)}")
        train_ds = ConcatDataset([train_real, gen_ds])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return train_ds, val_ds, test_ds


# ── Model ──────────────────────────────────────────────────────────
def build_model(name: str, n_classes: int) -> nn.Module:
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif name == "efficientnet_b2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


# ── Training loop ──────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0., 0, 0
    device_type = device.type
    use_amp = device_type != "cpu"
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            out  = model(imgs)
            loss = criterion(out, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    device_type = device.type
    use_amp = device_type != "cpu"
    total_loss, all_preds, all_labels = 0., [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            out  = model(imgs)
            loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n = len(all_labels)
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, acc, macro_f1, all_preds, all_labels


def train(
    mode: str,
    model_name: str,
    gen_multiplier: int = 1,
    denoise_tag: str = "ds035",
    seed: int = SEED,
    epochs: int = EPOCHS,
):
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n{'='*65}")
    print(f"Mode: {mode} | Model: {model_name} | Seed: {seed} | Device: {device}")

    # Discover classes
    class_names = sorted(d.name for d in (PROC_DIR / "train").iterdir() if d.is_dir())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    n_cls = len(class_names)
    print(f"Classes ({n_cls}): {class_names}")

    augment  = (mode == "real_augmented")
    train_tf, val_tf = get_transforms(mode, augment=augment)
    train_ds, val_ds, test_ds = build_dataset(
        mode, gen_multiplier, denoise_tag, class_to_idx, train_tf, val_tf
    )

    pin_memory = torch.cuda.is_available()   # pin_memory only beneficial for CUDA
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)

    model     = build_model(model_name, n_cls).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    # GradScaler: CUDA only. MPS does not support float64 internally used by GradScaler,
    # so we use autocast without scaling on MPS (still gives speed/memory benefits).
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    # Experiment tag for saving
    exp_tag = f"{mode}_{model_name}_seed{seed}"
    if mode == "real_generated":
        exp_tag += f"_x{gen_multiplier}_{denoise_tag}"

    best_f1    = 0.
    best_epoch = 0
    history    = []

    print(f"Training {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device, class_names)
        scheduler.step()
        elapsed = time.time() - t0

        row = dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                   val_loss=val_loss, val_acc=val_acc, val_f1=val_f1)
        history.append(row)

        flag = ""
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            ckpt_path  = MODELS_DIR / f"{exp_tag}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "class_names": class_names,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
            }, ckpt_path)
            flag = "  ← best"

        print(f"  Ep {epoch:3d}/{epochs}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}  [{elapsed:.1f}s]{flag}")

    # ── Final test evaluation ──────────────────────────────────────
    print(f"\nLoading best model (epoch {best_epoch}, val_f1={best_f1:.4f})...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc, test_f1, preds, labels = evaluate(
        model, test_loader, criterion, device, class_names
    )
    report = classification_report(labels, preds, target_names=class_names,
                                   digits=4, output_dict=True)
    cm     = confusion_matrix(labels, preds).tolist()

    print(f"\nTest accuracy: {test_acc:.4f}  macro-F1: {test_f1:.4f}")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    # ── Save results ───────────────────────────────────────────────
    result = {
        "experiment": exp_tag,
        "mode":       mode,
        "model":      model_name,
        "seed":       seed,
        "gen_multiplier": gen_multiplier if mode == "real_generated" else None,
        "denoise_tag":    denoise_tag    if mode == "real_generated" else None,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
        "history": history,
    }
    out_dir = RESULTS_DIR / ("baseline" if "real_only" in mode or "augmented" in mode
                             else "augmented")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{exp_tag}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved → {result_path}")

    # Save baseline model separately for the filter step
    if mode == "real_only":
        baseline_path = MODELS_DIR / "baseline_cnn.pt"
        import shutil
        shutil.copy2(ckpt_path, baseline_path)
        print(f"Baseline checkpoint copied → {baseline_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",          type=str,
                        choices=["real_only", "real_augmented", "real_generated"],
                        default="real_only")
    parser.add_argument("--model",         type=str,
                        choices=["resnet50", "efficientnet_b0", "efficientnet_b2"],
                        default="efficientnet_b0")
    parser.add_argument("--gen_multiplier",type=int,   default=1,
                        help="1x, 2x, or 5x real dataset size for generated images")
    parser.add_argument("--denoise_tag",   type=str,   default="ds035",
                        help="Which denoise-strength generated folder to use: ds025|ds035|ds045")
    parser.add_argument("--seed",          type=int,   default=SEED)
    parser.add_argument("--epochs",        type=int,   default=EPOCHS)
    parser.add_argument("--run_all",       action="store_true",
                        help="Run real_only, real_augmented, real_generated in sequence")
    args = parser.parse_args()

    if args.run_all:
        for mode in ["real_only", "real_augmented", "real_generated"]:
            train(mode, args.model, args.gen_multiplier, args.denoise_tag,
                  args.seed, args.epochs)
    else:
        train(args.mode, args.model, args.gen_multiplier, args.denoise_tag,
              args.seed, args.epochs)


if __name__ == "__main__":
    main()
