"""
Step 4: Quality gate — filter generated images before CNN training.

Two-stage filter:
  Stage 1 (class-identity): Run a pre-trained CNN classifier (baseline,
    Real-only) on each generated image. Keep only those where
    argmax(softmax) == target_class  AND  max(softmax) >= confidence_threshold.

  Stage 2 (sharpness/quality): Discard images with Laplacian variance
    below a floor percentile of real-train images.

Input:
    data/generated/{class}/{ds_tag}/*.png

Output:
    data/filtered/{class}/{ds_tag}/*.png

Requires:
    models/baseline_cnn.pt  (trained by 05_train_cnn.py with --mode real_only)
    → run Step 5 in real_only mode first, then come back to filter.

Usage:
    python 04_filter_generated.py --classifier_ckpt models/baseline_cnn.pt
    python 04_filter_generated.py --classifier_ckpt models/baseline_cnn.pt \
        --conf_threshold 0.7 --sharp_floor_pctile 20
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
GEN_DIR      = ROOT / "data" / "generated"
FILTERED_DIR = ROOT / "data" / "filtered"
TRAIN_DIR    = ROOT / "data" / "processed" / "train"
FILTERED_DIR.mkdir(parents=True, exist_ok=True)

# ── Defaults ───────────────────────────────────────────────────────
CONF_THRESHOLD      = 0.70
SHARP_FLOOR_PCTILE  = 20     # drop below this percentile of real-train sharpness
IMG_SIZE            = 224    # must match CNN training size


def laplacian_variance(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_real_sharp_floor(class_name: str, pctile: int) -> float:
    """Return Laplacian-variance floor from real train images of this class."""
    real_dir = TRAIN_DIR / class_name
    scores = []
    for p in real_dir.rglob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(p))
        if img is not None:
            scores.append(laplacian_variance(img))
    if not scores:
        return 0.0
    return float(np.percentile(scores, pctile))


def load_classifier(ckpt_path: Path, class_names: list[str], device: torch.device):
    n_cls = len(class_names)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, n_cls)
    ckpt = torch.load(ckpt_path, map_location=device)
    # Support both raw state_dict and checkpoint dicts
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def filter_class(
    class_name: str,
    class_idx: int,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    conf_threshold: float,
    sharp_floor: float,
):
    from PIL import Image

    class_gen_dir = GEN_DIR / class_name
    if not class_gen_dir.exists():
        print(f"  [SKIP] no generated images for {class_name}")
        return {}

    stats = {}
    for ds_dir in sorted(class_gen_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        out_dir = FILTERED_DIR / class_name / ds_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = [p for p in ds_dir.glob("*.png")]
        kept = 0
        fail_cls = 0
        fail_sharp = 0

        for p in tqdm(imgs, desc=f"  {class_name}/{ds_dir.name}"):
            # Sharpness gate
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            if laplacian_variance(img_bgr) < sharp_floor:
                fail_sharp += 1
                continue

            # Classifier gate
            pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            tensor = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1)[0]
            pred_cls  = probs.argmax().item()
            confidence = probs[class_idx].item()

            if pred_cls != class_idx or confidence < conf_threshold:
                fail_cls += 1
                continue

            shutil.copy2(p, out_dir / p.name)
            kept += 1

        total = len(imgs)
        print(f"    [{class_name}/{ds_dir.name}] "
              f"kept={kept}/{total}  "
              f"(fail_cls={fail_cls}, fail_sharp={fail_sharp})")
        stats[ds_dir.name] = dict(total=total, kept=kept,
                                  fail_cls=fail_cls, fail_sharp=fail_sharp)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_ckpt", type=str,
                        default=str(ROOT / "models" / "baseline_cnn.pt"),
                        help="Path to baseline CNN checkpoint (Real-only trained)")
    parser.add_argument("--conf_threshold",     type=float, default=CONF_THRESHOLD)
    parser.add_argument("--sharp_floor_pctile", type=int,   default=SHARP_FLOOR_PCTILE)
    parser.add_argument("--class_name",         type=str,   default=None,
                        help="Filter single class only")
    args = parser.parse_args()

    ckpt_path = Path(args.classifier_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {ckpt_path}\n"
            "Run Step 5 (05_train_cnn.py --mode real_only) first."
        )

    # Discover classes from train dir
    class_names = sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"Classes ({len(class_names)}): {class_names}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model    = load_classifier(ckpt_path, class_names, device)
    transform = build_transform()

    classes_to_run = [args.class_name] if args.class_name else class_names

    all_stats = {}
    for cls in classes_to_run:
        print(f"\n{'='*55}\nFiltering: {cls}")
        sharp_floor = compute_real_sharp_floor(cls, args.sharp_floor_pctile)
        print(f"  Sharpness floor (Lap.var p{args.sharp_floor_pctile}): {sharp_floor:.1f}")
        all_stats[cls] = filter_class(
            cls,
            class_to_idx[cls],
            model, transform, device,
            args.conf_threshold,
            sharp_floor,
        )

    # Save report
    report_path = ROOT / "results" / "filter_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "conf_threshold":     args.conf_threshold,
            "sharp_floor_pctile": args.sharp_floor_pctile,
            "classes":            all_stats,
        }, f, indent=2)
    print(f"\nFilter report saved → {report_path}")


if __name__ == "__main__":
    main()
