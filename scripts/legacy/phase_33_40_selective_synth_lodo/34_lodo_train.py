"""
Script 34: Leave-One-Domain-Out training baseline.

Train on 3 source domains, validate on an in-domain split from those sources,
and test on the completely held-out target domain.

This is the first benchmark-reset step for the next research cycle:
  - harder than the current random val-split evaluation
  - directly measures unseen-domain generalization
  - supports low-data and hard-class-imbalance settings

Outputs:
  models/lodo_{model}_{heldout}.pt
  results/lodo_baseline/{model}/heldout_{domain}/report.json
  results/lodo_baseline/{model}/heldout_{domain}/report.md
  results/lodo_baseline/{model}/summary.json           (when --all_holdouts)
  results/lodo_baseline/{model}/summary.md             (when --all_holdouts)

Examples:
  python3 scripts/legacy/phase_33_40_selective_synth_lodo/34_lodo_train.py --heldout_domain domain_e_amc
  python3 scripts/legacy/phase_33_40_selective_synth_lodo/34_lodo_train.py --all_holdouts --model vgg16
  python3 scripts/legacy/phase_33_40_selective_synth_lodo/34_lodo_train.py --heldout_domain domain_c_mll23 --train_fraction 0.25
  python3 scripts/legacy/phase_33_40_selective_synth_lodo/34_lodo_train.py --heldout_domain domain_b_raabin \
      --class_fraction monocyte=0.25 --class_fraction eosinophil=0.25
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed_multidomain"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "lodo_baseline"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_LABELS = {
    "domain_a_pbc": "PBC (Spain)",
    "domain_b_raabin": "Raabin (Iran)",
    "domain_c_mll23": "MLL23 (Germany)",
    "domain_e_amc": "AMC (Korea)",
}
DOMAIN_SHORT = {
    "domain_a_pbc": "pbc",
    "domain_b_raabin": "raabin",
    "domain_c_mll23": "mll23",
    "domain_e_amc": "amc",
}
MULTI_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_IDX = {c: i for i, c in enumerate(MULTI_CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png"}
IMG_SIZE = 224
NUM_WORKERS = 0


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def collect_samples(data_dir: Path) -> list[tuple[Path, int, int]]:
    samples = []
    for d_idx, domain in enumerate(DOMAINS):
        for cls in MULTI_CLASSES:
            c_idx = CLASS_IDX[cls]
            cls_dir = data_dir / domain / cls
            if not cls_dir.exists():
                warnings.warn(f"[WARN] missing directory: {cls_dir}")
                continue
            paths = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            samples.extend((p, c_idx, d_idx) for p in paths)
    return samples


class SampleDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, c_idx, d_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, c_idx, d_idx


def print_combo_stats(samples: list[tuple[Path, int, int]], title: str):
    combo = Counter((c_idx, d_idx) for _, c_idx, d_idx in samples)
    total = len(samples)
    print(f"  {title}: {total} images")
    for d_idx, domain in enumerate(DOMAINS):
        if not any(x[1] == d_idx for x in combo):
            continue
        row = " | ".join(
            f"{MULTI_CLASSES[c_idx][:4]}={combo.get((c_idx, d_idx), 0)}"
            for c_idx in range(len(MULTI_CLASSES))
        )
        print(f"    {DOMAIN_LABELS[domain][:12]:12s}: {row}")


def stratified_fraction(
    samples: list[tuple[Path, int, int]],
    fraction: float,
    seed: int,
) -> list[tuple[Path, int, int]]:
    if fraction >= 1.0:
        return list(samples)
    if fraction <= 0.0:
        raise ValueError("fraction must be > 0")

    rng = random.Random(seed)
    buckets: dict[tuple[int, int], list[tuple[Path, int, int]]] = defaultdict(list)
    for item in samples:
        _, c_idx, d_idx = item
        buckets[(c_idx, d_idx)].append(item)

    selected = []
    for key, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        n_keep = max(1, int(round(len(items) * fraction)))
        n_keep = min(n_keep, len(items))
        selected.extend(items[:n_keep])
    return selected


def parse_class_fractions(items: list[str]) -> dict[int, float]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --class_fraction value: {item}")
        class_name, raw_fraction = item.split("=", 1)
        class_name = class_name.strip().lower()
        if class_name not in CLASS_IDX:
            raise ValueError(f"unknown class in --class_fraction: {class_name}")
        fraction = float(raw_fraction)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"class fraction must be in (0, 1], got {fraction}")
        result[CLASS_IDX[class_name]] = fraction
    return result


def apply_class_fractions(
    samples: list[tuple[Path, int, int]],
    class_fractions: dict[int, float],
    seed: int,
) -> list[tuple[Path, int, int]]:
    if not class_fractions:
        return list(samples)

    rng = random.Random(seed)
    buckets: dict[tuple[int, int], list[tuple[Path, int, int]]] = defaultdict(list)
    for item in samples:
        _, c_idx, d_idx = item
        buckets[(c_idx, d_idx)].append(item)

    selected = []
    for (c_idx, _d_idx), items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        fraction = class_fractions.get(c_idx, 1.0)
        n_keep = max(1, int(round(len(items) * fraction)))
        n_keep = min(n_keep, len(items))
        selected.extend(items[:n_keep])
    return selected


def stratified_train_val_split(
    samples: list[tuple[Path, int, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int, int]], list[tuple[Path, int, int]]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    rng = random.Random(seed)
    buckets: dict[tuple[int, int], list[tuple[Path, int, int]]] = defaultdict(list)
    for item in samples:
        _, c_idx, d_idx = item
        buckets[(c_idx, d_idx)].append(item)

    train_samples = []
    val_samples = []
    for key, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        if len(items) == 1:
            train_samples.extend(items)
            continue
        n_val = int(round(len(items) * val_ratio))
        n_val = max(1, min(n_val, len(items) - 1))
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    return train_samples, val_samples


def get_sample_weights(samples: list[tuple[Path, int, int]]) -> list[float]:
    combo_counts = Counter((c_idx, d_idx) for _, c_idx, d_idx in samples)
    return [1.0 / combo_counts[(c_idx, d_idx)] for _, c_idx, d_idx in samples]


def build_model(model_name: str, full_finetune: bool) -> nn.Module:
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(MULTI_CLASSES))
        return model

    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(MULTI_CLASSES))
        if not full_finetune:
            for param in model.features.parameters():
                param.requires_grad = False
        return model

    raise ValueError(f"unsupported model: {model_name}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0

    for imgs, labels, _domains in tqdm(loader, desc="  train", ncols=70, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == labels).sum().item()
        total += bs

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds = []
    all_labels = []

    for imgs, labels, _domains in tqdm(loader, desc="  eval ", ncols=70, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += bs
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "macro_f1": macro_f1,
        "preds": all_preds,
        "labels": all_labels,
    }


def build_class_metrics(labels: list[int], preds: list[int]) -> dict[str, dict]:
    report = classification_report(
        labels,
        preds,
        labels=list(range(len(MULTI_CLASSES))),
        target_names=MULTI_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    out = {}
    for class_name in MULTI_CLASSES:
        entry = report[class_name]
        out[class_name] = {
            "precision": round(entry["precision"], 4),
            "recall": round(entry["recall"], 4),
            "f1": round(entry["f1-score"], 4),
            "support": int(entry["support"]),
        }
    return out


def build_confusion_matrix(labels: list[int], preds: list[int]) -> list[list[int]]:
    matrix = [[0 for _ in MULTI_CLASSES] for _ in MULTI_CLASSES]
    for t, p in zip(labels, preds):
        matrix[t][p] += 1
    return matrix


def make_run_markdown(run: dict) -> str:
    class_rows = []
    for class_name in MULTI_CLASSES:
        item = run["test"]["per_class"][class_name]
        class_rows.append(
            f"| {class_name} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['support']} |"
        )

    cm = run["test"]["confusion_matrix"]
    cm_lines = [
        "| actual \\ pred | baso | eosi | lymp | mono | neut |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = ["baso", "eosi", "lymp", "mono", "neut"]
    for idx, row in enumerate(cm):
        cm_lines.append(
            f"| {labels[idx]} | " + " | ".join(str(v) for v in row) + " |"
        )

    return "\n".join([
        "# LODO Baseline Report",
        "",
        f"- Model: `{run['model_name']}`",
        f"- Held-out domain: `{run['heldout_domain']}` ({DOMAIN_LABELS[run['heldout_domain']]})",
        f"- Device: `{run['device']}`",
        f"- Train images: `{run['n_train']}`",
        f"- Val images: `{run['n_val']}`",
        f"- Test images: `{run['n_test']}`",
        f"- Train fraction: `{run['train_fraction']}`",
        f"- Class fractions: `{run['class_fraction_map']}`",
        "",
        "## Best Validation",
        "",
        f"- Best epoch: `{run['best_epoch']}`",
        f"- Val macro-F1: `{run['best_val_macro_f1']:.4f}`",
        f"- Val accuracy: `{run['best_val_acc']:.4f}`",
        "",
        "## Held-out Test",
        "",
        f"- Accuracy: `{run['test']['acc']:.4f}`",
        f"- Macro-F1: `{run['test']['macro_f1']:.4f}`",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
        *class_rows,
        "",
        "## Confusion Matrix",
        "",
        *cm_lines,
        "",
    ])


def make_summary_markdown(model_name: str, runs: list[dict]) -> str:
    rows = []
    macro_f1s = []
    accs = []
    for run in runs:
        macro_f1s.append(run["test"]["macro_f1"])
        accs.append(run["test"]["acc"])
        rows.append(
            f"| {DOMAIN_LABELS[run['heldout_domain']]} | {run['test']['acc']:.4f} | "
            f"{run['test']['macro_f1']:.4f} | {run['n_test']} |"
        )

    rows.append(
        f"| **Average** | **{np.mean(accs):.4f}** | **{np.mean(macro_f1s):.4f}** | - |"
    )
    return "\n".join([
        "# LODO Baseline Summary",
        "",
        f"- Model: `{model_name}`",
        f"- Runs: `{len(runs)}`",
        "",
        "| Held-out Domain | Accuracy | Macro-F1 | N |",
        "|---|---:|---:|---:|",
        *rows,
        "",
    ])


def run_single_holdout(args, heldout_domain: str) -> dict:
    device = get_device()
    set_seed(args.seed)

    heldout_idx = DOMAINS.index(heldout_domain)
    all_samples = collect_samples(DATA_DIR)
    source_samples = [s for s in all_samples if s[2] != heldout_idx]
    target_samples = [s for s in all_samples if s[2] == heldout_idx]

    source_samples = stratified_fraction(source_samples, args.train_fraction, args.seed)
    source_samples = apply_class_fractions(source_samples, args.class_fraction_map, args.seed)
    train_samples, val_samples = stratified_train_val_split(source_samples, args.val_ratio, args.seed)

    print(f"\n{'=' * 72}")
    print(f"Script 34 - LODO | model={args.model} | heldout={heldout_domain}")
    print(f"device={device} | epochs={args.epochs} | batch={args.batch_size}")
    if args.model == "vgg16":
        print(f"full_finetune={args.full_finetune}")
    print(f"train_fraction={args.train_fraction}")
    print(f"class_fraction_map={args.class_fraction_map}")
    print(f"{'=' * 72}")
    print_combo_stats(train_samples, "train")
    print_combo_stats(val_samples, "val")
    print_combo_stats(target_samples, "heldout test")

    train_ds = SampleDataset(train_samples, transform=get_train_transform())
    val_ds = SampleDataset(val_samples, transform=get_eval_transform())
    test_ds = SampleDataset(target_samples, transform=get_eval_transform())

    sampler = WeightedRandomSampler(
        get_sample_weights(train_samples),
        num_samples=len(train_samples),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = build_model(args.model, args.full_finetune).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: total={total_params:,}, trainable={trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1
    )

    run_dir = RESULTS_DIR / args.model / f"heldout_{DOMAIN_SHORT[heldout_domain]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"lodo_{args.model}_{DOMAIN_SHORT[heldout_domain]}.pt"

    if args.dry_run:
        print("[DRY RUN] initialization OK")
        dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
        with torch.no_grad():
            out = model(dummy)
        print(f"  forward OK: {tuple(out.shape)}")
        return {
            "heldout_domain": heldout_domain,
            "model_name": args.model,
            "device": str(device),
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "n_test": len(target_samples),
            "train_fraction": args.train_fraction,
            "class_fraction_map": {
                MULTI_CLASSES[c_idx]: frac for c_idx, frac in args.class_fraction_map.items()
            },
            "dry_run": True,
        }

    best_f1 = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = evaluate(model, val_loader, criterion, device)
        scheduler.step(epoch)

        history.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": {"loss": tr["loss"], "acc": tr["acc"]},
            "val": {"loss": vl["loss"], "acc": vl["acc"], "macro_f1": vl["macro_f1"]},
        })

        flag = ""
        if vl["macro_f1"] > best_f1:
            best_f1 = vl["macro_f1"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "full_finetune": args.full_finetune,
                "heldout_domain": heldout_domain,
                "class_names": MULTI_CLASSES,
                "model_state_dict": model.state_dict(),
                "val_f1": vl["macro_f1"],
                "val_acc": vl["acc"],
                "train_args": {
                    **vars(args),
                    "heldout_domain": heldout_domain,
                },
            }, ckpt_path)
            flag = "  <= best"

        print(
            f"  epoch {epoch:02d}/{args.epochs} | "
            f"val_loss={vl['loss']:.4f} | val_acc={vl['acc']*100:.1f}% | "
            f"val_f1={vl['macro_f1']:.4f}{flag}"
        )

    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_eval = evaluate(model, test_loader, criterion, device)
    per_class = build_class_metrics(test_eval["labels"], test_eval["preds"])
    confusion = build_confusion_matrix(test_eval["labels"], test_eval["preds"])

    run = {
        "heldout_domain": heldout_domain,
        "model_name": args.model,
        "device": str(device),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(target_samples),
        "train_fraction": args.train_fraction,
        "class_fraction_map": {
            MULTI_CLASSES[c_idx]: frac for c_idx, frac in args.class_fraction_map.items()
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_ckpt["val_f1"], 4),
        "best_val_acc": round(best_ckpt["val_acc"], 4),
        "history": history,
        "test": {
            "loss": round(test_eval["loss"], 4),
            "acc": round(test_eval["acc"], 4),
            "macro_f1": round(test_eval["macro_f1"], 4),
            "per_class": per_class,
            "confusion_matrix": confusion,
        },
    }

    with open(run_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
    (run_dir / "report.md").write_text(make_run_markdown(run), encoding="utf-8")

    print(
        f"  heldout result | acc={run['test']['acc']:.4f} | "
        f"macro_f1={run['test']['macro_f1']:.4f}"
    )
    print(f"  saved: {run_dir / 'report.json'}")

    return run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leave-One-Domain-Out baseline training"
    )
    parser.add_argument("--heldout_domain", choices=DOMAINS)
    parser.add_argument("--all_holdouts", action="store_true")
    parser.add_argument("--model", choices=["efficientnet_b0", "vgg16"], default="efficientnet_b0")
    parser.add_argument("--full_finetune", action="store_true",
                        help="only applies to vgg16")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--label_smooth", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="fraction of source-domain training pool to keep")
    parser.add_argument("--class_fraction", action="append", default=[],
                        help="repeatable: class=fraction, e.g. monocyte=0.25")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.all_holdouts == bool(args.heldout_domain):
        parser.error("choose exactly one of --heldout_domain or --all_holdouts")
    if not (0.0 < args.train_fraction <= 1.0):
        parser.error("--train_fraction must be in (0, 1]")
    if args.model != "vgg16" and args.full_finetune:
        parser.error("--full_finetune is only valid for --model vgg16")

    try:
        args.class_fraction_map = parse_class_fractions(args.class_fraction)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    holdouts = DOMAINS if args.all_holdouts else [args.heldout_domain]
    runs = [run_single_holdout(args, heldout_domain) for heldout_domain in holdouts]

    if args.dry_run or len(runs) == 1:
        return

    summary = {
        "model_name": args.model,
        "runs": runs,
        "avg_acc": round(float(np.mean([r["test"]["acc"] for r in runs])), 4),
        "avg_macro_f1": round(float(np.mean([r["test"]["macro_f1"] for r in runs])), 4),
    }
    model_dir = RESULTS_DIR / args.model
    with open(model_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    (model_dir / "summary.md").write_text(
        make_summary_markdown(args.model, runs),
        encoding="utf-8",
    )
    print(f"\nsummary saved: {model_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
