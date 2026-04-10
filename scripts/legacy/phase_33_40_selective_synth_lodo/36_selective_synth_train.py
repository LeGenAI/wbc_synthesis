"""
Script 36: LODO training with selective synthetic subsets.

Train on 3 source domains of real data plus a chosen synthetic subset manifest,
validate on a real-only split from the source domains, and test on the held-out
target domain.

Key rule:
  - synthetic items from the held-out target domain are excluded by default
    to avoid target-domain leakage in LODO evaluation

Outputs:
  models/selective_{model}_{subset}_{heldout}.pt
  results/selective_synth/{model}/heldout_{domain}/{subset}/report.json
  results/selective_synth/{model}/heldout_{domain}/{subset}/report.md

Examples:
  python3 scripts/legacy/phase_33_40_selective_synth_lodo/36_selective_synth_train.py \
      --heldout_domain domain_e_amc \
      --subset_manifest results/selective_synth/subsets/S2_cnn_correct.json

  python3 scripts/legacy/phase_33_40_selective_synth_lodo/36_selective_synth_train.py \
      --heldout_domain domain_c_mll23 \
      --subset_manifest results/selective_synth/subsets/S7_hard_classes_only.json \
      --model vgg16 --full_finetune
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
RESULTS_DIR = ROOT / "results" / "selective_synth"
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
DOMAIN_IDX = {d: i for i, d in enumerate(DOMAINS)}
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


def collect_real_samples(data_dir: Path) -> list[tuple[str, str, str]]:
    samples = []
    for domain in DOMAINS:
        for cls in MULTI_CLASSES:
            cls_dir = data_dir / domain / cls
            if not cls_dir.exists():
                warnings.warn(f"[WARN] missing directory: {cls_dir}")
                continue
            paths = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            for path in paths:
                samples.append((str(path.resolve()), cls, domain))
    return samples


def stratified_fraction(samples: list[tuple[str, str, str]], fraction: float, seed: int):
    if fraction >= 1.0:
        return list(samples)
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for item in samples:
        _path, cls, domain = item
        buckets[(cls, domain)].append(item)
    selected = []
    for key, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        n_keep = max(1, int(round(len(items) * fraction)))
        n_keep = min(n_keep, len(items))
        selected.extend(items[:n_keep])
    return selected


def parse_class_fractions(items: list[str]) -> dict[str, float]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --class_fraction value: {item}")
        class_name, raw_fraction = item.split("=", 1)
        class_name = class_name.strip().lower()
        if class_name not in MULTI_CLASSES:
            raise ValueError(f"unknown class in --class_fraction: {class_name}")
        fraction = float(raw_fraction)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"class fraction must be in (0, 1], got {fraction}")
        result[class_name] = fraction
    return result


def apply_class_fractions(samples: list[tuple[str, str, str]], class_fractions: dict[str, float], seed: int):
    if not class_fractions:
        return list(samples)
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for item in samples:
        _path, cls, domain = item
        buckets[(cls, domain)].append(item)
    selected = []
    for (cls, _domain), items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        fraction = class_fractions.get(cls, 1.0)
        n_keep = max(1, int(round(len(items) * fraction)))
        n_keep = min(n_keep, len(items))
        selected.extend(items[:n_keep])
    return selected


def stratified_train_val_split(samples: list[tuple[str, str, str]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for item in samples:
        _path, cls, domain = item
        buckets[(cls, domain)].append(item)
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


def load_synth_manifest(path: Path) -> tuple[dict, list[tuple[str, str, str]]]:
    data = json.loads(path.read_text())
    items = []
    for item in data["items"]:
        items.append((item["file_abs"], item["class_name"], item["domain"]))
    return data, items


class MixedDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str, str]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_name, domain = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, CLASS_IDX[cls_name], DOMAIN_IDX[domain]


def print_combo_stats(samples: list[tuple[str, str, str]], title: str):
    combo = Counter((cls, domain) for _path, cls, domain in samples)
    print(f"  {title}: {len(samples)} images")
    for domain in DOMAINS:
        if not any(d == domain for _, d in combo):
            continue
        row = " | ".join(
            f"{cls[:4]}={combo.get((cls, domain), 0)}"
            for cls in MULTI_CLASSES
        )
        print(f"    {DOMAIN_LABELS[domain][:12]:12s}: {row}")


def get_sample_weights(samples: list[tuple[str, str, str]]) -> list[float]:
    combo_counts = Counter((cls, domain) for _path, cls, domain in samples)
    return [1.0 / combo_counts[(cls, domain)] for _path, cls, domain in samples]


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
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
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


def make_report_md(run: dict) -> str:
    rows = []
    for class_name in MULTI_CLASSES:
        item = run["test"]["per_class"][class_name]
        rows.append(
            f"| {class_name} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['support']} |"
        )

    return "\n".join([
        "# Selective Synthetic LODO Report",
        "",
        f"- Model: `{run['model_name']}`",
        f"- Held-out domain: `{run['heldout_domain']}` ({DOMAIN_LABELS[run['heldout_domain']]})",
        f"- Subset: `{run['subset_id']}` / `{run['subset_name']}`",
        f"- Subset manifest: `{run['subset_manifest']}`",
        f"- Exclude held-out synth: `{run['exclude_heldout_synth']}`",
        f"- Device: `{run['device']}`",
        f"- Real train images: `{run['n_real_train']}`",
        f"- Synthetic train images: `{run['n_synth_train']}`",
        f"- Combined train images: `{run['n_train_total']}`",
        f"- Val images: `{run['n_val']}`",
        f"- Test images: `{run['n_test']}`",
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
        *rows,
        "",
    ])


def parse_args():
    parser = argparse.ArgumentParser(
        description="LODO training with selective synthetic subset manifests"
    )
    parser.add_argument("--heldout_domain", choices=DOMAINS, required=True)
    parser.add_argument("--subset_manifest", required=True)
    parser.add_argument("--model", choices=["efficientnet_b0", "vgg16"], default="efficientnet_b0")
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--label_smooth", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--class_fraction", action="append", default=[])
    parser.add_argument("--include_heldout_synth", action="store_true",
                        help="allow synthetic items whose source domain matches held-out domain")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.model != "vgg16" and args.full_finetune:
        parser.error("--full_finetune is only valid for --model vgg16")
    if not (0.0 < args.train_fraction <= 1.0):
        parser.error("--train_fraction must be in (0, 1]")
    try:
        args.class_fraction_map = parse_class_fractions(args.class_fraction)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    device = get_device()
    set_seed(args.seed)

    heldout_domain = args.heldout_domain
    subset_manifest_path = Path(args.subset_manifest)
    if not subset_manifest_path.is_absolute():
        subset_manifest_path = (ROOT / subset_manifest_path).resolve()

    real_samples = collect_real_samples(DATA_DIR)
    source_real = [s for s in real_samples if s[2] != heldout_domain]
    target_test = [s for s in real_samples if s[2] == heldout_domain]

    source_real = stratified_fraction(source_real, args.train_fraction, args.seed)
    source_real = apply_class_fractions(source_real, args.class_fraction_map, args.seed)
    train_real, val_real = stratified_train_val_split(source_real, args.val_ratio, args.seed)

    subset_info, synth_items = load_synth_manifest(subset_manifest_path)
    if not args.include_heldout_synth:
        synth_items = [s for s in synth_items if s[2] != heldout_domain]

    print(f"\n{'=' * 76}")
    print(f"Script 36 - Selective Synth LODO | model={args.model} | heldout={heldout_domain}")
    print(f"subset={subset_info['subset_id']} ({subset_info['name']})")
    print(f"device={device} | epochs={args.epochs} | batch={args.batch_size}")
    print(f"train_fraction={args.train_fraction} | class_fraction_map={args.class_fraction_map}")
    print(f"include_heldout_synth={args.include_heldout_synth}")
    print(f"{'=' * 76}")
    print_combo_stats(train_real, "real train")
    print_combo_stats(val_real, "real val")
    print_combo_stats(target_test, "heldout test")
    print_combo_stats(synth_items, "synthetic train subset")

    combined_train = train_real + synth_items

    train_ds = MixedDataset(combined_train, transform=get_train_transform())
    val_ds = MixedDataset(val_real, transform=get_eval_transform())
    test_ds = MixedDataset(target_test, transform=get_eval_transform())

    sampler = WeightedRandomSampler(
        get_sample_weights(combined_train),
        num_samples=len(combined_train),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    model = build_model(args.model, args.full_finetune).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: total={total_params:,}, trainable={trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    subset_slug = subset_info["subset_id"].lower()
    ckpt_path = MODELS_DIR / f"selective_{args.model}_{subset_slug}_{DOMAIN_SHORT[heldout_domain]}.pt"
    run_dir = RESULTS_DIR / args.model / f"heldout_{DOMAIN_SHORT[heldout_domain]}" / subset_info["subset_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[DRY RUN] initialization OK")
        dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
        with torch.no_grad():
            out = model(dummy)
        print(f"  forward OK: {tuple(out.shape)}")
        return

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
                "subset_id": subset_info["subset_id"],
                "subset_name": subset_info["name"],
                "subset_manifest": str(subset_manifest_path),
                "class_names": MULTI_CLASSES,
                "model_state_dict": model.state_dict(),
                "val_f1": vl["macro_f1"],
                "val_acc": vl["acc"],
                "train_args": vars(args),
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

    run = {
        "heldout_domain": heldout_domain,
        "model_name": args.model,
        "subset_id": subset_info["subset_id"],
        "subset_name": subset_info["name"],
        "subset_manifest": str(subset_manifest_path),
        "exclude_heldout_synth": not args.include_heldout_synth,
        "device": str(device),
        "n_real_train": len(train_real),
        "n_synth_train": len(synth_items),
        "n_train_total": len(combined_train),
        "n_val": len(val_real),
        "n_test": len(target_test),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_ckpt["val_f1"], 4),
        "best_val_acc": round(best_ckpt["val_acc"], 4),
        "history": history,
        "test": {
            "loss": round(test_eval["loss"], 4),
            "acc": round(test_eval["acc"], 4),
            "macro_f1": round(test_eval["macro_f1"], 4),
            "per_class": build_class_metrics(test_eval["labels"], test_eval["preds"]),
            "confusion_matrix": build_confusion_matrix(test_eval["labels"], test_eval["preds"]),
        },
    }

    with open(run_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
    (run_dir / "report.md").write_text(make_report_md(run), encoding="utf-8")

    print(
        f"  heldout result | acc={run['test']['acc']:.4f} | "
        f"macro_f1={run['test']['macro_f1']:.4f}"
    )
    print(f"  saved: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
