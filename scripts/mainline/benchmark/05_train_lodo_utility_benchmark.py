#!/usr/bin/env python3
"""
Stage 05: leakage-safe LODO utility benchmark.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mainline.common.config import apply_overrides, dump_yaml_config, load_yaml_config
from scripts.mainline.common.constants import CLASS_TO_IDX, CLASSES, DOMAIN_SHORT, DOMAINS, HARD_CLASSES
from scripts.mainline.common.manifests import load_manifest_items
from scripts.mainline.common.reporting import markdown_table, write_json, write_text
from scripts.mainline.common.runtime import (
    build_backbone,
    build_lr_scheduler,
    ensure_dir,
    get_device,
    resolve_project_path,
    set_seed,
)
from scripts.mainline.common.split import stratified_class_fractions, stratified_fraction

EVAL_TTA_MODES = {"none", "hflip", "fivecrop", "hflip_fivecrop"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ManifestDataset(Dataset):
    def __init__(self, items: list[dict], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = Image.open(item["file_abs"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, CLASS_TO_IDX[item["class_name"]]


class EvalTransform:
    def __init__(self, image_size: int, eval_tta_mode: str):
        self.image_size = image_size
        self.eval_tta_mode = eval_tta_mode
        self.resize_size = max(image_size + 32, 256)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.eval_tta_mode == "none":
            resized = TF.resize(image, [self.image_size, self.image_size], antialias=True)
            return normalize_eval_image(resized)
        if self.eval_tta_mode == "hflip":
            resized = TF.resize(image, [self.image_size, self.image_size], antialias=True)
            views = [resized, TF.hflip(resized)]
        elif self.eval_tta_mode == "fivecrop":
            resized = TF.resize(image, [self.resize_size, self.resize_size], antialias=True)
            views = list(TF.five_crop(resized, [self.image_size, self.image_size]))
        elif self.eval_tta_mode == "hflip_fivecrop":
            resized = TF.resize(image, [self.resize_size, self.resize_size], antialias=True)
            views = list(TF.five_crop(resized, [self.image_size, self.image_size]))
            views.extend(TF.five_crop(TF.hflip(resized), [self.image_size, self.image_size]))
        else:
            raise ValueError(f"Unsupported eval_tta_mode: {self.eval_tta_mode}")
        return torch.stack([normalize_eval_image(view) for view in views], dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the mainline leakage-safe LODO utility benchmark.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mainline/benchmark/real_only.yaml",
        help="Path to the benchmark YAML config.",
    )
    parser.add_argument("--heldout-domain", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--synthetic-manifest", type=str, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--all-heldouts", action="store_true")
    parser.add_argument("--eval-tta-mode", type=str, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run benchmark with multiple seeds for mean/std reporting.",
    )
    parser.add_argument(
        "--class-fraction",
        action="append",
        default=None,
        help="Optional per-class train fraction override, e.g. monocyte=0.25",
    )
    return parser.parse_args()


def parse_cli_class_fractions(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    parsed: dict[str, float] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"class-fraction must use CLASS=FRACTION, got: {raw}")
        class_name, fraction_text = raw.split("=", 1)
        class_name = class_name.strip()
        if class_name not in CLASSES:
            raise ValueError(f"Unknown class in class-fraction: {class_name}")
        parsed[class_name] = float(fraction_text)
    return parsed


def validate_config(config: dict) -> dict:
    if config["heldout_domain"] not in DOMAINS:
        raise ValueError(f"Unknown heldout domain: {config['heldout_domain']}")
    if config["backbone"] not in {"efficientnet_b0", "vgg16"}:
        raise ValueError(f"Unsupported backbone: {config['backbone']}")
    if config["mode"] not in {"real_only", "real_plus_synth"}:
        raise ValueError(f"Unsupported mode: {config['mode']}")
    eval_tta_mode = str(config.get("eval_tta_mode", "none"))
    if eval_tta_mode not in EVAL_TTA_MODES:
        raise ValueError(
            f"Unsupported eval_tta_mode: {eval_tta_mode}. "
            f"Expected one of {sorted(EVAL_TTA_MODES)}"
        )
    if not (0.0 < float(config["train_fraction"]) <= 1.0):
        raise ValueError(f"train_fraction must be in (0, 1], got {config['train_fraction']}")
    if config["mode"] == "real_plus_synth" and not config.get("synthetic_manifest"):
        raise ValueError("synthetic_manifest is required in real_plus_synth mode")
    raw_class_fractions = config.get("class_fractions") or {}
    if not isinstance(raw_class_fractions, dict):
        raise ValueError("class_fractions must be a mapping of class_name -> fraction")
    class_fractions = {}
    for class_name, fraction in raw_class_fractions.items():
        if class_name not in CLASSES:
            raise ValueError(f"Unknown class in class_fractions: {class_name}")
        fraction = float(fraction)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"class fraction for {class_name} must be in (0, 1], got {fraction}")
        class_fractions[class_name] = fraction
    return {
        **config,
        "epochs": int(config["epochs"]),
        "batch_size": int(config["batch_size"]),
        "seed": int(config["seed"]),
        "train_fraction": float(config["train_fraction"]),
        "lr": float(config["lr"]),
        "weight_decay": float(config["weight_decay"]),
        "num_workers": int(config.get("num_workers", 0)),
        "image_size": int(config.get("image_size", 224)),
        "full_finetune": bool(config.get("full_finetune", False)),
        "synthetic_sampling_weight": float(config.get("synthetic_sampling_weight", 1.0)),
        "leakage_filter": bool(config.get("leakage_filter", True)),
        "lr_step_size": int(config.get("lr_step_size", 10)),
        "lr_gamma": float(config.get("lr_gamma", 0.5)),
        "class_fractions": class_fractions,
        "all_heldouts": bool(config.get("all_heldouts", False)),
        "eval_tta_mode": eval_tta_mode,
    }


def get_train_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.1),
        ]
    )


def normalize_eval_image(image: Image.Image) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)


def get_eval_transform(image_size: int, eval_tta_mode: str):
    return EvalTransform(image_size=image_size, eval_tta_mode=eval_tta_mode)


def load_split_manifests(manifest_root: Path, heldout_domain: str) -> tuple[list[dict], list[dict], list[dict]]:
    split_root = manifest_root / f"heldout_{heldout_domain}"
    if not split_root.exists():
        raise FileNotFoundError(
            f"Missing manifest directory for heldout domain {heldout_domain}: {split_root}"
        )
    train_items = load_manifest_items(split_root / "train_manifest.json")
    val_items = load_manifest_items(split_root / "val_manifest.json")
    test_items = load_manifest_items(split_root / "test_manifest.json")
    return train_items, val_items, test_items


def validate_synth_items(items: list[dict]) -> list[dict]:
    required = {"file_abs", "class_name", "domain", "source_type", "policy_id"}
    validated = []
    for idx, item in enumerate(items):
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"Synthetic manifest item {idx} missing fields: {missing}")
        if item["source_type"] != "synthetic":
            raise ValueError(f"Synthetic manifest item {idx} must use source_type=synthetic")
        if item["class_name"] not in CLASSES:
            raise ValueError(f"Unknown synthetic class at item {idx}: {item['class_name']}")
        if item["domain"] not in DOMAINS:
            raise ValueError(f"Unknown synthetic domain at item {idx}: {item['domain']}")
        path = Path(item["file_abs"])
        if not path.exists():
            raise FileNotFoundError(f"Synthetic file missing at item {idx}: {path}")
        validated.append(copy.deepcopy(item))
    return validated


def apply_leakage_filter(items: list[dict], heldout_domain: str) -> tuple[list[dict], dict]:
    kept = []
    excluded = []
    for item in items:
        if item["domain"] == heldout_domain:
            excluded.append(item)
            continue
        clone = copy.deepcopy(item)
        clone["split"] = "train"
        kept.append(clone)
    return kept, {
        "excluded_for_heldout_domain": len(excluded),
        "excluded_domain": heldout_domain,
    }


def get_sample_weights(items: list[dict], synthetic_sampling_weight: float) -> list[float]:
    counter = Counter(
        (item["class_name"], item["domain"], item.get("source_type", "real")) for item in items
    )
    weights = []
    for item in items:
        base = 1.0 / counter[(item["class_name"], item["domain"], item.get("source_type", "real"))]
        if item.get("source_type") == "synthetic":
            base *= synthetic_sampling_weight
        weights.append(base)
    return weights


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for images, targets in tqdm(dataloader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
    return running_loss / max(total, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="eval", leave=False):
            targets = targets.to(device)
            if images.ndim == 5:
                batch_size, n_views, channels, height, width = images.shape
                logits = model(images.reshape(batch_size * n_views, channels, height, width).to(device))
                logits = logits.reshape(batch_size, n_views, len(CLASSES)).mean(dim=1)
            else:
                logits = model(images.to(device))
            loss = criterion(logits, targets)
            losses.append(loss.item() * targets.size(0))
            predictions = logits.argmax(dim=1)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        labels=list(range(len(CLASSES))),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        class_name: {
            "precision": round(float(report[class_name]["precision"]), 4),
            "recall": round(float(report[class_name]["recall"]), 4),
            "f1": round(float(report[class_name]["f1-score"]), 4),
            "support": int(report[class_name]["support"]),
        }
        for class_name in CLASSES
    }

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "loss": round(float(sum(losses) / max(len(y_true), 1)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist(),
    }


def count_samples(items: list[dict]) -> dict[str, object]:
    by_combo = Counter((item["domain"], item["class_name"], item.get("source_type", "real")) for item in items)
    rows = []
    for (domain, class_name, source_type), count in sorted(by_combo.items()):
        rows.append(
            {
                "domain": domain,
                "class_name": class_name,
                "source_type": source_type,
                "count": count,
            }
        )
    return {"total": len(items), "rows": rows}


def render_report_markdown(report: dict) -> str:
    train_rows = [
        [row["domain"], row["class_name"], row["source_type"], row["count"]]
        for row in report["split_stats"]["train"]["rows"]
    ]
    val_rows = [
        [row["domain"], row["class_name"], row["source_type"], row["count"]]
        for row in report["split_stats"]["val"]["rows"]
    ]
    test_rows = [
        [row["domain"], row["class_name"], row["source_type"], row["count"]]
        for row in report["split_stats"]["test"]["rows"]
    ]
    metric_rows = [
        ["val", report["val"]["accuracy"], report["val"]["macro_f1"], report["val"]["loss"]],
        ["test", report["test"]["accuracy"], report["test"]["macro_f1"], report["test"]["loss"]],
    ]
    class_rows = [
        [
            class_name,
            report["test"]["per_class"][class_name]["precision"],
            report["test"]["per_class"][class_name]["recall"],
            report["test"]["per_class"][class_name]["f1"],
            report["test"]["per_class"][class_name]["support"],
        ]
        for class_name in CLASSES
    ]
    lines = [
        "# Mainline LODO Utility Benchmark Report",
        "",
        f"- Run name: `{report['run_name']}`",
        f"- Mode: `{report['mode']}`",
        f"- Backbone: `{report['backbone']}`",
        f"- Held-out domain: `{report['heldout_domain']}`",
        f"- Device: `{report['device']}`",
        f"- Eval TTA mode: `{report['eval_tta_mode']}`",
        "",
        "## Hard-class Setting",
        "",
        f"- hard_classes: `{report['hard_classes']}`",
        f"- class_fractions: `{report['config'].get('class_fractions', {})}`",
        "",
        "## Metrics",
        "",
        markdown_table(["Split", "Accuracy", "Macro-F1", "Loss"], metric_rows),
        "",
        "## Leakage Guard",
        "",
        f"- excluded_for_heldout_domain: `{report['synthetic_guard']['excluded_for_heldout_domain']}`",
        f"- synthetic_train_items_used: `{report['synthetic_guard']['synthetic_train_items_used']}`",
        "",
        "## Split Stats",
        "",
        "### Train",
        "",
        markdown_table(["Domain", "Class", "Source", "Count"], train_rows),
        "",
        "### Val",
        "",
        markdown_table(["Domain", "Class", "Source", "Count"], val_rows),
        "",
        "### Test",
        "",
        markdown_table(["Domain", "Class", "Source", "Count"], test_rows),
        "",
        "## Test Per-class",
        "",
        markdown_table(["Class", "Precision", "Recall", "F1", "Support"], class_rows),
        "",
        f"- Confusion matrix image: `{report['artifacts']['confusion_matrix_png']}`",
        "",
    ]
    return "\n".join(lines)


def plot_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    arr = np.array(matrix)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def build_run_name(config: dict) -> str:
    train_fraction_tag = str(config["train_fraction"]).replace(".", "p")
    domain_short = DOMAIN_SHORT[config["heldout_domain"]]
    parts = [
        config["mode"],
        config["backbone"],
        f"heldout_{domain_short}",
        f"tf{train_fraction_tag}",
        f"seed{config['seed']}",
    ]
    if config["mode"] == "real_plus_synth" and config.get("synthetic_manifest"):
        manifest_stem = Path(str(config["synthetic_manifest"])).stem
        manifest_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", manifest_stem)
        parts.append(f"synth_{manifest_stem}")
    if config.get("class_fractions"):
        fraction_tag = "_".join(
            f"{class_name[:4]}{str(value).replace('.', 'p')}"
            for class_name, value in sorted(config["class_fractions"].items())
        )
        parts.append(f"cf_{fraction_tag}")
    if config.get("eval_tta_mode", "none") != "none":
        parts.append(f"tta_{config['eval_tta_mode']}")
    return "__".join(parts)


def update_summary(heldout_root: Path) -> None:
    run_reports = sorted(heldout_root.glob("*/report.json"))
    rows = []
    for report_path in run_reports:
        with open(report_path, "r", encoding="utf-8") as handle:
            import json

            report = json.load(handle)
        rows.append(
            {
                "run_name": report["run_name"],
                "mode": report["mode"],
                "backbone": report["backbone"],
                "test_macro_f1": report["test"]["macro_f1"],
                "test_accuracy": report["test"]["accuracy"],
                "excluded_for_heldout_domain": report["synthetic_guard"]["excluded_for_heldout_domain"],
                "eval_tta_mode": report.get("eval_tta_mode", report.get("config", {}).get("eval_tta_mode", "none")),
            }
        )
    write_json(heldout_root / "summary.json", {"runs": rows})
    markdown_rows = [
        [
            row["run_name"],
            row["mode"],
            row["backbone"],
            row["test_accuracy"],
            row["test_macro_f1"],
            row["excluded_for_heldout_domain"],
            row["eval_tta_mode"],
        ]
        for row in rows
    ]
    write_text(
        heldout_root / "summary.md",
        "\n".join(
            [
                "# Mainline Benchmark Summary",
                "",
                markdown_table(
                    ["Run", "Mode", "Backbone", "Test Acc", "Test Macro-F1", "Leakage Excluded", "Eval TTA"],
                    markdown_rows,
                ),
                "",
            ]
        ),
    )


def run_single_seed(config: dict) -> dict:
    """Run one full train/eval cycle for a single seed. Returns the report dict."""
    set_seed(config["seed"])

    manifest_root = resolve_project_path(PROJECT_ROOT, config["manifest_root"])
    output_root = ensure_dir(resolve_project_path(PROJECT_ROOT, config["output_root"]))
    heldout_root = ensure_dir(output_root / config["backbone"] / f"heldout_{config['heldout_domain']}")
    run_name = build_run_name(config)
    run_root = ensure_dir(heldout_root / run_name)

    train_items, val_items, test_items = load_split_manifests(manifest_root, config["heldout_domain"])
    train_items = stratified_fraction(train_items, config["train_fraction"], config["seed"])
    train_items = stratified_class_fractions(train_items, config["class_fractions"], config["seed"])

    leakage_stats = {
        "excluded_for_heldout_domain": 0,
        "synthetic_train_items_used": 0,
        "synthetic_manifest": None,
    }
    if config["mode"] == "real_plus_synth":
        synth_manifest_path = resolve_project_path(PROJECT_ROOT, config["synthetic_manifest"])
        synth_items = validate_synth_items(load_manifest_items(synth_manifest_path))
        synth_items, exclusion_stats = apply_leakage_filter(synth_items, config["heldout_domain"])
        train_items = list(train_items) + synth_items
        leakage_stats.update(exclusion_stats)
        leakage_stats["synthetic_train_items_used"] = len(synth_items)
        leakage_stats["synthetic_manifest"] = str(synth_manifest_path)

    device = get_device()
    model = build_backbone(
        model_name=config["backbone"],
        num_classes=len(CLASSES),
        full_finetune=config["full_finetune"],
    ).to(device)

    train_loader = DataLoader(
        ManifestDataset(train_items, transform=get_train_transform(config["image_size"])),
        batch_size=config["batch_size"],
        sampler=WeightedRandomSampler(
            get_sample_weights(train_items, config["synthetic_sampling_weight"]),
            num_samples=len(train_items),
            replacement=True,
        ),
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        ManifestDataset(val_items, transform=get_eval_transform(config["image_size"], config["eval_tta_mode"])),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        ManifestDataset(test_items, transform=get_eval_transform(config["image_size"], config["eval_tta_mode"])),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    criterion = nn.CrossEntropyLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = build_lr_scheduler(
        optimizer,
        step_size=config["lr_step_size"],
        gamma=config["lr_gamma"],
    )

    best_state = None
    best_val_macro_f1 = float("-inf")
    history = []
    for epoch in range(1, config["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 4),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "lr": round(float(scheduler.get_last_lr()[0]), 6),
            }
        )
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a best validation checkpoint.")

    model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    best_model_path = run_root / "best_model.pt"
    torch.save(best_state, best_model_path)

    confusion_matrix_path = run_root / "confusion_matrix.png"
    plot_confusion_matrix(confusion_matrix_path, test_metrics["confusion_matrix"])

    report = {
        "run_name": run_name,
        "mode": config["mode"],
        "backbone": config["backbone"],
        "heldout_domain": config["heldout_domain"],
        "device": str(device),
        "eval_tta_mode": config["eval_tta_mode"],
        "hard_classes": HARD_CLASSES,
        "config": config,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
        "split_stats": {
            "train": count_samples(train_items),
            "val": count_samples(val_items),
            "test": count_samples(test_items),
        },
        "synthetic_guard": leakage_stats,
        "artifacts": {
            "best_model_pt": str(best_model_path.resolve()),
            "confusion_matrix_png": str(confusion_matrix_path.resolve()),
        },
    }

    write_json(run_root / "report.json", report)
    write_text(run_root / "report.md", render_report_markdown(report))
    dump_yaml_config(run_root / "resolved_config.yaml", config)
    update_summary(heldout_root)
    print(f"Wrote benchmark outputs to: {run_root}")
    return report


def write_all_heldouts_summary(reports_by_domain: dict[str, list[dict]], output_root: Path, backbone: str) -> None:
    rows = []
    for heldout_domain, reports in sorted(reports_by_domain.items()):
        test_f1s = [report["test"]["macro_f1"] for report in reports]
        test_accs = [report["test"]["accuracy"] for report in reports]
        rows.append(
            {
                "heldout_domain": heldout_domain,
                "n_runs": len(reports),
                "seeds": [report["config"]["seed"] for report in reports],
                "eval_tta_mode": reports[0]["eval_tta_mode"] if reports else "none",
                "test_macro_f1_mean": round(float(np.mean(test_f1s)), 4),
                "test_macro_f1_std": round(float(np.std(test_f1s)), 4),
                "test_accuracy_mean": round(float(np.mean(test_accs)), 4),
                "test_accuracy_std": round(float(np.std(test_accs)), 4),
            }
        )

    summary_root = ensure_dir(output_root / backbone)
    write_json(summary_root / "all_heldouts_summary.json", {"runs": rows})
    write_text(
        summary_root / "all_heldouts_summary.md",
        "\n".join(
            [
                "# All-heldouts Benchmark Summary",
                "",
                markdown_table(
                    ["Heldout", "N Runs", "Seeds", "Eval TTA", "Test Acc Mean", "Test Acc Std", "Test Macro-F1 Mean", "Test Macro-F1 Std"],
                    [
                        [
                            row["heldout_domain"],
                            row["n_runs"],
                            row["seeds"],
                            row["eval_tta_mode"],
                            row["test_accuracy_mean"],
                            row["test_accuracy_std"],
                            row["test_macro_f1_mean"],
                            row["test_macro_f1_std"],
                        ]
                        for row in rows
                    ],
                ),
                "",
            ]
        ),
    )


def aggregate_multi_seed(reports: list[dict], output_path: Path) -> None:
    """Write a summary with mean +/- std across seeds."""
    test_f1s = [r["test"]["macro_f1"] for r in reports]
    test_accs = [r["test"]["accuracy"] for r in reports]
    per_class_f1s: dict[str, list[float]] = {cls: [] for cls in CLASSES}
    for r in reports:
        for cls in CLASSES:
            per_class_f1s[cls].append(r["test"]["per_class"][cls]["f1"])

    summary = {
        "n_seeds": len(reports),
        "seeds": [r["config"]["seed"] for r in reports],
        "eval_tta_mode": reports[0]["eval_tta_mode"] if reports else "none",
        "test_macro_f1_mean": round(float(np.mean(test_f1s)), 4),
        "test_macro_f1_std": round(float(np.std(test_f1s)), 4),
        "test_accuracy_mean": round(float(np.mean(test_accs)), 4),
        "test_accuracy_std": round(float(np.std(test_accs)), 4),
        "per_class_f1_mean": {
            cls: round(float(np.mean(vals)), 4) for cls, vals in per_class_f1s.items()
        },
        "per_class_f1_std": {
            cls: round(float(np.std(vals)), 4) for cls, vals in per_class_f1s.items()
        },
        "run_names": [r["run_name"] for r in reports],
    }

    write_json(output_path / "multi_seed_summary.json", summary)

    rows = [
        [
            cls,
            f"{summary['per_class_f1_mean'][cls]:.4f}",
            f"{summary['per_class_f1_std'][cls]:.4f}",
        ]
        for cls in CLASSES
    ]
    md = "\n".join([
        "# Multi-seed Benchmark Summary",
        "",
        f"- Seeds: `{summary['seeds']}`",
        f"- Eval TTA mode: `{summary['eval_tta_mode']}`",
        f"- Test Macro-F1: `{summary['test_macro_f1_mean']:.4f} +/- {summary['test_macro_f1_std']:.4f}`",
        f"- Test Accuracy: `{summary['test_accuracy_mean']:.4f} +/- {summary['test_accuracy_std']:.4f}`",
        "",
        "## Per-class F1",
        "",
        markdown_table(["Class", "F1 Mean", "F1 Std"], rows),
        "",
    ])
    write_text(output_path / "multi_seed_summary.md", md)
    print(f"Multi-seed summary: F1 = {summary['test_macro_f1_mean']:.4f} +/- {summary['test_macro_f1_std']:.4f}")


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = load_yaml_config(config_path)
    cli_class_fractions = parse_cli_class_fractions(args.class_fraction)
    config = apply_overrides(
        config,
        {
            "heldout_domain": args.heldout_domain,
            "backbone": args.backbone,
            "synthetic_manifest": args.synthetic_manifest,
            "train_fraction": args.train_fraction,
            "epochs": args.epochs,
            "all_heldouts": True if args.all_heldouts else None,
            "class_fractions": cli_class_fractions,
            "eval_tta_mode": args.eval_tta_mode,
        },
    )
    config = validate_config(config)

    seeds = args.seeds or [config["seed"]]
    heldout_domains = DOMAINS if config.get("all_heldouts") else [config["heldout_domain"]]
    reports_by_domain: dict[str, list[dict]] = {}

    for heldout_domain in heldout_domains:
        domain_config = {**config, "heldout_domain": heldout_domain}
        if len(heldout_domains) > 1:
            print(f"\n{'#' * 72}")
            print(f"All-heldouts sweep: heldout={heldout_domain}")
            print(f"{'#' * 72}")

        if len(seeds) == 1:
            domain_config["seed"] = seeds[0]
            reports_by_domain[heldout_domain] = [run_single_seed(domain_config)]
            continue

        reports = []
        for idx, seed in enumerate(seeds, start=1):
            print(f"\n{'='*60}")
            print(f"Running heldout={heldout_domain} seed {seed} ({idx}/{len(seeds)})")
            print(f"{'='*60}")
            seed_config = {**domain_config, "seed": seed}
            report = run_single_seed(seed_config)
            reports.append(report)

        output_root = ensure_dir(resolve_project_path(PROJECT_ROOT, config["output_root"]))
        heldout_root = output_root / config["backbone"] / f"heldout_{heldout_domain}"
        aggregate_multi_seed(reports, heldout_root)
        reports_by_domain[heldout_domain] = reports

    if len(heldout_domains) > 1:
        output_root = ensure_dir(resolve_project_path(PROJECT_ROOT, config["output_root"]))
        write_all_heldouts_summary(reports_by_domain, output_root, config["backbone"])


if __name__ == "__main__":
    main()
