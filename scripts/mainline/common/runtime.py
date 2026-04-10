"""Runtime helpers shared across mainline scripts."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from scripts.mainline.common.constants import SUPPORTED_BACKBONES


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "CLAUDE.md").exists() and (candidate / "scripts").exists():
            return candidate
    raise RuntimeError(f"Could not find project root from: {start}")


def resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_backbone(
    model_name: str,
    num_classes: int,
    full_finetune: bool = False,
) -> nn.Module:
    if model_name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {model_name}")

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        head_params = model.classifier.parameters()
    else:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head_params = model.classifier.parameters()

    if not full_finetune:
        for param in model.parameters():
            param.requires_grad = False
        for param in head_params:
            param.requires_grad = True

    return model

