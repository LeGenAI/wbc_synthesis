"""YAML config loading and override helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def dump_yaml_config(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def apply_overrides(config: dict, overrides: dict) -> dict:
    merged = deepcopy(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged

