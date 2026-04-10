"""Split and sampling helpers for manifest-based pipelines."""

from __future__ import annotations

import random
from collections import defaultdict


def stratified_train_val_split(
    items: list[dict],
    val_ratio: float,
    seed: int,
    group_fields: tuple[str, ...] = ("class_name", "domain"),
) -> tuple[list[dict], list[dict]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        buckets[tuple(item[field] for field in group_fields)].append(item)

    rng = random.Random(seed)
    train_items: list[dict] = []
    val_items: list[dict] = []

    for _, bucket_items in sorted(buckets.items()):
        bucket = list(bucket_items)
        rng.shuffle(bucket)
        if len(bucket) == 1:
            train_items.extend(bucket)
            continue
        n_val = int(round(len(bucket) * val_ratio))
        n_val = max(1, min(n_val, len(bucket) - 1))
        val_items.extend(bucket[:n_val])
        train_items.extend(bucket[n_val:])

    return train_items, val_items


def stratified_fraction(
    items: list[dict],
    fraction: float,
    seed: int,
    group_fields: tuple[str, ...] = ("class_name", "domain"),
) -> list[dict]:
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction >= 1.0:
        return list(items)

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        buckets[tuple(item[field] for field in group_fields)].append(item)

    rng = random.Random(seed)
    selected: list[dict] = []
    for _, bucket_items in sorted(buckets.items()):
        bucket = list(bucket_items)
        rng.shuffle(bucket)
        n_keep = max(1, int(round(len(bucket) * fraction)))
        n_keep = min(n_keep, len(bucket))
        selected.extend(bucket[:n_keep])
    return selected


def stratified_class_fractions(
    items: list[dict],
    class_fractions: dict[str, float],
    seed: int,
    group_fields: tuple[str, ...] = ("class_name", "domain"),
) -> list[dict]:
    if not class_fractions:
        return list(items)

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        buckets[tuple(item[field] for field in group_fields)].append(item)

    rng = random.Random(seed)
    selected: list[dict] = []
    for key, bucket_items in sorted(buckets.items()):
        bucket = list(bucket_items)
        rng.shuffle(bucket)
        class_name = key[0]
        fraction = float(class_fractions.get(class_name, 1.0))
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError(f"class fraction for {class_name} must be in (0, 1], got {fraction}")
        if fraction >= 1.0:
            selected.extend(bucket)
            continue
        n_keep = max(1, int(round(len(bucket) * fraction)))
        n_keep = min(n_keep, len(bucket))
        selected.extend(bucket[:n_keep])
    return selected
