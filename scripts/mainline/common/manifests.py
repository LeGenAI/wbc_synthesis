"""Manifest helpers used by stage 01 and stage 05."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from scripts.mainline.common.constants import IMG_EXTS


def build_image_id(file_rel: str) -> str:
    return file_rel.replace("/", "__")


def collect_inventory(
    dataset_root: Path,
    project_root: Path,
    domains: list[str],
    classes: list[str],
) -> list[dict]:
    items: list[dict] = []
    for domain in domains:
        for class_name in classes:
            class_dir = dataset_root / domain / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            image_paths = sorted(
                path for path in class_dir.iterdir() if path.suffix.lower() in IMG_EXTS
            )
            if not image_paths:
                raise RuntimeError(f"No images found under: {class_dir}")

            for path in image_paths:
                file_abs = str(path.resolve())
                file_rel = path.resolve().relative_to(project_root).as_posix()
                items.append(
                    {
                        "file_abs": file_abs,
                        "file_rel": file_rel,
                        "class_name": class_name,
                        "domain": domain,
                        "split": "inventory",
                        "source_type": "real",
                        "image_id": build_image_id(file_rel),
                    }
                )
    return items


def write_manifest_payload(
    manifest_type: str,
    items: list[dict],
    metadata: dict | None = None,
) -> dict:
    return {
        "manifest_type": manifest_type,
        "metadata": metadata or {},
        "n_items": len(items),
        "items": items,
    }


def load_manifest_items(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json_load(handle.read(), path)
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError(f"Invalid manifest payload: {path}")
    items = payload["items"]
    if not isinstance(items, list):
        raise ValueError(f"Manifest items must be a list: {path}")
    return items


def json_load(text: str, path: Path) -> object:
    import json

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON manifest {path}: {exc}") from exc


def count_by_domain_class(items: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    counter = Counter((item["domain"], item["class_name"]) for item in items)
    for (domain, class_name), value in sorted(counter.items()):
        counts.setdefault(domain, {})[class_name] = value
    return dict(counts)

