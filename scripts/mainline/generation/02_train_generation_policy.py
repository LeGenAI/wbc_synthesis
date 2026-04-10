#!/usr/bin/env python3
"""Stage 02: freeze a reusable generation-policy artifact."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mainline.common.config import apply_overrides, dump_yaml_config, load_yaml_config
from scripts.mainline.common.constants import (
    CLASSES,
    DOMAINS,
    DOMAIN_LABELS,
    DOMAIN_SHORT,
    NEGATIVE_PROMPT_DEFAULT,
    PROMPT_STYLES,
)
from scripts.mainline.common.manifests import load_manifest_items, write_manifest_payload
from scripts.mainline.common.policy import build_generation_prompt, slugify
from scripts.mainline.common.reporting import markdown_table, write_json, write_text
from scripts.mainline.common.runtime import ensure_dir, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reusable mainline generation-policy artifact.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mainline/generation/policy_v1.yaml",
        help="Path to the generation-policy YAML config.",
    )
    parser.add_argument("--policy-id", type=str, default=None)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--lora-dir", type=str, default=None)
    parser.add_argument("--reference-images-per-domain", type=int, default=None)
    parser.add_argument("--prompt-style", type=str, default=None)
    return parser.parse_args()


def validate_config(config: dict) -> dict:
    if config["class_name"] not in CLASSES:
        raise ValueError(f"Unknown class_name: {config['class_name']}")
    source_domains = list(config["source_domains"])
    target_domains = list(config["target_domains"])
    invalid_source = sorted(set(source_domains) - set(DOMAINS))
    invalid_target = sorted(set(target_domains) - set(DOMAINS))
    if invalid_source:
        raise ValueError(f"Unknown source_domains: {invalid_source}")
    if invalid_target:
        raise ValueError(f"Unknown target_domains: {invalid_target}")
    if config["prompt_style"] not in PROMPT_STYLES:
        raise ValueError(f"Unsupported prompt_style: {config['prompt_style']}")
    strengths = [round(float(value), 2) for value in config["strengths"]]
    if not strengths:
        raise ValueError("strengths must contain at least one value")
    for value in strengths:
        if not (0.0 < value < 1.0):
            raise ValueError(f"strengths must be in (0, 1): {value}")

    # Warn about source/target overlap when a heldout domain is implied.
    # If a domain appears in both source and target, holding it out later
    # would require removing it from source too — easy to forget.
    overlap = sorted(set(source_domains) & set(target_domains))
    if overlap:
        import warnings
        warnings.warn(
            f"source_domains and target_domains overlap on {overlap}. "
            f"If any of these become a heldout domain in stage 03/05, "
            f"the reference pool must also exclude them to avoid leakage.",
            stacklevel=2,
        )

    return {
        **config,
        "policy_id": slugify(config["policy_id"]),
        "source_domains": source_domains,
        "target_domains": target_domains,
        "strengths": strengths,
        "guidance_scale": float(config["guidance_scale"]),
        "num_inference_steps": int(config["num_inference_steps"]),
        "image_size": int(config["image_size"]),
        "reference_images_per_domain": int(config["reference_images_per_domain"]),
        "seed": int(config["seed"]),
        "negative_prompt": config.get("negative_prompt") or NEGATIVE_PROMPT_DEFAULT,
    }


def select_reference_pool(items: list[dict], class_name: str, source_domains: list[str]) -> list[dict]:
    filtered = [
        item for item in items if item["class_name"] == class_name and item["domain"] in source_domains
    ]
    if not filtered:
        raise RuntimeError(
            f"No inventory items found for class={class_name} source_domains={source_domains}"
        )
    return filtered


def sample_reference_examples(
    items: list[dict], per_domain: int, seed: int,
) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in items:
        grouped.setdefault(item["domain"], []).append(item)
    rng = random.Random(seed)
    selected: list[dict] = []
    for domain, domain_items in sorted(grouped.items()):
        pool = list(domain_items)
        rng.shuffle(pool)
        selected.extend(pool[: min(len(pool), per_domain)])
    return selected


def build_policy_markdown(policy_spec: dict, reference_counts: dict[str, int]) -> str:
    settings_rows = [
        ["policy_id", policy_spec["policy_id"]],
        ["class_name", policy_spec["class_name"]],
        ["base_model", policy_spec["base_model"]],
        ["lora_dir", policy_spec["lora_dir"]],
        ["prompt_style", policy_spec["prompt_style"]],
        ["target_domains", ", ".join(policy_spec["target_domains"])],
        ["strengths", ", ".join(str(v) for v in policy_spec["strengths"])],
        ["guidance_scale", policy_spec["guidance_scale"]],
        ["num_inference_steps", policy_spec["num_inference_steps"]],
    ]
    reference_rows = [
        [domain, DOMAIN_LABELS[domain], reference_counts.get(domain, 0)] for domain in policy_spec["source_domains"]
    ]
    prompt_rows = [
        [domain, build_generation_prompt(policy_spec["class_name"], domain, policy_spec["prompt_style"])]
        for domain in policy_spec["target_domains"]
    ]
    lines = [
        "# Mainline Generation Policy Card",
        "",
        "## Settings",
        "",
        markdown_table(["Field", "Value"], settings_rows),
        "",
        "## Reference Pool",
        "",
        markdown_table(["Domain", "Label", "Count"], reference_rows),
        "",
        "## Target-domain Prompt Examples",
        "",
        markdown_table(["Target Domain", "Prompt"], prompt_rows),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = load_yaml_config(config_path)
    config = apply_overrides(
        config,
        {
            "policy_id": args.policy_id,
            "class_name": args.class_name,
            "lora_dir": args.lora_dir,
            "reference_images_per_domain": args.reference_images_per_domain,
            "prompt_style": args.prompt_style,
        },
    )
    config = validate_config(config)

    inventory_manifest = resolve_project_path(PROJECT_ROOT, config["inventory_manifest"])
    lora_dir = resolve_project_path(PROJECT_ROOT, config["lora_dir"])
    output_root = ensure_dir(resolve_project_path(PROJECT_ROOT, config["output_root"]))
    policy_dir = ensure_dir(output_root / config["policy_id"])

    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA directory does not exist: {lora_dir}")
    if not (lora_dir / "pytorch_lora_weights.safetensors").exists():
        raise FileNotFoundError(f"Missing LoRA weights in: {lora_dir}")

    inventory_items = load_manifest_items(inventory_manifest)
    reference_pool = select_reference_pool(
        inventory_items,
        class_name=config["class_name"],
        source_domains=config["source_domains"],
    )
    sampled_examples = sample_reference_examples(
        reference_pool,
        per_domain=config["reference_images_per_domain"],
        seed=config["seed"],
    )
    reference_counts = Counter(item["domain"] for item in reference_pool)

    policy_spec = {
        "policy_id": config["policy_id"],
        "class_name": config["class_name"],
        "base_model": config["base_model"],
        "lora_dir": str(lora_dir),
        "inventory_manifest": str(inventory_manifest),
        "source_domains": config["source_domains"],
        "source_domain_short": {domain: DOMAIN_SHORT[domain] for domain in config["source_domains"]},
        "target_domains": config["target_domains"],
        "prompt_style": config["prompt_style"],
        "negative_prompt": config["negative_prompt"],
        "strengths": config["strengths"],
        "guidance_scale": config["guidance_scale"],
        "num_inference_steps": config["num_inference_steps"],
        "image_size": config["image_size"],
        "seed": config["seed"],
        "reference_pool_manifest": str(policy_dir / "reference_pool_manifest.json"),
        "reference_example_manifest": str(policy_dir / "reference_examples_manifest.json"),
        "reference_pool_count": len(reference_pool),
        "reference_examples_per_domain": config["reference_images_per_domain"],
        "prompt_examples": {
            domain: build_generation_prompt(config["class_name"], domain, config["prompt_style"])
            for domain in config["target_domains"]
        },
    }

    write_json(
        policy_dir / "reference_pool_manifest.json",
        write_manifest_payload(
            "generation_policy_reference_pool",
            reference_pool,
            {"policy_id": config["policy_id"], "class_name": config["class_name"]},
        ),
    )
    write_json(
        policy_dir / "reference_examples_manifest.json",
        write_manifest_payload(
            "generation_policy_reference_examples",
            sampled_examples,
            {"policy_id": config["policy_id"], "class_name": config["class_name"]},
        ),
    )
    write_json(policy_dir / "policy_spec.json", policy_spec)
    write_text(
        policy_dir / "policy_card.md",
        build_policy_markdown(policy_spec, dict(reference_counts)),
    )
    dump_yaml_config(policy_dir / "resolved_config.yaml", config)
    print(f"Wrote policy artifact to: {policy_dir}")


if __name__ == "__main__":
    main()
