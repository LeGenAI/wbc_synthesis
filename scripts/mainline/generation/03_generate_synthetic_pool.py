#!/usr/bin/env python3
"""Stage 03: generate a synthetic pool from a frozen policy artifact."""

from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mainline.common.config import apply_overrides, dump_yaml_config, load_yaml_config
from scripts.mainline.common.constants import DOMAIN_LABELS, DOMAIN_SHORT, DOMAINS
from scripts.mainline.common.manifests import load_manifest_items, write_manifest_payload
from scripts.mainline.common.policy import build_generation_prompt
from scripts.mainline.common.reporting import markdown_table, write_json, write_text
from scripts.mainline.common.runtime import ensure_dir, get_device, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic pool from a mainline policy.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mainline/generation/generate_v1.yaml",
        help="Path to the synthetic-generation YAML config.",
    )
    parser.add_argument("--heldout-domain", type=str, default=None)
    parser.add_argument("--policy-dir", type=str, default=None)
    parser.add_argument("--reference-manifest", type=str, default=None)
    parser.add_argument("--n-per-domain", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_config(config: dict, policy_spec: dict) -> dict:
    heldout_domain = config.get("heldout_domain")
    if heldout_domain is not None and heldout_domain not in DOMAINS:
        raise ValueError(f"Unknown heldout_domain: {heldout_domain}")
    target_domains = config.get("target_domains") or list(policy_spec["target_domains"])
    invalid_target = sorted(set(target_domains) - set(DOMAINS))
    if invalid_target:
        raise ValueError(f"Unknown target_domains: {invalid_target}")

    # Leakage guard: heldout domain must not appear in source_domains
    if heldout_domain is not None:
        source_domains = list(policy_spec.get("source_domains", []))
        if heldout_domain in source_domains:
            raise ValueError(
                f"Heldout domain '{heldout_domain}' is also listed in "
                f"the policy's source_domains {source_domains}. "
                f"This would leak held-out data into the reference pool."
            )
        if heldout_domain in target_domains:
            import warnings
            warnings.warn(
                f"Heldout domain '{heldout_domain}' is in target_domains. "
                f"Generated images targeting this domain will be excluded "
                f"from benchmark training by the leakage filter.",
                stacklevel=2,
            )
    max_images = config.get("max_images")
    if max_images is not None:
        max_images = int(max_images)
        if max_images <= 0:
            raise ValueError(f"max_images must be positive when set: {max_images}")
    return {
        **config,
        "heldout_domain": heldout_domain,
        "target_domains": list(target_domains),
        "n_per_domain": int(config["n_per_domain"]),
        "n_seeds": int(config["n_seeds"]),
        "seed": int(config["seed"]),
        "max_images": max_images,
        "dry_run": bool(config.get("dry_run", False)),
        "force": bool(config.get("force", False)),
    }


def load_policy_spec(policy_dir: Path) -> dict:
    policy_path = policy_dir / "policy_spec.json"
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing policy_spec.json under {policy_dir}")
    import json

    with open(policy_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_reference_items(policy_spec: dict, reference_manifest: str | None) -> list[dict]:
    manifest_path = Path(reference_manifest) if reference_manifest else Path(policy_spec["reference_pool_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = resolve_project_path(PROJECT_ROOT, manifest_path)
    return load_manifest_items(manifest_path)


def apply_reference_filter(items: list[dict], heldout_domain: str | None, source_domains: list[str]) -> tuple[list[dict], dict]:
    kept = []
    excluded_for_heldout = 0
    excluded_for_source = 0
    for item in items:
        if heldout_domain is not None and item["domain"] == heldout_domain:
            excluded_for_heldout += 1
            continue
        if item["domain"] not in source_domains:
            excluded_for_source += 1
            continue
        kept.append(copy.deepcopy(item))
    return kept, {
        "excluded_for_heldout_domain": excluded_for_heldout,
        "excluded_for_source_domain_mismatch": excluded_for_source,
    }


def sample_reference_items(
    items: list[dict], n_per_domain: int, seed: int,
) -> list[dict]:
    import random

    grouped: dict[str, list[dict]] = {}
    for item in items:
        grouped.setdefault(item["domain"], []).append(item)
    rng = random.Random(seed)
    selected: list[dict] = []
    for domain, domain_items in sorted(grouped.items()):
        pool = list(domain_items)
        rng.shuffle(pool)
        selected.extend(pool[: min(len(pool), n_per_domain)])
    return selected


def build_run_name(policy_id: str, heldout_domain: str | None, n_per_domain: int, n_seeds: int, seed: int) -> str:
    heldout_tag = DOMAIN_SHORT[heldout_domain] if heldout_domain else "noheldout"
    return f"{policy_id}__heldout_{heldout_tag}__n{n_per_domain}__seeds{n_seeds}__seed{seed}"


def load_pipeline(base_model: str, lora_dir: Path, device: torch.device):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    variant = "fp16" if device.type == "cuda" else None
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        variant=variant,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )
    pipe.load_lora_weights(str(lora_dir))
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_one(
    pipe,
    ref_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
):
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=ref_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]


def summarize_items(items: list[dict]) -> list[list[object]]:
    counter = Counter((item["domain"], item["class_name"]) for item in items)
    rows = []
    for (domain, class_name), count in sorted(counter.items()):
        rows.append([domain, class_name, count])
    return rows


def render_summary(run_report: dict) -> str:
    source_rows = run_report["reference_pool_rows"]
    generated_rows = run_report["generated_rows"]
    lines = [
        "# Mainline Synthetic Pool Generation Summary",
        "",
        f"- policy_id: `{run_report['policy_id']}`",
        f"- run_name: `{run_report['run_name']}`",
        f"- class_name: `{run_report['class_name']}`",
        f"- heldout_domain: `{run_report['heldout_domain']}`",
        f"- device: `{run_report['device']}`",
        f"- dry_run: `{run_report['dry_run']}`",
        "",
        "## Reference Pool",
        "",
        markdown_table(["Domain", "Class", "Count"], source_rows),
        "",
        "## Generated Pool",
        "",
        markdown_table(["Target Domain", "Class", "Count"], generated_rows),
        "",
        "## Leakage Guard",
        "",
        f"- excluded_for_heldout_domain: `{run_report['guard']['excluded_for_heldout_domain']}`",
        f"- excluded_for_source_domain_mismatch: `{run_report['guard']['excluded_for_source_domain_mismatch']}`",
        f"- target_domains_overlapping_heldout: `{run_report['target_domains_overlapping_heldout']}`",
        f"- benchmark_compatible_generated_count: `{run_report['benchmark_compatible_generated_count']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    raw_config = load_yaml_config(config_path)
    raw_config = apply_overrides(
        raw_config,
        {
            "heldout_domain": args.heldout_domain,
            "policy_dir": args.policy_dir,
            "reference_manifest": args.reference_manifest,
            "n_per_domain": args.n_per_domain,
            "n_seeds": args.n_seeds,
            "max_images": args.max_images,
            "dry_run": True if args.dry_run else None,
            "force": True if args.force else None,
        },
    )

    policy_dir = resolve_project_path(PROJECT_ROOT, raw_config["policy_dir"])
    policy_spec = load_policy_spec(policy_dir)
    config = validate_config(raw_config, policy_spec)

    lora_dir = resolve_project_path(PROJECT_ROOT, policy_spec["lora_dir"])
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA directory does not exist: {lora_dir}")

    run_name = build_run_name(
        policy_id=policy_spec["policy_id"],
        heldout_domain=config["heldout_domain"],
        n_per_domain=config["n_per_domain"],
        n_seeds=config["n_seeds"],
        seed=config["seed"],
    )
    image_root = ensure_dir(resolve_project_path(PROJECT_ROOT, config["output_image_root"])) / run_name
    run_root = ensure_dir(resolve_project_path(PROJECT_ROOT, config["output_root"])) / run_name
    if run_root.exists() and any(run_root.iterdir()) and not config["force"]:
        raise FileExistsError(f"Run directory already exists. Use --force to overwrite: {run_root}")
    ensure_dir(run_root)
    ensure_dir(image_root)

    reference_items = load_reference_items(policy_spec, config.get("reference_manifest"))
    filtered_reference_items, guard = apply_reference_filter(
        reference_items,
        heldout_domain=config["heldout_domain"],
        source_domains=policy_spec["source_domains"],
    )
    sampled_reference_items = sample_reference_items(
        filtered_reference_items,
        n_per_domain=config["n_per_domain"],
        seed=config["seed"],
    )
    if not sampled_reference_items:
        raise RuntimeError("No reference items remain after filtering and sampling")

    generated_items = []
    generated_count = 0
    max_images = config["max_images"]
    device = get_device()
    overlapping_target_domains = []
    if config["heldout_domain"] is not None:
        overlapping_target_domains = [
            domain for domain in config["target_domains"] if domain == config["heldout_domain"]
        ]

    if not config["dry_run"]:
        pipe = load_pipeline(policy_spec["base_model"], lora_dir, device)
    else:
        pipe = None

    stop_generation = False
    for ref_item in sampled_reference_items:
        ref_image = Image.open(ref_item["file_abs"]).convert("RGB").resize(
            (policy_spec["image_size"], policy_spec["image_size"]),
            Image.LANCZOS,
        )
        ref_stem = Path(ref_item["file_abs"]).stem
        ref_domain_short = DOMAIN_SHORT[ref_item["domain"]]
        for target_domain in config["target_domains"]:
            prompt = build_generation_prompt(
                policy_spec["class_name"],
                target_domain=target_domain,
                prompt_style=policy_spec["prompt_style"],
            )
            for strength in policy_spec["strengths"]:
                for seed_idx in range(config["n_seeds"]):
                    seed = config["seed"] + seed_idx
                    filename = (
                        f"src_{ref_domain_short}__ref_{ref_stem}__tgt_{DOMAIN_SHORT[target_domain]}"
                        f"__str_{str(strength).replace('.', 'p')}__seed_{seed}.png"
                    )
                    target_dir = ensure_dir(image_root / policy_spec["class_name"] / target_domain)
                    image_path = target_dir / filename
                    if not config["dry_run"]:
                        generated = generate_one(
                            pipe=pipe,
                            ref_image=ref_image,
                            prompt=prompt,
                            negative_prompt=policy_spec["negative_prompt"],
                            strength=strength,
                            guidance_scale=policy_spec["guidance_scale"],
                            num_inference_steps=policy_spec["num_inference_steps"],
                            seed=seed,
                        )
                        generated.save(image_path)
                    item = {
                        "file_abs": str(image_path.resolve()),
                        "file_rel": image_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
                        "class_name": policy_spec["class_name"],
                        "domain": target_domain,
                        "split": "train",
                        "source_type": "synthetic",
                        "policy_id": policy_spec["policy_id"],
                        "ref_image_id": ref_item["image_id"],
                        "ref_domain": ref_item["domain"],
                        "ref_file_abs": ref_item["file_abs"],
                        "ref_file_rel": ref_item.get("file_rel"),
                        "target_domain": target_domain,
                        "strength": strength,
                        "seed": seed,
                        "prompt_style": policy_spec["prompt_style"],
                        "prompt": prompt,
                    }
                    generated_items.append(item)
                    generated_count += 1
                    if max_images is not None and generated_count >= max_images:
                        stop_generation = True
                        break
                if stop_generation:
                    break
            if stop_generation:
                break
        if stop_generation:
            break

    synthetic_manifest = write_manifest_payload(
        "synthetic_pool",
        generated_items,
        {
            "policy_id": policy_spec["policy_id"],
            "heldout_domain": config["heldout_domain"],
            "dry_run": config["dry_run"],
        },
    )
    report = {
        "policy_id": policy_spec["policy_id"],
        "run_name": run_name,
        "class_name": policy_spec["class_name"],
        "heldout_domain": config["heldout_domain"],
        "device": str(device),
        "dry_run": config["dry_run"],
        "policy_dir": str(policy_dir),
        "synthetic_manifest": str(run_root / "synthetic_manifest.json"),
        "reference_pool_count": len(filtered_reference_items),
        "reference_sample_count": len(sampled_reference_items),
        "generated_count": len(generated_items),
        "benchmark_compatible_generated_count": len(
            [item for item in generated_items if item["domain"] != config["heldout_domain"]]
        )
        if config["heldout_domain"] is not None
        else len(generated_items),
        "guard": guard,
        "reference_pool_rows": summarize_items(sampled_reference_items),
        "generated_rows": summarize_items(generated_items),
        "target_domains": config["target_domains"],
        "target_domains_overlapping_heldout": overlapping_target_domains,
    }

    write_json(run_root / "synthetic_manifest.json", synthetic_manifest)
    write_json(run_root / "report.json", report)
    write_text(run_root / "summary.md", render_summary(report))
    dump_yaml_config(run_root / "resolved_config.yaml", config)
    print(f"Wrote synthetic pool run to: {run_root}")


if __name__ == "__main__":
    main()
