"""Helpers for generation-policy stage artifacts and prompts."""

from __future__ import annotations

import re

from scripts.mainline.common.constants import (
    CLASS_MORPHOLOGY,
    DOMAIN_GENERATION_PROMPTS,
    DOMAIN_SHORT,
    PROMPT_STYLES,
)


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")


def build_generation_prompt(
    class_name: str,
    target_domain: str,
    prompt_style: str,
) -> str:
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unsupported prompt style: {prompt_style}")

    morphology = CLASS_MORPHOLOGY[class_name]
    domain_prompt = DOMAIN_GENERATION_PROMPTS[target_domain]

    if prompt_style == "standard":
        return (
            f"microscopy image of a single {class_name} white blood cell, "
            f"peripheral blood smear, {morphology}, {domain_prompt}, "
            "sharp focus, realistic, clinical lab imaging"
        )
    if prompt_style == "clinical":
        return (
            f"clinical hematology microscopy, isolated {class_name} leukocyte, "
            f"{morphology}, {domain_prompt}, bright-field microscopy, "
            "diagnostic slide quality, preserved nucleus morphology"
        )
    return (
        f"domain-conditioned hematology image, single {class_name} white blood cell, "
        f"{morphology}, target style {DOMAIN_SHORT[target_domain]}, "
        f"{domain_prompt}, realistic stain and scanner characteristics"
    )
