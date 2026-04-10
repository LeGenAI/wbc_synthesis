"""Canonical constants for the mainline WBC pipeline."""

from __future__ import annotations

DOMAINS = [
    "domain_a_pbc",
    "domain_b_raabin",
    "domain_c_mll23",
    "domain_e_amc",
]

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

CLASSES = [
    "basophil",
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SUPPORTED_BACKBONES = {"efficientnet_b0", "vgg16"}

