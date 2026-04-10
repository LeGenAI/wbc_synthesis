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

HARD_CLASSES = [
    "eosinophil",
    "monocyte",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SUPPORTED_BACKBONES = {"efficientnet_b0", "vgg16"}

PROMPT_STYLES = {"standard", "clinical", "target_domain"}

DOMAIN_GENERATION_PROMPTS = {
    "domain_a_pbc": "May-Grunwald Giemsa stain, CellaVision automated analyzer, Barcelona Spain",
    "domain_b_raabin": "Giemsa stain, smartphone microscope camera, Iran hospital",
    "domain_c_mll23": "Pappenheim stain, Metafer scanner, Germany clinical lab",
    "domain_e_amc": "Romanowsky stain, miLab automated analyzer, South Korea AMC",
}

CLASS_MORPHOLOGY = {
    "basophil": "bilobed nucleus with dark purple-black granules filling cytoplasm",
    "eosinophil": "bilobed nucleus with bright orange-red granules",
    "lymphocyte": "large round nucleus with scant agranular cytoplasm",
    "monocyte": "kidney-shaped or folded nucleus with grey-blue cytoplasm",
    "neutrophil": "multilobed nucleus with pale pink granules",
}

NEGATIVE_PROMPT_DEFAULT = (
    "cartoon, illustration, text, watermark, multiple cells, "
    "heavy artifacts, unrealistic colors, deformed nucleus, blurry"
)

LORA_DIR_PATTERN = "lora/weights/multidomain_{class_name}"

DEFAULT_PRODUCTION_EPOCHS = 30
DEFAULT_DEV_EPOCHS = 3
