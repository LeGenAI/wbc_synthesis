# WBC Synthesis: SDXL-LoRA Augmentation for Robust WBC Classification

**Goal:** Improve 8-class WBC classification robustness using SDXL base + class-wise LoRA fine-tuning for data augmentation.

**Success criteria:**
- Val/Test macro-F1 ↑ vs Real-only baseline
- Corruption test: mean F1 drop ↓ (more robust)

---

## Directory Structure

```
wbc_synthesis/
├── data/
│   ├── raw/                      # Kaggle dataset (populated by 00_download_data.py)
│   ├── processed/
│   │   ├── train/{class}/        # 70% split, 512×512
│   │   ├── val/{class}/          # 15% split
│   │   ├── test/{class}/         # 15% split
│   │   └── train_sharp/{class}/  # Top-70% by Laplacian variance (LoRA input)
│   ├── generated/{class}/{ds*}/  # Raw SDXL-LoRA outputs
│   └── filtered/{class}/{ds*}/   # Quality-gated outputs (for CNN training)
├── lora/
│   ├── configs/
│   └── weights/{class}/          # Per-class LoRA checkpoints
├── models/
│   ├── baseline_cnn.pt           # Real-only baseline (used by filter script)
│   └── *.pt                      # All experiment checkpoints
├── results/
│   ├── baseline/                 # real_only, real_augmented JSON results
│   ├── augmented/                # real_generated JSON results
│   ├── ablation/                 # A1–A4 JSON results
│   └── corrupted/                # Per-checkpoint corruption eval JSONs
├── scripts/
│   ├── legacy/
│   │   └── phase_00_07_initial_pipeline/
│   ├── mainline/
│   │   ├── data/
│   │   ├── generation/
│   │   ├── scoring/
│   │   ├── benchmark/
│   │   └── reporting/
│   └── README.md
├── reference/
│   ├── reference_matrix.md
│   ├── reference_matrix.csv
│   └── references.bib
├── logs/                         # LoRA training logs
└── requirements.txt
```

---

## Execution Order

> Legacy note: the original `00~07` pipeline has been moved under
> `scripts/legacy/phase_00_07_initial_pipeline/`.

### 0. Install dependencies
```bash
pip install -r requirements.txt
# Login to HuggingFace (for SDXL model weights)
huggingface-cli login
# Set up Kaggle credentials (~/.kaggle/kaggle.json)
```

### 1. Download dataset
```bash
python scripts/legacy/phase_00_07_initial_pipeline/00_download_data.py
```

### 2. Prepare data (split + preprocess + sharp filter)
```bash
python scripts/legacy/phase_00_07_initial_pipeline/01_prepare_data.py --seed 0
# Repeat with --seed 1 and --seed 2 for multi-seed experiments
```

### 3. Train class-wise LoRA (run once per class, or --all)
```bash
# Single class
python scripts/legacy/phase_00_07_initial_pipeline/02_train_lora.py --class_name neutrophil

# All classes sequentially
python scripts/legacy/phase_00_07_initial_pipeline/02_train_lora.py --all

# Dry run (inspect commands without training)
python scripts/legacy/phase_00_07_initial_pipeline/02_train_lora.py --all --dry_run
```
> **Note:** Requires diffusers `train_dreambooth_lora_sdxl.py` in PATH.
> Install via: `pip install diffusers[training]` and copy from
> `$(python -c "import diffusers; print(diffusers.__file__.replace('__init__.py',''))")examples/dreambooth/`

### 4. Generate images (img2img, 3 denoise levels)
```bash
# All classes, 1× multiplier, 3 denoise levels
python scripts/legacy/phase_00_07_initial_pipeline/03_generate.py --all --multiplier 1

# Specific class and denoise strength
python scripts/legacy/phase_00_07_initial_pipeline/03_generate.py --class_name basophil --denoise 0.35

# For ablation A4: also generate 2× and 5×
python scripts/legacy/phase_00_07_initial_pipeline/03_generate.py --all --multiplier 2
python scripts/legacy/phase_00_07_initial_pipeline/03_generate.py --all --multiplier 5
```

### 5a. Train baseline CNN (Real-only) — needed BEFORE filtering
```bash
python scripts/legacy/phase_00_07_initial_pipeline/05_train_cnn.py --mode real_only --model efficientnet_b0 --seed 42
# This also saves models/baseline_cnn.pt for the filter step
```

### 5b. Filter generated images
```bash
python scripts/legacy/phase_00_07_initial_pipeline/04_filter_generated.py \
    --classifier_ckpt models/baseline_cnn.pt \
    --conf_threshold 0.7 \
    --sharp_floor_pctile 20
```

### 6. Train all CNN variants
```bash
# Baseline 1: Real-only (already done in 5a)

# Baseline 2: Real + traditional augmentation
python scripts/legacy/phase_00_07_initial_pipeline/05_train_cnn.py --mode real_augmented --model efficientnet_b0 --seed 42

# Experiment: Real + SDXL-generated (1×, denoise 0.35)
python scripts/legacy/phase_00_07_initial_pipeline/05_train_cnn.py --mode real_generated --model efficientnet_b0 \
    --gen_multiplier 1 --denoise_tag ds035 --seed 42

# Or run all three modes automatically:
python scripts/legacy/phase_00_07_initial_pipeline/05_train_cnn.py --run_all --model efficientnet_b0 --seed 42
```

### 7. Robustness evaluation
```bash
# Single checkpoint
python scripts/legacy/phase_00_07_initial_pipeline/06_robustness_eval.py --ckpt models/real_only_efficientnet_b0_seed42_best.pt

# All checkpoints
python scripts/legacy/phase_00_07_initial_pipeline/06_robustness_eval.py --all_ckpts
```

### 8. Ablation studies
```bash
# Single ablation
python scripts/legacy/phase_00_07_initial_pipeline/07_ablation.py --ablation A4

# All ablations
python scripts/legacy/phase_00_07_initial_pipeline/07_ablation.py --all

# Dry run
python scripts/legacy/phase_00_07_initial_pipeline/07_ablation.py --all --dry_run

# Summary table
python scripts/legacy/phase_00_07_initial_pipeline/07_ablation.py --summarize
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LoRA strategy | Class-wise (8 LoRAs) | Avoids class identity mixing |
| Generation method | img2img, denoise 0.25–0.45 | More stable than text2img on medical images |
| LoRA data | Top-70% Laplacian-variance | Remove blurry samples that hurt LoRA quality |
| Quality gate | Classifier confidence ≥ 0.7 | Catch wrong-class or artifact-heavy generations |
| Evaluation | macro-F1 (primary) + corruption robustness | Handles class imbalance + real-world variation |

---

## Ablation Map

| ID | What varies | Fixed |
|---|---|---|
| A1 | Class-wise LoRA vs Single LoRA + token | mult=1×, ds=0.35, filter=ON |
| A2 | text2img vs img2img | mult=1×, filter=ON |
| A3 | Filter ON vs OFF | mult=1×, ds=0.35, img2img |
| A4 | Multiplier: 1× / 2× / 5× | ds=0.35, filter=ON, img2img |

---

## Expected Results Template

| Experiment | Test Acc | Macro-F1 | Mean F1 Drop (corruption) |
|---|---|---|---|
| Real-only | — | — | — |
| Real + Trad. Aug | — | — | — |
| Real + SDXL 1× | — | — | — |
| Real + SDXL 2× | — | — | — |
| Real + SDXL 5× | — | — | — |
