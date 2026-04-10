# Mainline Pipeline

This directory contains the canonical research pipeline aligned to the paper supplementary structure.

## Stages

1. `data/01_prepare_multidomain_dataset.py`
   Source-domain normalization, split control, manifest creation.
2. `generation/02_train_generation_policy.py`
   Freeze a reusable generation-policy artifact (per class).
3. `generation/03_generate_synthetic_pool.py`
   Generate a synthetic pool from a frozen policy (per class).
4. `scoring/04_score_synthetic_pool.py`
   Two-stage quality gate plus reference-linked diagnostics (`ssim`, `cell_ssim`, `background_ssim`, `region_gap`).
5. `benchmark/05_train_lodo_utility_benchmark.py`
   Leakage-safe LODO utility benchmark with multi-seed, hard-class scarcity, low-data presets, test-time augmentation, and all-heldouts sweep support.
6. `reporting/06_make_submission_package.py`
   Figures, tables, appendix, and supplementary artifact assembly.

## End-to-end workflow

Since each class has its own LoRA, stages 02 and 03 run **per class**.
The merge step combines per-class manifests before scoring and benchmarking.

```bash
# Stage 01: build manifests
python -m scripts.mainline.data.01_prepare_multidomain_dataset \
    --config configs/mainline/data/base.yaml

# Stage 02: build policy per class (repeat for each class)
for cls in basophil eosinophil lymphocyte monocyte neutrophil; do
    python -m scripts.mainline.generation.02_train_generation_policy \
        --config configs/mainline/generation/policy_v1.yaml \
        --class-name $cls \
        --lora-dir lora/weights/multidomain_$cls \
        --policy-id policy_v1_${cls}_raabin
done

# Stage 03: generate per class (repeat for each class)
for cls in basophil eosinophil lymphocyte monocyte neutrophil; do
    python -m scripts.mainline.generation.03_generate_synthetic_pool \
        --config configs/mainline/generation/generate_v1_production.yaml \
        --policy-dir results/mainline/generation/policies/policy_v1_${cls}_raabin
done

# Merge per-class synthetic manifests
python -m scripts.mainline.generation.merge_synthetic_manifests \
    --manifests results/mainline/generation/runs/*/synthetic_manifest.json \
    --output results/mainline/generation/runs/combined_synthetic_manifest.json \
    --heldout-domain domain_b_raabin

# Stage 04: score and filter
python -m scripts.mainline.scoring.04_score_synthetic_pool \
    --synthetic-manifest results/mainline/generation/runs/combined_synthetic_manifest.json \
    --classifier-ckpt <path-to-real-only-best-model.pt> \
    --real-manifest results/mainline/data/heldout_domain_b_raabin/train_manifest.json \
    --output-root results/mainline/scoring/run_01

# Stage 05: benchmark (real-only baseline)
python -m scripts.mainline.benchmark.05_train_lodo_utility_benchmark \
    --config configs/mainline/benchmark/real_only_production.yaml \
    --seeds 42 123 456

# Stage 05: hard-class scarcity benchmark
python -m scripts.mainline.benchmark.05_train_lodo_utility_benchmark \
    --config configs/mainline/benchmark/real_only_hardclass_production.yaml \
    --seeds 42 123 456

# Stage 05: low-data baseline (CytoDiff-aligned scarcity)
python -m scripts.mainline.benchmark.05_train_lodo_utility_benchmark \
    --config configs/mainline/benchmark/real_only_lowdata_0p10_production.yaml \
    --seeds 42 123 456

# Stage 05: TTA baseline (Putzu-style evaluation axis)
python -m scripts.mainline.benchmark.05_train_lodo_utility_benchmark \
    --config configs/mainline/benchmark/real_only_tta_hflip_fivecrop_production.yaml \
    --seeds 42 123 456

# Stage 05: all-heldouts sweep
python -m scripts.mainline.benchmark.05_train_lodo_utility_benchmark \
    --config configs/mainline/benchmark/real_only_production.yaml \
    --all-heldouts \
    --seeds 42 123 456

# Stage 05: benchmark (real + filtered synth)
python -m scripts.mainline.benchmark.05_train_lodo_utility_benchmark \
    --config configs/mainline/benchmark/real_plus_synth_production.yaml \
    --synthetic-manifest results/mainline/scoring/run_01/filtered_manifest.json \
    --seeds 42 123 456
```

## Config tiers

- `configs/mainline/*/*.yaml` — dev configs for quick pipeline testing.
- `configs/mainline/*/*_production.yaml` — paper-quality settings (epochs=30, larger pools).
- `configs/mainline/benchmark/real_only_lowdata_0p{10,25,50}_production.yaml` — canonical low-data baselines for scarcity reporting.

## Design rules

- New experiments follow the stage numbering above, not the legacy numbering.
- Stages 02 and 03 produce per-class artifacts; the merge step combines them.
- Stage 04 sits between generation and benchmarking as a quality gate.
- Stage 04 also emits generation-side diagnostics from reference-linked provenance.
- Novelty claims must be checked against `references/reference_matrix.md`.
- Canonical baseline backbone: `efficientnet_b0`. VGG16 is a robustness axis.
- Stage 02 produces `policy_spec.json`; stage 03 produces `synthetic_manifest.json`.
- Multi-seed runs (--seeds) are required for paper reporting.
- `HARD_CLASSES = [eosinophil, monocyte]` is the canonical scarce-class setting for auxiliary robustness studies.
- `eval_tta_mode` supports `none`, `hflip`, `fivecrop`, and `hflip_fivecrop` for Putzu-style test-time augmentation baselines.
