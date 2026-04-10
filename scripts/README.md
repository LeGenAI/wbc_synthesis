# Scripts Layout

## Principle

`scripts/` is now split into two lanes:

- `scripts/legacy/`
  historical experiment code preserved for reproducibility and audit
- `scripts/mainline/`
  forward-looking canonical pipeline aligned to the paper-submission supplementary structure

## Legacy layout

- `scripts/legacy/phase_00_07_initial_pipeline/`
  original single-domain SDXL-LoRA augmentation pipeline from `README.md`
- `scripts/legacy/phase_08_17_domain_gap_multidomain/`
  domain-gap recognition and multidomain transition
- `scripts/legacy/phase_18_32_generation_ablation/`
  generation ablations and diversity/strength sweeps
- `scripts/legacy/phase_33_40_selective_synth_lodo/`
  selective synthesis and LODO benchmark branch
- `scripts/legacy/phase_41_61_boundary_v2/`
  boundary-aware V2 and hybrid-manifest branch
- `scripts/legacy/shared_support/`
  utility scripts, download helpers, and shared training backbones

## Mainline layout

- `scripts/mainline/data/`
  dataset normalization, split control, and manifest creation
- `scripts/mainline/generation/`
  generation-policy training and synthetic pool creation
- `scripts/mainline/scoring/`
  per-image scoring, preservation diagnostics, and policy manifests
- `scripts/mainline/benchmark/`
  leakage-safe utility benchmark and training/evaluation entry points
- `scripts/mainline/reporting/`
  paper figures, tables, and submission package assembly

## Naming rule for new work

New mainline scripts should follow the supplementary-method order rather than the old global numbering.

- data
- generation
- scoring
- benchmark
- reporting

Do not add new research code under `scripts/legacy/`.
