# Boundary-Aware Contextual WBC Generation V2

## Added Scripts

- `scripts/legacy/phase_41_61_boundary_v2/boundary_aware_utils.py`
  - shared prompts, domain tokens, mask extraction, CNN scoring, boundary score
- `scripts/legacy/phase_41_61_boundary_v2/42_preprocess_contextual_multidomain.py`
  - contextual `384x384` preprocessing branch under `data/processed_contextual_multidomain/`
- `scripts/legacy/phase_41_61_boundary_v2/43_build_contextual_masks.py`
  - center-biased heuristic cell masks under `data/processed_contextual_masks/`
- `scripts/legacy/phase_41_61_boundary_v2/44_contextual_lora_train.py`
  - contextual LoRA entrypoint for `contextual_multidomain_{class}`
- `scripts/legacy/phase_41_61_boundary_v2/45_background_aware_generate.py`
  - two-stage background-first inpaint + optional low-strength refine generation
- `scripts/legacy/phase_41_61_boundary_v2/46_boundary_subset_builder.py`
  - low-margin, cell-preserving, background-diversifying subset builder
- `scripts/legacy/phase_41_61_boundary_v2/47_boundary_lodo_train.py`
  - wrapper to evaluate V2 manifests with existing Script 36 LODO path
- `scripts/legacy/phase_41_61_boundary_v2/48_boundary_acceptance_review.py`
  - compares V2 generation and V2 LODO outputs against baseline / refreshed S7
- `scripts/legacy/phase_41_61_boundary_v2/49_boundary_v2_bootstrap.sh`
  - bootstrap runner for the V2 pipeline

## Output Layout

- Contextual data: `data/processed_contextual_multidomain/`
- Contextual masks: `data/processed_contextual_masks/`
- Contextual LoRA weights: `lora/weights/contextual_multidomain_*`
- V2 generations: `data/generated_boundary_v2/`
- V2 reports: `results/boundary_v2_generation/`
- V2 manifests: `results/boundary_selective_synth/`
- V2 LODO summary: `results/boundary_v2_lodo/`
- Acceptance review: `results/boundary_v2_acceptance/review.md`

## Command Order

```bash
python3 /Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_41_61_boundary_v2/42_preprocess_contextual_multidomain.py --domain all
python3 /Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_41_61_boundary_v2/43_build_contextual_masks.py
python3 /Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_41_61_boundary_v2/44_contextual_lora_train.py --all_classes
python3 /Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_41_61_boundary_v2/45_background_aware_generate.py --all_classes --cross_domain_mode cross_only
python3 /Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_41_61_boundary_v2/46_boundary_subset_builder.py
python3 /Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_41_61_boundary_v2/47_boundary_lodo_train.py
python3 /Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_41_61_boundary_v2/48_boundary_acceptance_review.py
```

## Acceptance Targets

- hard classes: `monocyte`, `eosinophil`
- near-boundary rate: at least `5x` current branch
- region gap: `cell_ssim - background_ssim >= 0.08`
- LODO utility: not worse than refreshed `S7` by more than `0.02` macro-F1 on hard domains
