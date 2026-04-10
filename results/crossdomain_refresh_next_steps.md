# Cross-Domain Refresh Next Steps

## Current Position

The refreshed LoRA and `cross_only` pipeline now shows a split outcome.

- `monocyte` is strong enough to justify scaling.
- `eosinophil` is still unstable under aggressive denoise.

This means the next batch should not be symmetric across classes.

## Immediate Action Plan

### 1. Monocyte Expansion

Goal:
Scale the current `cross_only` probe to a larger run without changing the recipe too much.

Recommended settings:

- keep `cross_domain_mode=cross_only`
- keep `n_seeds=1` initially
- increase `n_per_domain` from `2` to `6` or `8`
- keep `ds=0.25, 0.35, 0.45`
- add explicit filtering for weak Raabin reference images before generation

Reason:
The current monocyte result is already good enough that the main question is reproducibility at larger scale, not whether cross-domain generation works at all.

### 2. Eosinophil Safe-Zone Rerun

Goal:
Retest eosinophil in a constrained operating region instead of expanding diversity immediately.

Recommended settings:

- keep `cross_domain_mode=cross_only`
- restrict denoise to `0.25` and `0.35`
- exclude `0.45`
- keep the refreshed LoRA weights
- start with the same `n_per_domain=2` for a clean controlled comparison

Reason:
The current run shows that eosinophil is usable at low and medium denoise, but collapses at `0.45`.

### 3. Reference-Side Analysis

Goal:
Identify weak source references before the next large run.

Recommended checks:

- monocyte: inspect Raabin references first
- eosinophil: inspect AMC and MLL23 source references first
- rank reference images by downstream CNN confidence and SSIM distribution

Reason:
The current failure patterns depend more on source reference quality than on target prompt domain alone.

## Suggested Order

1. Run a constrained eosinophil rerun without `ds=0.45`.
2. In parallel, score monocyte Raabin source references and remove weak ones.
3. Launch a larger monocyte `cross_only` batch.
4. Compare the new eosinophil rerun against the current probe before scaling eosinophil further.

## Decision Rule

- If eosinophil `ds<=0.35` rises above the historical same-domain baseline, promote cross-domain eosinophil to large-scale generation.
- If eosinophil still stays below the same-domain baseline, keep cross-domain eosinophil as a targeted augmentation source rather than a full replacement pipeline.
- If monocyte stays above `0.93` CNN accuracy after scale-up, proceed to utility-aware subset selection on the larger cross-domain pool.
