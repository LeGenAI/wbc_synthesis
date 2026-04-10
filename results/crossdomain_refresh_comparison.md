# Cross-Domain Refresh Comparison

## Scope

This note compares the historical large-scale same-domain generation results with the newly rerun cross-domain-only generation results after the LoRA training and generation pipeline refresh.

- Historical same-domain reference: [overview.md](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/overview.md)
- New cross-domain reports:
  - [monocyte/report.json](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/monocyte/report.json)
  - [eosinophil/report.json](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/eosinophil/report.json)

## Headline Comparison

| Class | Setting | Sample Size | CNN Accuracy | CNN Conf. Mean | Sharpness Mean | SSIM Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Monocyte | Historical same-domain | 1,344 | 0.9040 | - | - | - |
| Monocyte | New cross-only | 288 | 0.9410 | 0.8921 | 7.56 | 0.9790 |
| Eosinophil | Historical same-domain | 1,344 | 0.8670 | - | - | - |
| Eosinophil | New cross-only | 288 | 0.8438 | 0.8487 | 9.44 | 0.9797 |

## Main Readout

1. The refreshed cross-domain pipeline is feasible for monocyte.
   Monocyte reached `94.10%` CNN accuracy in the new `cross_only` setting, which is above the historical same-domain result of `90.40%`.

2. Eosinophil remains the fragile class.
   Eosinophil reached `84.38%` in the new `cross_only` setting, slightly below the historical same-domain result of `86.70%`.

3. The comparison is directionally useful, but not perfectly apples-to-apples.
   The historical same-domain result used `n=1,344`, whereas the refreshed cross-domain probe used `n=288`.

## Cross-Only Breakdown

### Monocyte

- Aggregate:
  - `n=288`
  - `cnn_accuracy=0.9410`
  - `cnn_conf_mean=0.8921`
  - `sharpness_mean=7.56`
  - `ssim_mean=0.9790`
- By prompt domain:
  - `Raabin=1.0000`
  - `PBC=0.9306`
  - `AMC=0.9167`
  - `MLL23=0.9167`
- By reference domain:
  - `PBC=1.0000`
  - `AMC=1.0000`
  - `MLL23=1.0000`
  - `Raabin=0.7639`
- By denoise:
  - `0.25=0.9167`
  - `0.35=0.9062`
  - `0.45=1.0000`

Interpretation:
Monocyte cross-domain generation is already fairly stable. The main weakness is not the target prompt domain, but the source reference domain. In particular, Raabin references are materially weaker than the other three domains.

### Eosinophil

- Aggregate:
  - `n=288`
  - `cnn_accuracy=0.8438`
  - `cnn_conf_mean=0.8487`
  - `sharpness_mean=9.44`
  - `ssim_mean=0.9797`
- By prompt domain:
  - `AMC=0.8611`
  - `MLL23=0.8472`
  - `PBC=0.8472`
  - `Raabin=0.8194`
- By reference domain:
  - `Raabin=0.9167`
  - `PBC=0.8333`
  - `MLL23=0.8333`
  - `AMC=0.7917`
- By denoise:
  - `0.25=1.0000`
  - `0.35=0.9896`
  - `0.45=0.5417`

Interpretation:
Eosinophil is strongly sensitive to denoise strength. The model is still usable at `0.25` and `0.35`, but performance collapses at `0.45`. This indicates that the refreshed cross-domain pipeline is not yet robust enough for aggressive morphology/style perturbation in eosinophil.

## Pipeline-Level Interpretation

The refresh addressed two structural problems in the earlier pipeline.

1. Generation is now truly cross-domain.
   The new generation loop separates the source reference domain from the target prompt domain instead of reusing the same domain on both sides.

2. LoRA tuning is now healthier than before.
   The rerun saved intermediate checkpoints and ran validation prompts during training, which makes the resulting weights easier to inspect and less opaque than the earlier one-shot setup.

However, the refreshed pipeline is still class-dependent.

- Monocyte looks ready for a larger cross-domain batch.
- Eosinophil still needs a safer operating region.

## Recommended Next Step

1. Scale monocyte `cross_only` from the current probe (`n=288`) to a larger batch with the same recipe.
2. Rerun eosinophil `cross_only` with `denoise <= 0.35` only.
3. Add per-reference filtering for monocyte, especially for Raabin source references.
4. If the next eosinophil run is still unstable, move from pure class-level LoRA tuning to more constrained reference selection before increasing diversity axes again.
