# Retuned LoRA Variation Gallery

This gallery reviews how much variation the retuned cross-domain LoRA models introduce relative to the input reference images.

- Monocyte report: [report.json](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/monocyte/report.json)
- Eosinophil report: [report.json](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/eosinophil/report.json)
- Downstream utility review: [lodo_target_review.md](/Users/imds/Desktop/wbc_synthesis/results/crossdomain_next_batch/lodo_target_review.md)

## Summary

### Monocyte

- `n = 864`
- `CNN accuracy = 0.9919`
- `mean SSIM = 0.9783`
- strongest variation zone: `ds = 0.45`
- safe operating range: `0.25, 0.35, 0.45`

Interpretation:
The retuned monocyte LoRA preserves class identity while allowing visibly stronger deformation and style shift. The largest variation appears at `ds=0.45`, and it remains mostly correct.

### Eosinophil

- `n = 192`
- `CNN accuracy = 0.9948`
- `mean SSIM = 0.9901`
- strongest safe variation zone: `ds = 0.35`
- safe operating range: `0.25, 0.35`

Interpretation:
The retuned eosinophil LoRA is stable, but it achieves variation more conservatively than monocyte. It should be treated as a constrained cross-domain generator rather than an aggressive deformation model.

## Monocyte Cross-Domain Grids

These grids include the input references in the top row and the generated variants below them.

### `PBC -> AMC`
![monocyte pbc to amc grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/monocyte/grid_PBC_to_AMC.png)

### `PBC -> Raabin`
![monocyte pbc to raabin grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/monocyte/grid_PBC_to_Raabin.png)

### `MLL23 -> Raabin`
![monocyte mll23 to raabin grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/monocyte/grid_MLL23_to_Raabin.png)

### `Raabin -> AMC`
![monocyte raabin to amc grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/monocyte/grid_Raabin_to_AMC.png)

## Monocyte High-Variation Examples

These are low-SSIM but still CNN-correct examples from the retuned model.

### `Raabin -> MLL23`, `ds=0.45`, `clinical_hematology`, `SSIM=0.9041`, `conf=0.9331`
![monocyte high variation 1](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_b_raabin/to_domain_c_mll23/ds045/tpl2_clinical_hematology/0002_s0.png)

### `Raabin -> AMC`, `ds=0.45`, `clinical_hematology`, `SSIM=0.9043`, `conf=0.9356`
![monocyte high variation 2](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_b_raabin/to_domain_e_amc/ds045/tpl2_clinical_hematology/0002_s0.png)

### `Raabin -> PBC`, `ds=0.45`, `clinical_hematology`, `SSIM=0.9062`, `conf=0.9341`
![monocyte high variation 3](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_b_raabin/to_domain_a_pbc/ds045/tpl2_clinical_hematology/0002_s0.png)

### `MLL23 -> Raabin`, `ds=0.45`, `clinical_hematology`, `SSIM=0.9104`, `conf=0.9200`
![monocyte high variation 4](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_c_mll23/to_domain_b_raabin/ds045/tpl2_clinical_hematology/0004_s0.png)

## Eosinophil Cross-Domain Grids

These grids show that eosinophil variation is present, but still more conservative than monocyte.

### `AMC -> PBC`
![eosinophil amc to pbc grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/eosinophil/grid_AMC_to_PBC.png)

### `AMC -> Raabin`
![eosinophil amc to raabin grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/eosinophil/grid_AMC_to_Raabin.png)

### `PBC -> MLL23`
![eosinophil pbc to mll23 grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/eosinophil/grid_PBC_to_MLL23.png)

### `Raabin -> AMC`
![eosinophil raabin to amc grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/eosinophil/grid_Raabin_to_AMC.png)

## Eosinophil High-Variation Examples

These are the strongest safe variants among CNN-correct samples in the current safe denoise range.

### `PBC -> MLL23`, `ds=0.35`, `cytology`, `SSIM=0.9813`, `conf=0.9121`
![eosinophil high variation 1](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_a_pbc/to_domain_c_mll23/ds035/tpl3_cytology/0001_s0.png)

### `PBC -> Raabin`, `ds=0.35`, `standard`, `SSIM=0.9813`, `conf=0.9049`
![eosinophil high variation 2](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_a_pbc/to_domain_b_raabin/ds035/tpl0_standard/0001_s0.png)

### `PBC -> AMC`, `ds=0.35`, `cytology`, `SSIM=0.9814`, `conf=0.9117`
![eosinophil high variation 3](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_a_pbc/to_domain_e_amc/ds035/tpl3_cytology/0001_s0.png)

### `Raabin -> PBC`, `ds=0.35`, `standard`, `SSIM=0.9826`, `conf=0.9003`
![eosinophil high variation 4](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_b_raabin/to_domain_a_pbc/ds035/tpl0_standard/0001_s0.png)

## Readout

1. The retuned monocyte model now applies strong variation, including morphology-level perturbation and domain style transfer, without collapsing class identity.
2. The retuned eosinophil model applies a milder but still useful amount of variation, and should remain restricted to `ds <= 0.35`.
3. For paper framing, the most accurate claim is not just that the LoRA was retrained, but that it now produces controllable class-specific cross-domain variation with measurable downstream utility.
