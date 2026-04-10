# Cross-Domain Refresh Gallery

This gallery summarizes representative samples from the refreshed `cross_only` generation runs for `monocyte` and `eosinophil`.

- Comparison note: [crossdomain_refresh_comparison.md](/Users/imds/Desktop/wbc_synthesis/results/crossdomain_refresh_comparison.md)
- Monocyte report: [report.json](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/monocyte/report.json)
- Eosinophil report: [report.json](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/eosinophil/report.json)

## Monocyte: Strong Cross-Domain Cases

### `PBC -> Raabin`, `ds=0.25`, `standard`, `conf=0.9337`
![monocyte pbc to raabin ds025 standard](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_a_pbc/to_domain_b_raabin/ds025/tpl0_standard/0000_s0.png)

### `PBC -> AMC`, `ds=0.25`, `standard`, `conf=0.9340`
![monocyte pbc to amc ds025 standard](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_a_pbc/to_domain_e_amc/ds025/tpl0_standard/0000_s0.png)

### `MLL23 -> PBC`, `ds=0.25`, `oil_immersion`, `conf=0.9348`
![monocyte mll23 to pbc ds025 oil](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_c_mll23/to_domain_a_pbc/ds025/tpl1_oil_immersion/0000_s0.png)

### `MLL23 -> AMC`, `ds=0.25`, `oil_immersion`, `conf=0.9349`
![monocyte mll23 to amc ds025 oil](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_c_mll23/to_domain_e_amc/ds025/tpl1_oil_immersion/0000_s0.png)

### `PBC -> AMC`, `ds=0.45`, `oil_immersion`, `conf=0.9351`
![monocyte pbc to amc ds045 oil](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_a_pbc/to_domain_e_amc/ds045/tpl1_oil_immersion/0001_s0.png)

### `MLL23 -> Raabin`, `ds=0.45`, `oil_immersion`, `conf=0.9357`
![monocyte mll23 to raabin ds045 oil](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_c_mll23/to_domain_b_raabin/ds045/tpl1_oil_immersion/0000_s0.png)

## Monocyte: Weak Cases

These failures are concentrated in `Raabin` source references.

### `Raabin -> PBC`, `ds=0.35`, `oil_immersion`, `incorrect`, `conf=0.4992`
![monocyte raabin to pbc ds035 oil fail](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_b_raabin/to_domain_a_pbc/ds035/tpl1_oil_immersion/0001_s0.png)

### `Raabin -> MLL23`, `ds=0.35`, `oil_immersion`, `incorrect`, `conf=0.5137`
![monocyte raabin to mll23 ds035 oil fail](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_b_raabin/to_domain_c_mll23/ds035/tpl1_oil_immersion/0001_s0.png)

### `Raabin -> AMC`, `ds=0.25`, `oil_immersion`, `incorrect`, `conf=0.5193`
![monocyte raabin to amc ds025 oil fail](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_b_raabin/to_domain_e_amc/ds025/tpl1_oil_immersion/0001_s0.png)

### `Raabin -> AMC`, `ds=0.25`, `clinical_hematology`, `incorrect`, `conf=0.6197`
![monocyte raabin to amc ds025 clinical fail](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/monocyte/ref_domain_b_raabin/to_domain_e_amc/ds025/tpl2_clinical_hematology/0001_s0.png)

## Eosinophil: Stable Cases

These examples come from the low- and medium-denoise region, where eosinophil remains usable.

### `PBC -> Raabin`, `ds=0.25`, `standard`, `conf=0.9233`
![eosinophil pbc to raabin ds025 standard](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_a_pbc/to_domain_b_raabin/ds025/tpl0_standard/0000_s0.png)

### `AMC -> PBC`, `ds=0.25`, `oil_immersion`, `conf=0.9365`
![eosinophil amc to pbc ds025 oil](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_e_amc/to_domain_a_pbc/ds025/tpl1_oil_immersion/0000_s0.png)

### `AMC -> Raabin`, `ds=0.25`, `oil_immersion`, `conf=0.9364`
![eosinophil amc to raabin ds025 oil](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_e_amc/to_domain_b_raabin/ds025/tpl1_oil_immersion/0000_s0.png)

### `AMC -> MLL23`, `ds=0.25`, `oil_immersion`, `conf=0.9357`
![eosinophil amc to mll23 ds025 oil](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_e_amc/to_domain_c_mll23/ds025/tpl1_oil_immersion/0000_s0.png)

### `AMC -> PBC`, `ds=0.35`, `clinical_hematology`, `conf=0.9341`
![eosinophil amc to pbc ds035 clinical](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_e_amc/to_domain_a_pbc/ds035/tpl2_clinical_hematology/0000_s0.png)

### `AMC -> Raabin`, `ds=0.35`, `cytology`, `conf=0.9344`
![eosinophil amc to raabin ds035 cytology](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_e_amc/to_domain_b_raabin/ds035/tpl3_cytology/0000_s0.png)

## Eosinophil: Failure Cases

Most failures are concentrated at `ds=0.45`.

### `MLL23 -> PBC`, `ds=0.35`, `clinical_hematology`, `incorrect`, `conf=0.3764`
![eosinophil mll23 to pbc ds035 clinical fail](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_c_mll23/to_domain_a_pbc/ds035/tpl2_clinical_hematology/0001_s0.png)

### `MLL23 -> Raabin`, `ds=0.45`, `cytology`, `incorrect`, `conf=0.3848`
![eosinophil mll23 to raabin ds045 cytology fail](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_c_mll23/to_domain_b_raabin/ds045/tpl3_cytology/0000_s0.png)

### `MLL23 -> Raabin`, `ds=0.45`, `standard`, `incorrect`, `conf=0.3958`
![eosinophil mll23 to raabin ds045 standard fail](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_c_mll23/to_domain_b_raabin/ds045/tpl0_standard/0001_s0.png)

### `MLL23 -> AMC`, `ds=0.45`, `oil_immersion`, `incorrect`, `conf=0.5032`
![eosinophil mll23 to amc ds045 oil fail](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_c_mll23/to_domain_e_amc/ds045/tpl1_oil_immersion/0001_s0.png)

### `AMC -> MLL23`, `ds=0.45`, `clinical_hematology`, `incorrect`, `conf=0.5395`
![eosinophil amc to mll23 ds045 clinical fail](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_e_amc/to_domain_c_mll23/ds045/tpl2_clinical_hematology/0001_s0.png)

### `AMC -> Raabin`, `ds=0.45`, `clinical_hematology`, `incorrect`, `conf=0.5450`
![eosinophil amc to raabin ds045 clinical fail](/Users/imds/Desktop/wbc_synthesis/data/generated_diverse/eosinophil/ref_domain_e_amc/to_domain_b_raabin/ds045/tpl2_clinical_hematology/0001_s0.png)

## Summary

- `monocyte` is ready for a larger `cross_only` batch.
- `eosinophil` should be rerun with `ds=0.25` and `0.35` only.
- The gallery supports the same conclusion as the metrics: the refreshed pipeline is already usable, but not equally robust across classes.
