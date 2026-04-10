# Mainline LODO Utility Benchmark Report

- Run name: `real_plus_synth__efficientnet_b0__heldout_raabin__tf0p02__seed42__synth_leakage_smoke_manifest`
- Mode: `real_plus_synth`
- Backbone: `efficientnet_b0`
- Held-out domain: `domain_b_raabin`
- Device: `mps`

## Metrics

| Split | Accuracy | Macro-F1 | Loss |
|---|---|---|---|
| val | 0.3089 | 0.2813 | 1.5625 |
| test | 0.146 | 0.1573 | 1.7035 |

## Leakage Guard

- excluded_for_heldout_domain: `1`
- synthetic_train_items_used: `1`

## Split Stats

### Train

| Domain | Class | Source | Count |
|---|---|---|---|
| domain_a_pbc | basophil | real | 19 |
| domain_a_pbc | basophil | synthetic | 1 |
| domain_a_pbc | eosinophil | real | 50 |
| domain_a_pbc | lymphocyte | real | 19 |
| domain_a_pbc | monocyte | real | 23 |
| domain_a_pbc | neutrophil | real | 53 |
| domain_c_mll23 | basophil | real | 10 |
| domain_c_mll23 | eosinophil | real | 39 |
| domain_c_mll23 | lymphocyte | real | 89 |
| domain_c_mll23 | monocyte | real | 40 |
| domain_c_mll23 | neutrophil | real | 115 |
| domain_e_amc | basophil | real | 2 |
| domain_e_amc | eosinophil | real | 16 |
| domain_e_amc | lymphocyte | real | 67 |
| domain_e_amc | monocyte | real | 16 |
| domain_e_amc | neutrophil | real | 137 |

### Val

| Domain | Class | Source | Count |
|---|---|---|---|
| domain_a_pbc | basophil | real | 244 |
| domain_a_pbc | eosinophil | real | 623 |
| domain_a_pbc | lymphocyte | real | 243 |
| domain_a_pbc | monocyte | real | 284 |
| domain_a_pbc | neutrophil | real | 666 |
| domain_c_mll23 | basophil | real | 123 |
| domain_c_mll23 | eosinophil | real | 490 |
| domain_c_mll23 | lymphocyte | real | 1106 |
| domain_c_mll23 | monocyte | real | 502 |
| domain_c_mll23 | neutrophil | real | 1434 |
| domain_e_amc | basophil | real | 25 |
| domain_e_amc | eosinophil | real | 200 |
| domain_e_amc | lymphocyte | real | 842 |
| domain_e_amc | monocyte | real | 198 |
| domain_e_amc | neutrophil | real | 1709 |

### Test

| Domain | Class | Source | Count |
|---|---|---|---|
| domain_b_raabin | basophil | real | 301 |
| domain_b_raabin | eosinophil | real | 1066 |
| domain_b_raabin | lymphocyte | real | 3609 |
| domain_b_raabin | monocyte | real | 795 |
| domain_b_raabin | neutrophil | real | 10862 |

## Test Per-class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| basophil | 0.0094 | 0.3289 | 0.0182 | 301 |
| eosinophil | 0.187 | 0.2045 | 0.1953 | 1066 |
| lymphocyte | 0.3127 | 0.2771 | 0.2938 | 3609 |
| monocyte | 0.1296 | 0.0893 | 0.1057 | 795 |
| neutrophil | 0.9044 | 0.0958 | 0.1733 | 10862 |

- Confusion matrix image: `/Users/imds/Desktop/wbc_synthesis/results/mainline/benchmark/efficientnet_b0/heldout_domain_b_raabin/real_plus_synth__efficientnet_b0__heldout_raabin__tf0p02__seed42__synth_leakage_smoke_manifest/confusion_matrix.png`
