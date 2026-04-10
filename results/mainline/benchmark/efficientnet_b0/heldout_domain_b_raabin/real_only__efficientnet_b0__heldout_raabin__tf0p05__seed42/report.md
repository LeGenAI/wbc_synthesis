# Mainline LODO Utility Benchmark Report

- Run name: `real_only__efficientnet_b0__heldout_raabin__tf0p05__seed42`
- Mode: `real_only`
- Backbone: `efficientnet_b0`
- Held-out domain: `domain_b_raabin`
- Device: `mps`

## Metrics

| Split | Accuracy | Macro-F1 | Loss |
|---|---|---|---|
| val | 0.3792 | 0.3293 | 1.5082 |
| test | 0.3335 | 0.2297 | 1.5148 |

## Leakage Guard

- excluded_for_heldout_domain: `0`
- synthetic_train_items_used: `0`

## Split Stats

### Train

| Domain | Class | Source | Count |
|---|---|---|---|
| domain_a_pbc | basophil | real | 49 |
| domain_a_pbc | eosinophil | real | 125 |
| domain_a_pbc | lymphocyte | real | 49 |
| domain_a_pbc | monocyte | real | 57 |
| domain_a_pbc | neutrophil | real | 133 |
| domain_c_mll23 | basophil | real | 25 |
| domain_c_mll23 | eosinophil | real | 98 |
| domain_c_mll23 | lymphocyte | real | 221 |
| domain_c_mll23 | monocyte | real | 100 |
| domain_c_mll23 | neutrophil | real | 287 |
| domain_e_amc | basophil | real | 5 |
| domain_e_amc | eosinophil | real | 40 |
| domain_e_amc | lymphocyte | real | 168 |
| domain_e_amc | monocyte | real | 40 |
| domain_e_amc | neutrophil | real | 342 |

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
| basophil | 0.0118 | 0.1462 | 0.0219 | 301 |
| eosinophil | 0.2204 | 0.2477 | 0.2332 | 1066 |
| lymphocyte | 0.292 | 0.6523 | 0.4034 | 3609 |
| monocyte | 0.2008 | 0.0604 | 0.0928 | 795 |
| neutrophil | 0.8295 | 0.2612 | 0.3973 | 10862 |

- Confusion matrix image: `/Users/imds/Desktop/wbc_synthesis/results/mainline/benchmark/efficientnet_b0/heldout_domain_b_raabin/real_only__efficientnet_b0__heldout_raabin__tf0p05__seed42/confusion_matrix.png`
