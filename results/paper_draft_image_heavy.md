# Paper Draft: Utility-Aware and Domain-Specific Subset Selection for Synthetic WBC Augmentation

## Title Candidates

1. **Utility-Aware and Domain-Specific Subset Selection for Synthetic White Blood Cell Augmentation**
2. **Synthetic Data Helps Only When Selected Right: Domain-Specific Augmentation for WBC Classification**
3. **Not All Synthetic Cells Are Useful: Domain-Specific Subset Selection for Robust WBC Classification**
4. **Selective Synthetic Augmentation for Domain-Generalized White Blood Cell Classification**

## One-Sentence Claim

Synthetic WBC augmentation is effective only when the synthetic subset is selected according to downstream utility and target-domain failure mode, rather than added indiscriminately.

## Abstract

We study whether synthetic white blood cell (WBC) images improve multi-class WBC classification under domain shift. While prior experiments in this codebase showed that naive synthetic augmentation can fail or even degrade performance, we hypothesized that the true question is not whether synthetic data helps in general, but which synthetic subset helps for which target domain. To test this, we introduced a leave-one-domain-out (LODO) benchmark across four WBC domains and evaluated utility-aware subset selection strategies built from a large-scale synthetic corpus of 6,720 generated images. We compared baseline real-only training against multiple subset policies, including CNN-correct filtering, high-confidence filtering, hard-class-focused subsets, and class-targeted subsets. The baseline LODO setting yielded an average Macro-F1 of 0.5826, confirming that the benchmark is meaningfully challenging. In contrast, domain-specific subset selection improved the best per-domain Macro-F1 to 0.6874 on AMC (+0.0981), 0.6379 on PBC (+0.1980), and 0.6864 on Raabin (+0.1191), while MLL23 required no synthetic augmentation. Importantly, the optimal policy differed by held-out domain: AMC benefited most from hard-class-focused synthesis, PBC from high-confidence filtering, and Raabin from monocyte-only targeted synthesis. These results show that synthetic augmentation for medical image classification should be treated as a utility-aware and domain-specific data selection problem, not a quantity maximization problem.

## Contributions

1. We define a harder LODO benchmark for WBC classification that avoids ceiling effects present in easier validation splits.
2. We show that naive or universal synthetic augmentation policies are suboptimal and often harmful.
3. We demonstrate that the best synthetic subset depends on the held-out domain.
4. We show that class-targeted synthetic augmentation can rescue a severe domain-specific failure mode, especially for Raabin monocyte classification.

## Main Table

| Held-out Domain | Baseline Acc | Baseline Macro-F1 | Best Setting | Best Acc | Best Macro-F1 | Delta Macro-F1 |
|---|---:|---:|---|---:|---:|---:|
| AMC | 0.8674 | 0.5893 | `S7_hard_classes_only` | 0.8897 | 0.6874 | +0.0981 |
| PBC | 0.5587 | 0.4399 | `S3_high_conf_correct` | 0.7168 | 0.6379 | +0.1980 |
| Raabin | 0.8606 | 0.5673 | `S10_monocyte_only` | 0.8927 | 0.6864 | +0.1191 |
| MLL23 | 0.7822 | 0.7340 | `baseline` | 0.7822 | 0.7340 | +0.0000 |
| **Average** | **0.7672** | **0.5826** | **domain-best policy** | **0.8203** | **0.6864** | **+0.1038** |

## Subset Comparison Table

| Held-out Domain | Baseline | S2 `cnn_correct` | S3 `high_conf_correct` | S7 `hard_classes_only` | S10 `monocyte_only` |
|---|---:|---:|---:|---:|---:|
| AMC | 0.5893 | 0.6380 | 0.6212 | **0.6874** | - |
| PBC | 0.4399 | 0.4099 | **0.6379** | 0.4375 | - |
| Raabin | 0.5673 | 0.6007 | - | 0.6542 | **0.6864** |
| MLL23 | **0.7340** | 0.7256 | - | - | - |

## Result Section Draft

### 3.1 LODO benchmark reveals meaningful domain shift

Under the leave-one-domain-out benchmark, the real-only baseline achieved an average Macro-F1 of 0.5826 across four held-out domains. This is substantially lower than earlier easy-split evaluations and confirms that the new protocol exposes genuine domain shift. Among held-out domains, PBC was the most difficult (Macro-F1 = 0.4399), followed by Raabin (0.5673) and AMC (0.5893), while MLL23 remained relatively stable (0.7340). These results justify evaluating synthetic augmentation in LODO rather than in easier in-domain validation settings.

### 3.2 A single synthetic policy does not generalize across domains

We first tested `S2`, a utility-aware but domain-agnostic subset that retains only CNN-correct synthetic images. Although `S2` improved the overall average Macro-F1 from 0.5826 to 0.5936, its effect was highly domain-dependent. It improved AMC and Raabin, but slightly degraded MLL23 and clearly harmed PBC. This finding rejects the assumption that a single “clean” synthetic subset is universally optimal.

### 3.3 Best subset selection is domain-specific

Further ablations showed that each domain favored a different synthetic policy. AMC benefited most from `S7_hard_classes_only`, reaching a Macro-F1 of 0.6874. PBC benefited most from stricter filtering with `S3_high_conf_correct`, reaching 0.6379. Raabin improved most with `S10_monocyte_only`, reaching 0.6864. In contrast, MLL23 showed no benefit from synthetic augmentation, suggesting that synthetic data should not be applied indiscriminately even when it is available at scale.

### 3.4 Class-targeted synthesis can rescue domain-specific weak classes

The strongest class-specific result was observed in Raabin. In the baseline setting, monocyte F1 was only 0.0303, indicating near-collapse. Hard-class-focused synthesis (`S7`) increased monocyte F1 to 0.2191, and class-targeted monocyte-only synthesis (`S10`) further improved it to 0.3004. At the same time, `S10` also improved eosinophil F1 from 0.3899 to 0.6178 and raised overall Macro-F1 to 0.6864. This suggests that carefully targeted synthetic augmentation can address specific morphological blind spots without requiring global synthetic expansion.

### 3.5 Implication

Taken together, our results show that synthetic WBC augmentation is best framed as a utility-aware subset selection problem conditioned on target-domain failure modes. Quantity alone is not predictive of benefit; the key variable is whether the selected synthetic subset matches the weakness of the target domain.

## Figure Plan

### Figure 1. Overall pipeline

Real WBC domains -> large-scale synthetic generation -> utility-aware subset builder -> LODO training -> domain-specific subset selection.

### Figure 2. LODO benchmark difficulty

Bar chart of baseline Macro-F1 by held-out domain: PBC, Raabin, AMC, MLL23.

### Figure 3. Domain-specific subset effect

Grouped bars comparing baseline vs `S2/S3/S7/S10` for each held-out domain.

### Figure 4. Class-specific rescue on Raabin

Per-class F1 comparison for baseline vs `S7` vs `S10`.

### Figure 5. Qualitative synthetic examples

Representative domain-specific generated examples across AMC, PBC, and Raabin.

## Quantitative Figures

### Figure 2. LODO baseline difficulty

![Figure 2 - LODO Baseline Macro-F1](/Users/imds/Desktop/wbc_synthesis/results/paper_figures/fig02_lodo_baseline_macro_f1.png)

### Figure 3. Subset ablation across held-out domains

![Figure 3 - Subset Ablation](/Users/imds/Desktop/wbc_synthesis/results/paper_figures/fig03_subset_ablation_macro_f1.png)

### Figure 4. Best domain-specific policy vs baseline

![Figure 4 - Best Policy by Domain](/Users/imds/Desktop/wbc_synthesis/results/paper_figures/fig04_best_setting_by_domain.png)

### Figure 5. Raabin per-class rescue

![Figure 5 - Raabin Per-class Rescue](/Users/imds/Desktop/wbc_synthesis/results/paper_figures/fig05_raabin_per_class_rescue.png)

### Figure 6. AMC and PBC per-class best-policy comparison

![Figure 6 - AMC and PBC Best Per-class](/Users/imds/Desktop/wbc_synthesis/results/paper_figures/fig06_amc_pbc_best_per_class.png)

## Table Plan

### Table 1

Dataset and benchmark summary: number of real domains, number of classes, number of synthetic images, subset definitions.

### Table 2

LODO baseline performance by held-out domain.

### Table 3

Subset ablation results by held-out domain.

### Table 4

Per-class F1 comparison for the best domain-specific setting.

## Qualitative Figures

### AMC examples

AMC was best improved by `S7_hard_classes_only`, especially for basophil and eosinophil.

![AMC Basophil Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/basophil/grid_AMC.png)

![AMC Eosinophil Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/eosinophil/grid_AMC.png)

![AMC Monocyte Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/monocyte/grid_AMC.png)

![AMC Neutrophil Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/neutrophil/grid_AMC.png)

### PBC examples

PBC was not helped by generic clean filtering (`S2`), but was strongly improved by the stricter high-confidence subset `S3`.

![PBC Basophil Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/basophil/grid_PBC.png)

![PBC Eosinophil Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/eosinophil/grid_PBC.png)

![PBC Monocyte Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/monocyte/grid_PBC.png)

![PBC Neutrophil Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/neutrophil/grid_PBC.png)

### Raabin examples

Raabin responded best to class-targeted monocyte-only augmentation, indicating that the main bottleneck was not global data scarcity but a highly specific class failure mode.

![Raabin Monocyte Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/monocyte/grid_Raabin.png)

![Raabin Eosinophil Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/eosinophil/grid_Raabin.png)

![Raabin Basophil Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/basophil/grid_Raabin.png)

### MLL23 examples

MLL23 remained relatively stable and did not require synthetic augmentation in the current setup.

![MLL23 Lymphocyte Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/lymphocyte/grid_MLL23.png)

![MLL23 Neutrophil Grid](/Users/imds/Desktop/wbc_synthesis/results/diverse_generation/neutrophil/grid_MLL23.png)

## Discussion Draft

The main lesson from this study is that synthetic medical image augmentation should not be optimized for scale alone. Our large synthetic corpus was necessary, but it was not sufficient. Gains only emerged once synthetic samples were filtered according to downstream utility and target-domain weakness. This explains why earlier naive augmentation experiments in the codebase produced weak or negative results: they treated synthetic data as a bulk additive resource rather than a conditional intervention. The practical implication is simple. Before scaling generation further, one should first identify the held-out domain, its weak classes, and the subset policy most aligned with those failure modes.

## Next Writing Step

1. Convert this draft into a paper skeleton with `Introduction`, `Methods`, `Results`, and `Discussion`.
2. Add actual bar plots for Figure 2, Figure 3, and Figure 4 from the final result JSON files.
3. Replace some qualitative grids with tighter crop panels if journal layout becomes crowded.
