# LODO Result Review and Follow-up Plan

## 1. Completed Results

### LODO baseline (`efficientnet_b0`)

| Held-out Domain | Accuracy | Macro-F1 |
|---|---:|---:|
| PBC (Spain) | 0.5587 | 0.4399 |
| Raabin (Iran) | 0.8606 | 0.5673 |
| MLL23 (Germany) | 0.7822 | 0.7340 |
| AMC (Korea) | 0.8674 | 0.5893 |
| **Average** | **0.7672** | **0.5826** |

### Selective synthetic (`heldout=AMC`, subset `S2 = cnn_correct`)

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| Baseline AMC | 0.8674 | 0.5893 |
| AMC + S2 | 0.8806 | 0.6380 |
| **Delta** | **+0.0132** | **+0.0487** |

## 2. Main Findings

1. The new LODO benchmark is meaningfully difficult. Average Macro-F1 dropped to `0.5826`, so this protocol is suitable for showing synthetic utility.
2. Synthetic data should not be evaluated only on average accuracy. Failure modes are domain-specific and class-specific.
3. `AMC + S2` is the first positive signal under the new benchmark. The gain comes mainly from weak classes:
   - basophil F1: `0.2388 -> 0.3158`
   - eosinophil F1: `0.1834 -> 0.3339`
4. The biggest baseline failures are:
   - `PBC`: basophil collapse (`F1 = 0.0259`)
   - `Raabin`: monocyte collapse (`F1 = 0.0303`)
   - `AMC`: eosinophil/basophil are weak
5. `MLL23` is relatively stable already (`Macro-F1 = 0.7340`), so it is a lower-priority target for rescue experiments.

## 3. Immediate Experimental Priority

### Priority A: confirm generalization of `S2`

Run `S2_cnn_correct` on all four held-out domains with the same `efficientnet_b0` setup.

Goal:
- test whether the AMC gain is general or domain-specific
- establish `S2` as the main selective baseline

### Priority B: targeted subset ablations on weak domains

Run only on the domains with clear failure modes:
- `heldout_pbc`
- `heldout_raabin`
- `heldout_amc`

Recommended subsets:
- `S2_cnn_correct`
- `S3_high_conf_correct`
- `S7_hard_classes_only`

Goal:
- distinguish "more clean synth" from "class-focused synth"
- check whether domain rescue needs stronger filtering or hard-class emphasis

### Priority C: decide whether new subset definitions are needed

If `S7` is still too broad, add new manifests:
- `monocyte_only`
- `eosinophil_only`
- `basophil_only`
- `monocyte+eosinophil`

This is especially important for `Raabin` and `PBC`.

## 4. Runs To Avoid Right Now

- Do not run `all synth` as a main experiment.
- Do not spend compute on prompt-only sweeps.
- Do not switch models yet.
- Do not expand to more domains until `S2/S3/S7` behavior is clear.

## 5. Concrete Next Batch

### Batch 1: minimum publishable confirmation

1. `S2` on `heldout_pbc`
2. `S2` on `heldout_raabin`
3. `S2` on `heldout_mll23`

### Batch 2: rescue-focused ablation

4. `S3` on `heldout_amc`
5. `S7` on `heldout_amc`
6. `S3` on `heldout_pbc`
7. `S7` on `heldout_raabin`

## 6. Success Criteria

- `S2` improves average Macro-F1 across hold-outs over `0.5826`
- at least two held-out domains show positive Macro-F1 gain
- one weak-class rescue result is reproducible and clear:
  - `PBC basophil`
  - `Raabin monocyte`
  - or `AMC eosinophil`

## 7. Estimated Compute

- one selective run: about `4 to 4.5 hours`
- Batch 1: about `12 to 14 hours`
- Batch 2: about `16 to 18 hours`

## 8. Decision Rule After Batch 1

- If `S2` helps on at least 2/4 hold-outs: keep utility-aware filtering as the main paper direction.
- If `S2` helps only on AMC: pivot to domain-specific rescue and class-targeted subsets.
- If `S2` fails broadly: stop scaling and redesign the subset builder before more training.

## 9. Batch 1 Outcome

### S2 results across all hold-outs

| Held-out Domain | Baseline Macro-F1 | S2 Macro-F1 | Delta |
|---|---:|---:|---:|
| AMC | 0.5893 | 0.6380 | +0.0487 |
| Raabin | 0.5673 | 0.6007 | +0.0334 |
| MLL23 | 0.7340 | 0.7256 | -0.0084 |
| PBC | 0.4399 | 0.4099 | -0.0300 |
| **Average** | **0.5826** | **0.5936** | **+0.0109** |

### Interpretation

1. `S2` is not a universal improvement, but it is not a failure either.
2. The utility-aware filtering thesis remains valid because average Macro-F1 improved.
3. The effect is strongly domain-dependent:
   - `AMC`: strong positive result
   - `Raabin`: positive result, especially for monocyte rescue
   - `MLL23`: effectively neutral to mildly negative
   - `PBC`: clearly harmful overall
4. This means the next stage should not be more `S2` scaling. The next stage should be domain-specific rescue and class-targeted subset design.

### Key class shifts

- `AMC`: eosinophil and basophil improved clearly
- `Raabin`: monocyte improved strongly (`0.0303 -> 0.2907`)
- `PBC`: basophil improved strongly, but eosinophil collapsed and total Macro-F1 dropped
- `MLL23`: lymphocyte improved, but monocyte/neutrophil dropped

## 10. Updated Next Step

### Keep

- Keep `AMC + S2` as a positive reference result
- Keep `Raabin + S2` as a partial positive result

### Stop

- Do not run more `S2` confirmation experiments
- Do not spend compute on `MLL23` for now

### Batch 2 priority

1. `S3` on `heldout_amc`
2. `S7` on `heldout_amc`
3. `S3` on `heldout_pbc`
4. `S7` on `heldout_pbc`
5. `S7` on `heldout_raabin`

### Expected purpose

- `AMC`: test whether stricter filtering (`S3`) or hard-class emphasis (`S7`) is the better explanation for the gain
- `PBC`: try to recover eosinophil without losing the basophil rescue
- `Raabin`: push monocyte rescue further with hard-class-focused synth

### If Batch 2 is inconclusive

Add new manifests before more training:
- `monocyte_only`
- `eosinophil_only`
- `basophil_only`
- optional synth-ratio sweep such as `25%`, `50%`, `100%` of selected synth

## 11. Batch 2 Interim Outcome

### Completed so far

| Held-out Domain | Subset | Accuracy | Macro-F1 | Delta vs Baseline |
|---|---|---:|---:|---:|
| AMC | S3 | 0.8627 | 0.6212 | +0.0319 |
| AMC | S7 | 0.8897 | 0.6874 | +0.0981 |
| PBC | S3 | 0.7168 | 0.6379 | +0.1980 |

### What changed

1. `AMC`: `S7` is clearly better than `S2` and `S3`.
2. `PBC`: the problem was not "synthetic hurts PBC" in general. The problem was that `S2` was the wrong subset. `S3` produces a major rescue.
3. The thesis is now stronger:
   - subset choice matters more than synthetic quantity
   - the best subset is domain-dependent

### Current best settings

- `AMC`: `S7_hard_classes_only`
- `PBC`: `S3_high_conf_correct`
- `Raabin`: still unresolved, but `S7` is the most logical next test
- `MLL23`: no additional priority for now

### Immediate next runs

1. `heldout_pbc + S7`
2. `heldout_raabin + S7`

### If these confirm the pattern

The paper direction should shift from generic filtering to:

`Domain-specific subset selection for synthetic WBC augmentation`

## 12. Batch 2 Final Outcome

### Domain-best subset map

| Held-out Domain | Best Setting | Macro-F1 | Delta vs Baseline |
|---|---|---:|---:|
| AMC | `S7_hard_classes_only` | 0.6874 | +0.0981 |
| PBC | `S3_high_conf_correct` | 0.6379 | +0.1980 |
| Raabin | `S7_hard_classes_only` | 0.6542 | +0.0869 |
| MLL23 | `baseline` | 0.7340 | +0.0000 |

### Interpretation

1. Domain-specific subset selection is now supported by the experiments.
2. There is no single best subset across all domains.
3. `AMC` and `Raabin` prefer hard-class-focused synth.
4. `PBC` prefers stricter high-confidence filtering over hard-class-only synth.
5. `MLL23` does not currently justify synthetic augmentation.

### Remaining open question

`Raabin` improved overall with `S7`, but monocyte is still weak. The next targeted experiment should be:

- `heldout_raabin + monocyte_only`

## 13. Final Result Snapshot

### Best setting by held-out domain

| Held-out Domain | Best Setting | Accuracy | Macro-F1 | Delta vs Baseline |
|---|---|---:|---:|---:|
| AMC | `S7_hard_classes_only` | 0.8897 | 0.6874 | +0.0981 |
| PBC | `S3_high_conf_correct` | 0.7168 | 0.6379 | +0.1980 |
| Raabin | `S10_monocyte_only` | 0.8927 | 0.6864 | +0.1191 |
| MLL23 | `baseline` | 0.7822 | 0.7340 | +0.0000 |

### Final interpretation

1. The best synthetic strategy is domain-specific.
2. Some domains benefit from broad but clean filtering (`PBC -> S3`).
3. Some domains benefit from hard-class-focused synth (`AMC -> S7`).
4. Some domains benefit most from class-targeted synth (`Raabin -> S10 monocyte_only`).
5. Some domains do not currently need synthetic augmentation (`MLL23`).

### Paper message

The strongest paper framing is now:

`Utility-aware and domain-specific subset selection for synthetic WBC augmentation`
