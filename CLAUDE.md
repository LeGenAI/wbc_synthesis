# CLAUDE.md

This file provides repository guidance for coding agents working in this project.

## Project Status

This repository is no longer a single linear experiment.

It contains:

- a preserved `legacy` research history
- a new `mainline` scaffold for the next canonical research cycle
- a separate `references/` archive for source-grounded literature review and novelty framing

The current canonical interpretation of the project is:

> Which `generation policy` creates a `synthetic pool` that improves `held-out domain utility` in WBC classification?

This is not the same question as the original repo setup.

## Canonical Research Direction

The current mainline direction is defined by:

- `.claude/docs/research_audit.md`
- `.claude/docs/research_redesign_rq.md`
- `references/reference_matrix.md`

The important shift is:

1. The old question was:
   `Does SDXL-LoRA synthetic augmentation improve WBC classification in general?`
2. The current question is:
   `Which generation policy improves leakage-safe held-out domain utility, especially on hard classes?`

### Mainline assumptions

- The next research cycle is a `generation-policy redesign` cycle.
- `LODO/selective subset` remains as an evaluation frame, not the main intervention.
- `boundary-aware V2` is preserved as a support branch and failure/lesson branch, not the canonical main story.
- Strong novelty claims must be grounded in `references/reference_matrix.md` first.

## Important Correction

The repo originally started from an `8-class` single-domain setting:

- basophil
- eosinophil
- erythroblast
- ig
- lymphocyte
- monocyte
- neutrophil
- platelet

That is still historically true for the earliest branch.

However, the current canonical multi-domain research line is primarily centered on the `5-class` WBC setting used throughout the multi-domain / LODO / selective-synth / boundary-aware branches:

- basophil
- eosinophil
- lymphocyte
- monocyte
- neutrophil

Agents should not assume that the initial 8-class setup is the current paper target.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login
# Kaggle credentials at ~/.kaggle/kaggle.json when needed
```

Platform notes:

- Apple Silicon MPS is supported in several legacy scripts
- legacy training helpers avoid some fp16 / subprocess behaviors on macOS

## Directory Intent

### `scripts/legacy/`

Historical experiment code, preserved for reproducibility and audit.

- `phase_00_07_initial_pipeline/`
  original single-domain SDXL-LoRA augmentation pipeline
- `phase_08_17_domain_gap_multidomain/`
  domain-gap recognition and multidomain transition
- `phase_18_32_generation_ablation/`
  generation sweeps and ablations
- `phase_33_40_selective_synth_lodo/`
  diverse generation, selective subsets, LODO benchmark
- `phase_41_61_boundary_v2/`
  contextual / boundary-aware generation branch
- `shared_support/`
  download helpers and shared training backbones

No new canonical research code should be added under `legacy/`.

### `scripts/mainline/`

Forward-looking canonical pipeline for the next paper cycle.

Stage order follows the supplementary-method structure:

1. `data/`
2. `generation/`
3. `scoring/`
4. `benchmark/`
5. `reporting/`

Current files are scaffolds and should be filled in this order.

### `references/`

This directory is not optional bookkeeping.

It is a research gate.

- `references/reference_matrix.md`
  reviewed literature matrix
- `references/reference_matrix.csv`
  sortable table form
- `references/references.bib`
  validated citations
- `references/sources/`
  local copies of papers, landing pages, official repo READMEs, and BibTeX

Novelty language should be checked against this directory before it is strengthened anywhere else.

## Legacy Interpretation

When reading old code, interpret it by phase, not as one coherent final pipeline.

### Phase 00-07

Question:
`Can single-domain synthetic augmentation improve standard classification and robustness?`

Status:
historical starting point only

### Phase 08-17

Question:
`Does domain gap exist, and can multidomain learning reduce it?`

Status:
important transition phase

### Phase 18-32

Question:
`Which generation knobs matter most?`

Status:
ablation knowledge, not final story

### Phase 33-40

Question:
`Which synthetic subset helps which held-out domain?`

Status:
strongest downstream-positive legacy branch

### Phase 41-61

Question:
`Can boundary-aware/context-preserving generation create more useful hard samples?`

Status:
important exploratory branch, but not current canonical claim

## Current Benchmark Policy

The main benchmark direction is:

- leakage-safe `LODO selective utility benchmark`

Preferred interpretation:

- evaluate whether synthetic pools help held-out domain performance
- emphasize hard-class rescue and held-out utility
- do not treat easy split gains as primary evidence

Priority held-out domains in the redesign documents:

- `Raabin`
- `AMC`

These are currently the most informative targets for hard-class rescue analysis.

### Current baseline decision

For the current mainline cycle, use `efficientnet_b0` as the canonical baseline backbone.

Reason:

- it is materially lighter on local Apple Silicon / MPS
- it already completed a full-data mainline sanity run
- the current cycle is about `generation policy`, not backbone ablation

Interpret `vgg16` as a deferred follow-up axis for later robustness checks, not a required baseline in the current phase.

## Mainline Design Rules

Agents working on new research code should preserve these constraints.

1. The primary objective is not image realism by itself.
   The objective is downstream utility.
2. The main intervention should be `generation policy`, not endless subset heuristics.
3. Selection logic can remain as evaluation support, but should not replace the main research variable.
4. `boundary-aware` metrics alone are not a paper claim.
5. Avoid strong novelty claims unless they are supportable from `references/`.

## Practical Coding Guidance

### Path model

Most legacy scripts derive project root from their own file location.

Because scripts were reorganized into `scripts/legacy/...`, do not assume old hardcoded `scripts/<number>_...` paths are still valid.

When adding new code:

- prefer robust `Path(__file__)`-based root discovery
- avoid embedding old flat `scripts/` references

### Where to put new work

- new canonical data normalization logic:
  `scripts/mainline/data/`
- new generation training / sampling logic:
  `scripts/mainline/generation/`
- scoring / manifest logic:
  `scripts/mainline/scoring/`
- benchmark runners:
  `scripts/mainline/benchmark/`
- figure/table/submission assembly:
  `scripts/mainline/reporting/`

### What not to do

- do not add new numbered root-level scripts
- do not treat all legacy branches as equally canonical
- do not summarize the project as only “8-class robustness with SDXL-LoRA”
- do not write novelty claims without checking `references/reference_matrix.md`

## Recommended Read Order

If you need to understand the research before editing code, read in this order:

1. `.claude/docs/research_audit.md`
2. `references/reference_matrix.md`
3. `.claude/docs/research_redesign_rq.md`
4. `scripts/README.md`
5. the relevant `scripts/legacy/phase_*` directory

## Minimal Summary for Agents

If you remember only one thing, remember this:

The repo's historical code began as an `8-class single-domain augmentation` project, but the current canonical research direction is a `5-class multi-domain generation-policy` project whose main proof target is `leakage-safe held-out domain utility`, not generic realism or generic augmentation gains.
