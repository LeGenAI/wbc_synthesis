# Boundary-Aware Generation Redesign

## Problem Diagnosis

The user's objection is correct.

The current pipeline was built to maximize `class-preserving plausibility`, not to generate useful boundary-near synthetic samples.

That bias comes from three concrete design choices in the current code:

1. [preprocess_multidomain.py](/Users/imds/Desktop/wbc_synthesis/scripts/legacy/shared_support/preprocess_multidomain.py)
   normalizes every image to `224x224` by aggressive center crop.
   This removes a large portion of the original background/domain context before training even starts.

2. [10_multidomain_lora_train.py](/Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_08_17_domain_gap_multidomain/10_multidomain_lora_train.py)
   gives near-identical prompts to all images from the same domain and class.
   This encourages the LoRA to preserve the same central morphology instead of learning a richer separation between cell content and background style.

3. [35_diverse_subset_builder.py](/Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_33_40_selective_synth_lodo/35_diverse_subset_builder.py)
   and downstream filtering implicitly reward `easy`, high-confidence samples.
   That is the opposite of a support-vector-like augmentation objective.

## New Target

The new objective should be:

- preserve the central cell enough to keep biological identity
- diversify the background and domain cues aggressively
- favor samples near the decision boundary, not only easy in-manifold samples

In short:

`cell-preserving, background-diversifying, boundary-aware synthesis`

## Required Pipeline Changes

### 1. Stop destroying background context during preprocessing

Current issue:

- center crop removes domain-specific context too early

Required change:

- build a second processed dataset branch with larger canvas, for example `384x384`
- use bounded center jitter instead of strict center crop
- preserve more smear background around the cell

Expected effect:

- the generator sees meaningful background variation during training instead of a mostly cell-centered crop

### 2. Learn the cell/background split explicitly

Current issue:

- the model sees only full-frame images
- there is no objective telling it which region should stay stable and which region may vary more

Required change:

- estimate a center cell mask for each training image
- use that mask in analysis and generation
- in generation, move to a two-stage policy:
  1. background-first variation
  2. optional mild whole-image refinement

Practical implementation options:

- SDXL inpainting on the inverse mask
- masked compositing with domain-specific background priors
- background-only color/stain augmentation before img2img refinement

### 3. Replace easy-sample scoring with boundary-aware scoring

Current issue:

- current utility logic treats high CNN confidence as good

Required change:

- rank samples by a multi-objective score:
  - high cell preservation
  - low background similarity
  - moderate classifier margin
  - target class still preserved

A useful sample is not:

- maximally easy

A useful sample is:

- still target-consistent
- but close enough to the current boundary to reshape it

### 4. Make domain style more explicit than class morphology

Current issue:

- same class prompt dominates the conditioning
- background/stain/scanner style is only weakly represented in text

Required change:

- shorten domain prompts into consistent domain tokens
- consider turning on text-encoder LoRA for domain tokens only
- optionally train a domain-style adapter separately from the class-morphology adapter

### 5. Generate with asymmetric control

Current issue:

- current img2img treats the whole frame as one unit

Required change:

- keep center cell deformation mild
- allow stronger background style transfer
- optionally use two denoise scales:
  - low for cell region
  - high for background region

## Concrete Next Build

### Phase A: New diagnostics

Already added:

- [41_boundary_aware_variation_review.py](/Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_41_61_boundary_v2/41_boundary_aware_variation_review.py)

Purpose:

- re-score generated samples by:
  - `cell_ssim`
  - `background_ssim`
  - `cnn_entropy`
  - `target_margin`
  - `near_boundary`

This gives a better objective than raw correctness.

### Phase B: New training branch

Recommended next script additions:

1. `42_preprocess_contextual_multidomain.py`
   - larger canvas
   - center-jitter crop
   - preserve background

2. `43_build_cell_masks.py`
   - export cell masks for the contextual dataset

3. `44_background_aware_generate.py`
   - inverse-mask background inpainting
   - optional whole-image low-strength refinement

4. `45_boundary_subset_builder.py`
   - select samples with:
     - high cell preservation
     - high background shift
     - low positive margin

## Recommendation

Yes, the target you described is feasible.

But it is not a small hyperparameter tweak.

It requires changing the objective of the pipeline from:

`generate easy, class-correct synthetic images`

to:

`generate biologically valid but boundary-near, background-diversified synthetic images`

That means preprocessing, generation, and filtering all need to move together.
