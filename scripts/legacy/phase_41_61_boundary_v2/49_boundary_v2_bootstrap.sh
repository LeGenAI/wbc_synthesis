#!/bin/bash
set -euo pipefail

ROOT="/Users/imds/Desktop/wbc_synthesis"
PYTHON="${PYTHON:-python3}"

echo "[1/7] contextual preprocess"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/42_preprocess_contextual_multidomain.py" --domain all

echo "[2/7] contextual masks"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/43_build_contextual_masks.py"

echo "[3/7] contextual LoRA dry-run for all classes"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/44_contextual_lora_train.py" --all_classes --dry_run

echo "[4/7] boundary-aware generation dry-run for all classes"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/45_background_aware_generate.py" --all_classes --dry_run

echo "[5/7] boundary subset builder"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/46_boundary_subset_builder.py"

echo "[6/7] boundary LODO dry-run"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/47_boundary_lodo_train.py" --dry_run

echo "[7/7] acceptance review"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/48_boundary_acceptance_review.py"

echo "[done] boundary-aware V2 bootstrap completed"
