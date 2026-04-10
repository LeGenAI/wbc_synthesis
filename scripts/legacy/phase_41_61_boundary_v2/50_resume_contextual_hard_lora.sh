#!/bin/bash
set -euo pipefail

ROOT="/Users/imds/Desktop/wbc_synthesis"
PYTHON="${PYTHON:-python3}"

echo "[1/2] resume monocyte from latest checkpoint"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/44_contextual_lora_train.py" \
  --class_name monocyte \
  --resume_from_checkpoint latest \
  --disable_validation

echo "[2/2] resume eosinophil from latest checkpoint"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/44_contextual_lora_train.py" \
  --class_name eosinophil \
  --resume_from_checkpoint latest \
  --disable_validation

echo "[done] sequential contextual hard-class resume complete"
