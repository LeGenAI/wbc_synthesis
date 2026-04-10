#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/imds/Desktop/wbc_synthesis"
PYTHON="${PYTHON:-python3}"
LOG_DIR="$ROOT/results/boundary_v2_generation/logs"
mkdir -p "$LOG_DIR"

MONO_LORA="$ROOT/lora/weights/contextual_multidomain_monocyte/pytorch_lora_weights.safetensors"
EOS_LORA="$ROOT/lora/weights/contextual_multidomain_eosinophil/pytorch_lora_weights.safetensors"

if [[ ! -f "$MONO_LORA" ]]; then
  echo "[error] missing monocyte root LoRA weights: $MONO_LORA" >&2
  exit 1
fi

if [[ ! -f "$EOS_LORA" ]]; then
  echo "[error] missing eosinophil root LoRA weights: $EOS_LORA" >&2
  echo "[hint] finish contextual eosinophil training before running this probe" >&2
  exit 1
fi

echo "[1/3] monocyte boundary V2 generation probe"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/45_background_aware_generate.py" \
  --class_name monocyte \
  --n_per_domain 2 \
  --n_seeds 1 \
  --cross_domain_mode cross_only \
  2>&1 | tee "$LOG_DIR/monocyte_v2_probe.log"

echo "[2/3] eosinophil boundary V2 generation probe"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/45_background_aware_generate.py" \
  --class_name eosinophil \
  --n_per_domain 2 \
  --n_seeds 1 \
  --cross_domain_mode cross_only \
  2>&1 | tee "$LOG_DIR/eosinophil_v2_probe.log"

echo "[3/3] boundary-aware subset builder"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/46_boundary_subset_builder.py" \
  2>&1 | tee "$LOG_DIR/boundary_subset_builder.log"

echo "[done] boundary V2 hard probe finished"
