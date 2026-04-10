#!/bin/bash
set -euo pipefail

ROOT="/Users/imds/Desktop/wbc_synthesis"
PYTHON="/Library/Developer/CommandLineTools/usr/bin/python3"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

wait_for_monocyte() {
  while pgrep -f "10_multidomain_lora_train.py --class_name monocyte --steps 800" >/dev/null; do
    sleep 30
  done
}

echo "[batch] waiting for active monocyte LoRA run to finish"
wait_for_monocyte

echo "[batch] starting eosinophil LoRA retraining"
"$PYTHON" "$ROOT/scripts/legacy/phase_08_17_domain_gap_multidomain/10_multidomain_lora_train.py" \
  --class_name eosinophil \
  --steps 800 \
  --enable_default_validation

echo "[batch] starting monocyte cross-domain generation"
"$PYTHON" "$ROOT/scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py" \
  --class_name monocyte \
  --n_per_domain 2 \
  --n_seeds 1 \
  --cross_domain_mode cross_only

echo "[batch] starting eosinophil cross-domain generation"
"$PYTHON" "$ROOT/scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py" \
  --class_name eosinophil \
  --n_per_domain 2 \
  --n_seeds 1 \
  --cross_domain_mode cross_only

echo "[batch] completed"
