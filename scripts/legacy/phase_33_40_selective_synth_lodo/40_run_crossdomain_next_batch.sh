#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/imds/Desktop/wbc_synthesis"
LOG_DIR="$ROOT/results/crossdomain_next_batch/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"

echo "[1/2] Eosinophil cross_only rerun (safe denoise: 0.25, 0.35)"
python3 "$ROOT/scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py" \
  --class_name eosinophil \
  --cross_domain_mode cross_only \
  --n_per_domain 2 \
  --n_seeds 1 \
  --strengths 0.25 0.35 \
  --force | tee "$LOG_DIR/eosinophil_cross_only_safe.log"

echo "[2/2] Monocyte cross_only scale-up (n_per_domain=6)"
python3 "$ROOT/scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py" \
  --class_name monocyte \
  --cross_domain_mode cross_only \
  --n_per_domain 6 \
  --n_seeds 1 \
  --force | tee "$LOG_DIR/monocyte_cross_only_scaleup.log"

echo "Done."
