#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/imds/Desktop/wbc_synthesis"
PYTHON="${PYTHON:-python3}"
LOG_DIR="$ROOT/results/boundary_v2_generation/logs"
ARCHIVE_ROOT="$ROOT/results/boundary_v2_generation_archive"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$ARCHIVE_ROOT"

if [[ -d "$ROOT/results/boundary_v2_generation/eosinophil" ]]; then
  cp -R "$ROOT/results/boundary_v2_generation/eosinophil" \
    "$ARCHIVE_ROOT/eosinophil_probe_bg075_bg085_rf000_n1_$STAMP"
fi

echo "[ultra probe] eosinophil"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/45_background_aware_generate.py" \
  --class_name eosinophil \
  --n_per_domain 1 \
  --n_seeds 1 \
  --cross_domain_mode cross_only \
  --background_strengths 0.95 \
  --disable_refine \
  --force \
  2>&1 | tee "$LOG_DIR/eosinophil_v2_ultra_$STAMP.log"

echo "[done] eosinophil ultra boundary V2 probe"
