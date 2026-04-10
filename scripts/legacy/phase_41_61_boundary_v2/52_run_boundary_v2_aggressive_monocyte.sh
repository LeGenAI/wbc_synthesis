#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/imds/Desktop/wbc_synthesis"
PYTHON="${PYTHON:-python3}"
LOG_DIR="$ROOT/results/boundary_v2_generation/logs"
ARCHIVE_ROOT="$ROOT/results/boundary_v2_generation_archive"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$ARCHIVE_ROOT"

if [[ -d "$ROOT/results/boundary_v2_generation/monocyte" ]]; then
  cp -R "$ROOT/results/boundary_v2_generation/monocyte" \
    "$ARCHIVE_ROOT/monocyte_probe_bg065_rf020_n2_$STAMP"
fi

echo "[aggressive probe] monocyte"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/45_background_aware_generate.py" \
  --class_name monocyte \
  --n_per_domain 1 \
  --n_seeds 1 \
  --cross_domain_mode cross_only \
  --background_strengths 0.75 0.85 \
  --disable_refine \
  --force \
  2>&1 | tee "$LOG_DIR/monocyte_v2_aggressive_$STAMP.log"

echo "[done] monocyte aggressive boundary V2 probe"
