#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/imds/Desktop/wbc_synthesis"
PYTHON="${PYTHON:-python3}"
LOG_DIR="$ROOT/results/boundary_v2_generation/logs"
ARCHIVE_ROOT="$ROOT/results/boundary_v2_generation_archive"
STAMP="$(date +%Y%m%d_%H%M%S)"
BG_SSIM_MAX="${BG_SSIM_MAX:-0.90}"

mkdir -p "$LOG_DIR" "$ARCHIVE_ROOT"

if [[ -d "$ROOT/results/boundary_v2_generation/monocyte" ]]; then
  cp -R "$ROOT/results/boundary_v2_generation/monocyte" \
    "$ARCHIVE_ROOT/monocyte_probe_bg095_rf000_n1_$STAMP"
fi

echo "[1/2] monocyte bg095 scale-up"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/45_background_aware_generate.py" \
  --class_name monocyte \
  --n_per_domain 4 \
  --n_seeds 1 \
  --cross_domain_mode cross_only \
  --background_strengths 0.95 \
  --disable_refine \
  --force \
  2>&1 | tee "$LOG_DIR/monocyte_v2_bg095_scaleup_$STAMP.log"

echo "[2/2] rebuild relaxed boundary subset"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/46_boundary_subset_builder.py" \
  --background_ssim_max "$BG_SSIM_MAX" \
  2>&1 | tee "$LOG_DIR/boundary_subset_builder_bg${BG_SSIM_MAX/./}_$STAMP.log"

echo "[done] monocyte bg095 scale-up finished"
