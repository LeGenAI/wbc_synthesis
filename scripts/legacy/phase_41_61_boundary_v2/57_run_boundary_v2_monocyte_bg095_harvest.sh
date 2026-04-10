#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/imds/Desktop/wbc_synthesis"
PYTHON="${PYTHON:-python3}"
LOG_DIR="$ROOT/results/boundary_v2_generation/logs"
ARCHIVE_ROOT="$ROOT/results/boundary_v2_generation_archive"
STAMP="$(date +%Y%m%d_%H%M%S)"

N_PER_DOMAIN="${N_PER_DOMAIN:-16}"
N_SEEDS="${N_SEEDS:-2}"
BG_SSIM_MAX="${BG_SSIM_MAX:-0.88}"
MARGIN_MAX="${MARGIN_MAX:-0.30}"

mkdir -p "$LOG_DIR" "$ARCHIVE_ROOT"

if [[ -d "$ROOT/results/boundary_v2_generation/monocyte" ]]; then
  cp -R "$ROOT/results/boundary_v2_generation/monocyte" \
    "$ARCHIVE_ROOT/monocyte_probe_bg095_rf000_n${N_PER_DOMAIN}_s${N_SEEDS}_$STAMP"
fi

echo "[1/2] monocyte bg095 harvest"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/45_background_aware_generate.py" \
  --class_name monocyte \
  --n_per_domain "$N_PER_DOMAIN" \
  --n_seeds "$N_SEEDS" \
  --cross_domain_mode cross_only \
  --background_strengths 0.95 \
  --disable_refine \
  --force \
  2>&1 | tee "$LOG_DIR/monocyte_v2_bg095_harvest_n${N_PER_DOMAIN}_s${N_SEEDS}_$STAMP.log"

echo "[2/2] rebuild boundary subset"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/46_boundary_subset_builder.py" \
  --margin_max "$MARGIN_MAX" \
  --background_ssim_max "$BG_SSIM_MAX" \
  2>&1 | tee "$LOG_DIR/boundary_subset_builder_mm${MARGIN_MAX/./}_bg${BG_SSIM_MAX/./}_$STAMP.log"

echo "[done] monocyte bg095 harvest finished"
