#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/imds/Desktop/wbc_synthesis"
PYTHON="${PYTHON:-python3}"
LOG_DIR="$ROOT/results/boundary_v2_generation/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"

N_PER_DOMAIN="${N_PER_DOMAIN:-32}"
N_SEEDS="${N_SEEDS:-4}"
BG_SSIM_MAX="${BG_SSIM_MAX:-0.88}"
MARGIN_MAX="${MARGIN_MAX:-0.30}"
RUN_TAG="${RUN_TAG:-monocyte_mll23_raabin_bg095_n${N_PER_DOMAIN}_s${N_SEEDS}_$STAMP}"

ACTIVE_MONO_REPORT="$ROOT/results/boundary_v2_generation/monocyte/report.json"
ACTIVE_EOS_REPORT="$ROOT/results/boundary_v2_generation/eosinophil/report.json"
SAFE_MONO_REPORT="$ROOT/results/boundary_v2_generation_runs/$RUN_TAG/monocyte/report.json"

mkdir -p "$LOG_DIR"

declare -a REPORT_PATHS=()
for path in \
  "$ACTIVE_MONO_REPORT" \
  "$ACTIVE_EOS_REPORT" \
  "$SAFE_MONO_REPORT"
do
  REPORT_PATHS+=("$path")
done

for path in "$ROOT"/results/boundary_v2_generation_runs/*/monocyte/report.json; do
  [[ -f "$path" ]] || continue
  REPORT_PATHS+=("$path")
done

for path in "$ROOT"/results/boundary_v2_generation_archive/monocyte_probe_bg095*/report.json; do
  [[ -f "$path" ]] || continue
  REPORT_PATHS+=("$path")
done

echo "[1/2] monocyte bg095 safe harvest (MLL23 -> Raabin) run_tag=$RUN_TAG"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/45_background_aware_generate.py" \
  --class_name monocyte \
  --n_per_domain "$N_PER_DOMAIN" \
  --n_seeds "$N_SEEDS" \
  --cross_domain_mode cross_only \
  --background_strengths 0.95 \
  --disable_refine \
  --ref_domains domain_c_mll23 \
  --target_domains domain_b_raabin \
  --run_tag "$RUN_TAG" \
  --force \
  2>&1 | tee "$LOG_DIR/monocyte_v2_raabin_safe_harvest_n${N_PER_DOMAIN}_s${N_SEEDS}_$STAMP.log"

echo "[2/2] rebuild boundary subset from safe union"
"$PYTHON" "$ROOT/scripts/legacy/phase_41_61_boundary_v2/46_boundary_subset_builder.py" \
  --margin_max "$MARGIN_MAX" \
  --background_ssim_max "$BG_SSIM_MAX" \
  --report_paths "${REPORT_PATHS[@]}" \
  2>&1 | tee "$LOG_DIR/boundary_subset_builder_safe_union_mm${MARGIN_MAX/./}_bg${BG_SSIM_MAX/./}_$STAMP.log"

echo "[done] monocyte bg095 Raabin safe harvest finished"
