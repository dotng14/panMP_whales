#!/usr/bin/env bash
# Step 3 — matching-pursuit: place the dictionary (SOURCE=pan or randomized) on every window.
#   in : $GPL_DIR (step 1), $DICT_DIR (step 2a or 2b)
#   out: $MP_DIR/{summary.json, matchings.json, loop_stats.json}
# Usage: SOURCE=pan bash analysis/scripts/pipeline/03_matching_pursuit.sh [extra flags]
set -euo pipefail
source "$(dirname "$0")/config.sh"
require_dir "$GPL_DIR" 01_gpl.sh
require_dir "$DICT_DIR" "02_${SOURCE}_build.sh"

run_logged "03_matching_pursuit_$SOURCE" -m analysis.dict_learning matching-pursuit \
  --gpl-dir "$GPL_DIR" \
  --dictionary "$DICT_DIR" \
  --out "$MP_DIR" \
  "${WINDOW_ARGS[@]}" \
  --workers "$WORKERS" \
  --loop-stats \
  "${CATALOG_ARGS[@]}" \
  "$@"
