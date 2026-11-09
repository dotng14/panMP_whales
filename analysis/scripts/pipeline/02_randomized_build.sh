#!/usr/bin/env bash
# Step 2a — randomized-build: seeds sampled from GPL detections, MP + PCA refine,
# passes until 20% of the time bins are covered (at most 10 passes).
#   in : $GPL_DIR (step 1; needs its CSV_*.csv detections)
#   out: $RANDOMIZED_DIR/{atoms/, summary.json, matchings.json, outer_logs.jsonl, loop_stats.json}
# Usage: bash analysis/scripts/pipeline/02_randomized_build.sh [extra flags]
set -euo pipefail
source "$(dirname "$0")/config.sh"
require_dir "$GPL_DIR" 01_gpl.sh

run_logged 02_randomized_build -m analysis.dict_learning randomized-build \
  --gpl-dir "$GPL_DIR" \
  --out "$RANDOMIZED_DIR" \
  "${WINDOW_ARGS[@]}" \
  --m-per-outer 5 \
  --n-min 10 \
  --max-outer 10 \
  --min-coverage 0.2 \
  --refine-iters 1 \
  --seed 0 \
  --workers "$WORKERS" \
  --verbose \
  --loop-stats \
  "${CATALOG_ARGS[@]}" \
  "$@"
