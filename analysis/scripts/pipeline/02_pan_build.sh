#!/usr/bin/env bash
# Step 2b — pan-build: seeds from the pan matrix profile of the residual (8 PCA
# rows, 1–4 s motifs), MP + PCA refine, same passes and stop as step 2a.
# Needs no detections, so it also works where GPL detects poorly.
#   in : $GPL_DIR (step 1)
#   out: $PAN_BUILD_DIR/{atoms/, summary.json, matchings.json, outer_logs.jsonl, pan_seeds.jsonl, loop_stats.json}
# Usage: bash analysis/scripts/pipeline/02_pan_build.sh [extra flags]
set -euo pipefail
source "$(dirname "$0")/config.sh"
require_dir "$GPL_DIR" 01_gpl.sh

run_logged 02_pan_build -m analysis.dict_learning pan-build \
  --gpl-dir "$GPL_DIR" \
  --out "$PAN_BUILD_DIR" \
  "${WINDOW_ARGS[@]}" \
  --signal pca \
  --pca-rows 8 \
  --lengths-s 1,2,3,4 \
  --m-per-outer 5 \
  --n-min 10 \
  --max-outer 10 \
  --min-coverage 0.2 \
  --refine-iters 1 \
  --pan-threads "$PAN_THREADS" \
  --workers "$WORKERS" \
  --verbose \
  --loop-stats \
  "${CATALOG_ARGS[@]}" \
  "$@"
