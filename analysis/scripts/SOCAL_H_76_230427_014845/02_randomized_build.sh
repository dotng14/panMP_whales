#!/usr/bin/env bash
# Step 2 — Randomized dictionary build (random seeds, MP + PCA refine,
# outer loop until time coverage >= 0.5).
#   in : $GPL_DIR (from 01_gpl.sh)
#   out: $DICT_DIR/{atoms/template_*.npy, summary.json, matchings.json, outer_logs.jsonl}
# Usage: bash analysis/scripts/SOCAL_H_76_230427_014845/02_randomized_build.sh [extra flags]
set -euo pipefail
source "$(dirname "$0")/config.sh"
require_dir "$GPL_DIR" 01_gpl.sh

run_logged 02_randomized_build -m analysis.dict_learning randomized-build \
  --gpl-dir "$GPL_DIR" \
  --out "$DICT_DIR" \
  --m-per-outer 5 \
  --n-min 10 \
  --min-coverage 0.5 \
  --seed 0 \
  --verbose \
  "$@"
