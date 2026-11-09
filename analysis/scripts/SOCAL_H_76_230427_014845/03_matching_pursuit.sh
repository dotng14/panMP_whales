#!/usr/bin/env bash
# Step 3 — Mallat matching pursuit: place the learned dictionary on every GPL window.
#   in : $GPL_DIR (01_gpl.sh), $DICT_DIR (02_randomized_build.sh)
#   out: $MP_DIR/{summary.json, matchings.json}
# Usage: bash analysis/scripts/SOCAL_H_76_230427_014845/03_matching_pursuit.sh [extra flags]
set -euo pipefail
source "$(dirname "$0")/config.sh"
require_dir "$GPL_DIR" 01_gpl.sh
require_dir "$DICT_DIR" 02_randomized_build.sh

run_logged 03_matching_pursuit -m analysis.dict_learning matching-pursuit \
  --gpl-dir "$GPL_DIR" \
  --dictionary "$DICT_DIR" \
  --out "$MP_DIR" \
  "$@"
