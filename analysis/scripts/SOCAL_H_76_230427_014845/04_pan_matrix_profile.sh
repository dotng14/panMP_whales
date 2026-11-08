#!/usr/bin/env bash
# Step 4 — Pan matrix profile on the GPL band-energy signal at 10 Hz,
# motif lengths 10..50 samples (1..5 s) in steps of 10.
#   in : $GPL_DIR (01_gpl.sh) — independent of steps 2 and 3
#   out: $PAN_DIR/{pan.npy, pan.json, pairs.json, pan.png}
# Usage: bash analysis/scripts/SOCAL_H_76_230427_014845/04_pan_matrix_profile.sh [extra flags]
set -euo pipefail
source "$(dirname "$0")/config.sh"
require_dir "$GPL_DIR" 01_gpl.sh

run_logged 04_pan_matrix_profile -m analysis.matrix_profile pan \
  --gpl-dir "$GPL_DIR" \
  --rate 10 \
  --signal bands \
  --min-len 10 \
  --max-len 50 \
  --step 10 \
  --out "$PAN_DIR" \
  --plot \
  "$@"
