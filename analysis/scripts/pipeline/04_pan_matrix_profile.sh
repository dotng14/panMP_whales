#!/usr/bin/env bash
# Step 4 — pan matrix profile of the GPL band energy at 10 Hz, motifs of 1–5 s.
# Independent of steps 2 and 3.
#   in : $GPL_DIR (step 1)
#   out: $PAN_DIR/{pan.npy, pan.json, pairs.json, pan.png}
# Usage: bash analysis/scripts/pipeline/04_pan_matrix_profile.sh [extra flags]
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
  --top-k 10 \
  --threads "$PAN_THREADS" \
  --out "$PAN_DIR" \
  --plot \
  "$@"
