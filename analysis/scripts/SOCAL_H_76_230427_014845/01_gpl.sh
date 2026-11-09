#!/usr/bin/env bash
# Step 1 — GPL detector: whitened spectrogram windows + detections.
#   in : $WAV
#   out: $GPL_DIR/{CSV_*.csv, Detections_*.mat, Labels_*.tlab, spectrogram_npy/}
# Usage (from anywhere): bash analysis/scripts/SOCAL_H_76_230427_014845/01_gpl.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [[ ! -f "$WAV" ]]; then
  echo "error: WAV not found: $WAV" >&2
  exit 2
fi

run_logged 01_gpl -m analysis.GPL_Python.process_v4 "$WAV" \
  --output-dir "$GPL_DIR" \
  "$@"
