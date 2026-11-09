#!/usr/bin/env bash
# Step 1 — GPL: whitened spectrogram windows and detections, 10 Hz – 10 kHz.
#   in : $WAV
#   out: $GPL_DIR/{CSV_*.csv, Detections_*.mat, Labels_*.tlab, spectrogram_npy/}
# Usage: WAV=rec.wav bash analysis/scripts/pipeline/01_gpl.sh [extra process_v4 flags]
set -euo pipefail
source "$(dirname "$0")/config.sh"
if [[ ! -f "$WAV" ]]; then
  echo "error: WAV not found: ${WAV:-<unset>}" >&2
  exit 2
fi

run_logged 01_gpl -m analysis.GPL_Python.process_v4 "$WAV" \
  --output-dir "$GPL_DIR" \
  --freq-lo 10 \
  --freq-hi 10000 \
  "$@"
