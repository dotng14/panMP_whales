#!/usr/bin/env bash
# The randomized pipeline of the root README: GPL → randomized-build → matching-pursuit, plus the pan matrix profile.
# Usage: WAV=rec.wav [CATALOG=catalog.sqlite] bash analysis/scripts/pipeline/run_randomized_pipeline.sh
#   With GPL_DIR set to existing GPL output, step 1 is skipped.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export SOURCE=randomized
if [[ -z "${GPL_DIR:-}" ]]; then bash "$here/01_gpl.sh"; fi
bash "$here/02_randomized_build.sh"
bash "$here/03_matching_pursuit.sh"
bash "$here/04_pan_matrix_profile.sh"
