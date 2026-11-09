#!/usr/bin/env bash
# Shared settings for the pipeline scripts in this folder; sourced, not run.
# Every value can be overridden by exporting it before running a script.
#
#   WAV=/data/rec.wav bash analysis/scripts/pipeline/run_pan_pipeline.sh
#   GPL_DIR=/existing/gpl/output bash analysis/scripts/pipeline/02_pan_build.sh

# Repository root (this file lives in analysis/scripts/pipeline/).
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON="${PYTHON:-python}"

# The recording. Not needed when GPL_DIR already holds GPL output.
WAV="${WAV:-}"
if [[ -n "$WAV" ]]; then
  STEM="${STEM:-$(basename "$WAV" .wav)}"
elif [[ -n "${GPL_DIR:-}" ]]; then
  STEM="${STEM:-$(basename "$(dirname "$GPL_DIR")" .gpl)}"
else
  echo "error: set WAV (a recording) or GPL_DIR (existing GPL output)" >&2
  exit 2
fi

# Every output lives under one folder, one subfolder per step.
OUT_ROOT="${OUT_ROOT:-$HOME/whale_runs/$STEM}"
GPL_DIR="${GPL_DIR:-$OUT_ROOT/gpl}"
RANDOMIZED_DIR="${RANDOMIZED_DIR:-$OUT_ROOT/randomized_build}"
PAN_BUILD_DIR="${PAN_BUILD_DIR:-$OUT_ROOT/pan_build}"
PAN_DIR="${PAN_DIR:-$OUT_ROOT/pan_mp}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

# Which dictionary steps 3 and 5 use: "pan" (pan_build/) or "randomized" (randomized_build/).
SOURCE="${SOURCE:-pan}"
case "$SOURCE" in
  pan) DICT_DIR="${DICT_DIR:-$PAN_BUILD_DIR}" ;;
  randomized) DICT_DIR="${DICT_DIR:-$RANDOMIZED_DIR}" ;;
  *) echo "error: SOURCE must be pan or randomized, got $SOURCE" >&2; exit 2 ;;
esac
MP_DIR="${MP_DIR:-$OUT_ROOT/matching_pursuit_$SOURCE}"

# CPU budget: the server is shared, so stay well under its 32 cores.
WORKERS="${WORKERS:-4}"
PAN_THREADS="${PAN_THREADS:-4}"

# Optional: use only the first N_WINDOWS GPL windows in steps 2–5 (a quick look first).
N_WINDOWS="${N_WINDOWS:-}"
WINDOW_ARGS=()
if [[ -n "$N_WINDOWS" ]]; then
  WINDOW_ARGS=(--n-windows "$N_WINDOWS")
fi

# Optional SQLite catalog for the web viewer; when set, every step registers its output.
CATALOG="${CATALOG:-}"
CATALOG_ARGS=()
if [[ -n "$CATALOG" ]]; then
  CATALOG_ARGS=(--catalog "$CATALOG")
fi

mkdir -p "$OUT_ROOT" "$LOG_DIR"
cd "$REPO"

# Run a python module, tee its output into $LOG_DIR/<name>.log; return its exit code.
run_logged() {
  local name="$1"; shift
  local log="$LOG_DIR/$name.log"
  echo "[$(date -Iseconds)] $name: $PYTHON $*" | tee -a "$log"
  set +e
  "$PYTHON" "$@" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "[$(date -Iseconds)] $name: exit $status" | tee -a "$log"
  return "$status"
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "error: $1 not found — run $2 first" >&2
    exit 2
  fi
}
