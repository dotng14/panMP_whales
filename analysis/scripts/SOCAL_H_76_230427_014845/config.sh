#!/usr/bin/env bash
# Shared paths for the SOCAL_H_76_230427_014845 pipeline on ensemble.
# Sourced by 01_gpl.sh .. 04_pan_matrix_profile.sh; not meant to be run directly.
# Every value can be overridden by exporting it before running a step script.

# Repository root (this file lives in analysis/scripts/SOCAL_H_76_230427_014845/).
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# Python with numpy / stumpy / cupy installed (the env used by earlier 014845 runs).
PYTHON="${PYTHON:-/home/rivo/.pyenv/versions/3.13.5/envs/dev/bin/python}"

# Input recording (200 kHz, 75 min).
LIBRARY_ROOT="${LIBRARY_ROOT:-/home/yfreund/whale_processing}"
WAV_STEM="SOCAL_H_76_230427_014845.x"
WAV="${WAV:-$LIBRARY_ROOT/LeiData/$WAV_STEM.wav}"

# All outputs of this pipeline live under one folder, one subfolder per step.
OUT_ROOT="${OUT_ROOT:-$LIBRARY_ROOT/result/LeiData/dat_pipeline/$WAV_STEM}"
GPL_DIR="${GPL_DIR:-$OUT_ROOT/gpl}"          # step 1 output, input to steps 2-4
DICT_DIR="${DICT_DIR:-$OUT_ROOT/dictionary}" # step 2 output, input to step 3
MP_DIR="${MP_DIR:-$OUT_ROOT/matching_pursuit}"
PAN_DIR="${PAN_DIR:-$OUT_ROOT/pan_mp}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
cd "$REPO"

# Run a python module, tee its output into $LOG_DIR/<name>.log.
run_logged() {
  local name="$1"; shift
  local log="$LOG_DIR/$name.log"
  echo "[$(date -Iseconds)] $name: $PYTHON $*" | tee -a "$log"
  "$PYTHON" "$@" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  echo "[$(date -Iseconds)] $name: exit $status" | tee -a "$log"
  return "$status"
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "error: $1 not found — run $2 first" >&2
    exit 2
  fi
}
