#!/usr/bin/env bash
#
# Nightly GPL preprocessing for the ensemble server.
#
# Recursively scans the WAV library, skips files that already have a valid
# cache, and writes new caches into <LIBRARY>/result/ (mirroring the source
# tree). Designed to be run from cron; safe to run repeatedly. Runs as
# whichever account owns the crontab entry on that server (e.g. yfreund
# below) — a different account from anyone's local dev checkout.
#
# "Already cached" only skips the expensive pipeline (ProcessingOrchestrator).
# Each run still cheaply backfills repeats.json in place whenever
# REPEATS_VERSION has moved on (e.g. the averaged-template panel added for
# RepeatSpectrogram) — look for "repeats rebuilt vN" in the log to confirm a
# nightly run picked up a version bump across the library.
#
# Setup:
#   1. Edit REPO_DIR / VENV / LIBRARY below to match this server.
#   2. Ensure VENV has the analysis package installed: from REPO_DIR,
#      pip install -e '.[gpu]' (or -e . for a CPU-only host).
#   3. chmod +x backend/scripts/preprocess_cron.sh
#   4. Test once by hand:  backend/scripts/preprocess_cron.sh
#   5. Install the crontab entry (see bottom of this file).
#
set -euo pipefail

# ── Config — edit these for the ensemble server ────────────────────────────
REPO_DIR="${GPL_REPO_DIR:-/home/yfreund/gplrev3-webserver}"   # repo checkout
VENV="${GPL_VENV:-$REPO_DIR/.venv}"                            # python venv
LIBRARY="${GPL_LIBRARY:-/home/yfreund/whale_processing}"       # WAV library root

# GPU if available; the compute backend auto-falls back to CPU when CuPy/CUDA
# is missing (analysis/GPL_Python/gpl_compute/backend.py), so this is safe on CPU-only hosts.
export GPL_COMPUTE="${GPL_COMPUTE:-cuda}"
# export GPL_NUM_GPUS=2
# export CUDA_VISIBLE_DEVICES=0,1

LOG_DIR="$REPO_DIR/backend/logs"
LOCK_FILE="/tmp/gpl_preprocess.lock"
# ───────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Prevent overlapping runs: re-exec the script under an exclusive, non-blocking
# flock. If a previous run is still going, exit quietly.
if [ "${GPL_LOCKED:-}" != "1" ]; then
    exec env GPL_LOCKED=1 flock -n "$LOCK_FILE" "$0" "$@" || {
        log "Another preprocessing run is in progress; exiting."
        exit 0
    }
fi

log "Starting nightly preprocessing"
log "  repo=$REPO_DIR library=$LIBRARY GPL_COMPUTE=$GPL_COMPUTE"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    log "  GPU detected: $(nvidia-smi -L | head -n1)"
else
    log "  No GPU detected — compute backend will use CPU."
fi

# shellcheck disable=SC1090,SC1091
source "$VENV/bin/activate"
cd "$REPO_DIR/backend"

# --no-bmp matches the web pipeline (write_bmp=False); preview BMPs aren't
# needed for the cache and skipping them is faster.
python preprocess.py "$LIBRARY" --no-bmp

log "Preprocessing complete"

# ── Install (run once, then edit paths as needed) ──────────────────────────
# Edit your crontab with:  crontab -e
# Add this line for a 2 AM nightly run (logs are also captured by the script):
#
#   0 2 * * * /home/yfreund/gplrev3-webserver/backend/scripts/preprocess_cron.sh >> /home/yfreund/gplrev3-webserver/backend/logs/preprocess_cron.log 2>&1
#
# Verify with:  crontab -l
