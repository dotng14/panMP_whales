#!/usr/bin/env bash
# LeiData wav1 convenience wrapper around run_build_dictionary.sh
# (efficiency defaults + the fast 4-window / 20-round / eps=0.5 recipe).
#
# Usage (from repo root):
#   bash analysis/dict_learning/scripts/run_build_dictionary_leidata_wav1.sh
#
# Extra flags are forwarded (e.g. --no-progress). Override paths with
# CATALOG_DB / LIBRARY_ROOT, or efficiency with DICT_LEARNING_MP_*.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LIBRARY_ROOT="${LIBRARY_ROOT:-/home/yfreund/whale_processing}"
export CATALOG_DB="${CATALOG_DB:-$LIBRARY_ROOT/result/dict_learning_catalog.sqlite}"
WAV="$LIBRARY_ROOT/LeiData/SOCAL_H_76_230427_003345.x.wav"
OUT_DIR="$LIBRARY_ROOT/result/LeiData/dict_learning_templates/wav1_averaged"

exec bash "$HERE/../run_build_dictionary.sh" \
  --db "$CATALOG_DB" \
  --wav "$WAV" \
  --out-dir "$OUT_DIR" \
  --name "leidata_wav1_averaged" \
  --n-windows 4 \
  --window-offset 0 \
  --n-rounds 20 \
  --eps-step-fraction 0.5 \
  --min-repeats 3 \
  --min-template-width 12 \
  --max-template-width 24 \
  --template-method "pca" \
  --refine-iters 0 \
  "$@"
