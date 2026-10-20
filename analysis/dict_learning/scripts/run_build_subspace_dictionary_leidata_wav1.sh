#!/usr/bin/env bash
# Build a K=2 subspace dictionary for LeiData wave file 1 (manifest-only).
#
# Uses unified learn/build with --n-components 2. Writes basis npy + coeff
# matchings under OUT_DIR; does not register a catalog dictionary row.
#
# Prerequisites: catalog already indexed (used only to load windows/detections).
#
# Usage (from repo root):
#   bash analysis/dict_learning/scripts/run_build_subspace_dictionary_leidata_wav1.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

LIBRARY_ROOT="/home/yfreund/whale_processing"
DB="${CATALOG_DB:-$LIBRARY_ROOT/result/dict_learning_catalog.sqlite}"
WAV="$LIBRARY_ROOT/LeiData/SOCAL_H_76_230427_003345.x.wav"
OUT_DIR="${OUT_DIR:-$LIBRARY_ROOT/result/LeiData/dict_learning_templates/wav1_subspace_k2}"
N_WINDOWS="${N_WINDOWS:-4}"
N_TEMPLATES="${N_TEMPLATES:-20}"
K="${K:-2}"

python -m analysis.dict_learning.code.learn.build_dictionary \
  --db "$DB" \
  --wav "$WAV" \
  --out-dir "$OUT_DIR" \
  --name "leidata_wav1_subspace_k${K}" \
  --n-windows "$N_WINDOWS" \
  --window-offset 0 \
  --n-rounds "$N_TEMPLATES" \
  --eps-step-fraction 0.3 \
  --min-repeats 3 \
  --min-template-width 12 \
  --max-template-width 24 \
  --n-components "$K" \
  --refine-iters 0
