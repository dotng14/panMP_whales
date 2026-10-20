#!/usr/bin/env bash
# Compare standard rank-1 MP vs subspace (top-k PCA) MP on LeiData wave file 1.
#
# Uses the standalone prototype harness (analysis.dict_learning.experiments.subspace_mp_compare);
# it does NOT persist a dictionary to the catalog. It re-runs matching pursuit with
# K=1 and K=2 subspace atoms built from an already-built dictionary's matched
# patches, and reports residual energy removed, coverage, per-scan scoring cost
# (k_eff), and wall time.
#
# Prerequisite: build the dictionary first, e.g.
#   bash analysis/dict_learning/scripts/run_build_dictionary_leidata_wav1.sh
#
# Usage (from repo root):
#   bash analysis/dict_learning/scripts/run_subspace_mp_compare_leidata_wav1.sh
#
# Overridable via env: CATALOG_DB, DICT_DIR (specific stamped run), N_WINDOWS,
#   KS (e.g. "1 2 3"), EPS_STEP_FRACTION, OUT_JSON, PC1_SCREEN=1.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

LIBRARY_ROOT="/home/yfreund/whale_processing"
DB="${CATALOG_DB:-$LIBRARY_ROOT/result/dict_learning_catalog.sqlite}"
# Wave file 1 = first LeiData recording (sorted by name).
WAV="$LIBRARY_ROOT/LeiData/SOCAL_H_76_230427_003345.x.wav"
DICT_BASE="$LIBRARY_ROOT/result/LeiData/dict_learning_templates/wav1_averaged"

# Resolve the latest stamped build under DICT_BASE (override with DICT_DIR).
if [[ -n "${DICT_DIR:-}" ]]; then
  RUN_DICT_DIR="$DICT_DIR"
else
  RUN_DICT_DIR="$(ls -1d "$DICT_BASE"/*/ 2>/dev/null | sort | tail -n 1)"
fi
if [[ -z "${RUN_DICT_DIR:-}" || ! -d "$RUN_DICT_DIR" ]]; then
  echo "error: no built dictionary under $DICT_BASE" >&2
  echo "       run analysis/dict_learning/scripts/run_build_dictionary_leidata_wav1.sh first" >&2
  exit 1
fi
echo "Using dictionary: $RUN_DICT_DIR"

N_WINDOWS="${N_WINDOWS:-4}"
KS="${KS:-1 2}"
EPS_STEP_FRACTION="${EPS_STEP_FRACTION:-0.3}"
OUT_JSON="${OUT_JSON:-${RUN_DICT_DIR%/}/subspace_mp_compare_k1_k2.json}"

SCREEN_FLAG=()
if [[ "${PC1_SCREEN:-0}" == "1" ]]; then
  SCREEN_FLAG=(--pc1-screen)
fi

python -m analysis.dict_learning.experiments.subspace_mp_compare \
  --db "$DB" \
  --wav "$WAV" \
  --dict-dir "$RUN_DICT_DIR" \
  --n-windows "$N_WINDOWS" \
  --window-offset 0 \
  --ks $KS \
  --eps-step-fraction "$EPS_STEP_FRACTION" \
  --out "$OUT_JSON" \
  "${SCREEN_FLAG[@]}"
