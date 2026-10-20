#!/usr/bin/env bash
# Grow K=1 and K=2 dictionaries on LeiData wav1 and write a compare summary.
#
# Requirements this script enforces:
#   - >= 4 windows of real LeiData
#   - >= 5 templates for each of K=1 and K=2
#   - >= 2 refinement iterations
#   - reports residual energy reduction for both
#
# Then open the notebook for spectrogram overlays:
#   analysis/dict_learning/notebooks/subspace_mp_k1_k2_compare.ipynb
#
# Usage (from repo root):
#   bash analysis/dict_learning/scripts/run_compare_grow_k1_k2_leidata_wav1.sh
#
# Env overrides: CATALOG_DB, OUT_DIR, N_WINDOWS, N_TEMPLATES, REFINE_ITERS,
#   MIN_TEMPLATES, EPS_STEP_FRACTION, SEED_WINDOW_POLICY, SEED_WINDOW_NEIGHBORHOOD,
#   MAX_RELATIVE_DISTANCE (e.g. 0.5 = drop patches with >50% energy unexplained)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

LIBRARY_ROOT="/home/yfreund/whale_processing"
DB="${CATALOG_DB:-$LIBRARY_ROOT/result/dict_learning_catalog.sqlite}"
WAV="$LIBRARY_ROOT/LeiData/SOCAL_H_76_230427_003345.x.wav"
OUT_DIR="${OUT_DIR:-$LIBRARY_ROOT/result/LeiData/dict_learning_templates/wav1_k1_k2_grow_compare}"
N_WINDOWS="${N_WINDOWS:-4}"
N_TEMPLATES="${N_TEMPLATES:-5}"
MIN_TEMPLATES="${MIN_TEMPLATES:-5}"
REFINE_ITERS="${REFINE_ITERS:-2}"
EPS_STEP_FRACTION="${EPS_STEP_FRACTION:-0.1}"
SEED_WINDOW_POLICY="${SEED_WINDOW_POLICY:-low_coverage}"
SEED_WINDOW_NEIGHBORHOOD="${SEED_WINDOW_NEIGHBORHOOD:-0}"
EXTRA_ARGS=()
if [[ -n "${MAX_RELATIVE_DISTANCE:-}" ]]; then
  EXTRA_ARGS+=(--max-relative-distance "$MAX_RELATIVE_DISTANCE")
fi

python -m analysis.dict_learning.experiments.compare_grow_k1_k2 \
  --db "$DB" \
  --wav "$WAV" \
  --out-dir "$OUT_DIR" \
  --n-windows "$N_WINDOWS" \
  --n-templates "$N_TEMPLATES" \
  --min-templates "$MIN_TEMPLATES" \
  --refine-iters "$REFINE_ITERS" \
  --eps-step-fraction "$EPS_STEP_FRACTION" \
  --seed-window-policy "$SEED_WINDOW_POLICY" \
  --seed-window-neighborhood "$SEED_WINDOW_NEIGHBORHOOD" \
  "${EXTRA_ARGS[@]}" \
  --min-repeats 3 \
  --min-template-width 12 \
  --max-template-width 24 \
  --template-method pca

echo
echo "Next: open analysis/dict_learning/notebooks/subspace_mp_k1_k2_compare.ipynb"
echo "      (COMPARE_ROOT defaults to $OUT_DIR)"
