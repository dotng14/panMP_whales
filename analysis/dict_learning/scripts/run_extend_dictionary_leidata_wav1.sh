#!/usr/bin/env bash
# Chain extend_dictionary on LeiData wav1 across expanding window prefixes:
#   n_windows = 1 → 2 → 4 → 8 → 16 → 30
#
# K (n_components):
#   spans 1 and 2  → K=1  (catalog lineage via dictionary_id)
#   spans 4+       → K=10 (chain via previous stage out_dir; subspace is
#                          manifest-only and has no catalog dictionary_id)
#
# Env overrides:
#   CATALOG_DB           catalog sqlite path
#   INPUT_DICTIONARY_ID  seed stage 0 as a refinement (omit for cold start)
#   WINDOW_SPANS         space-separated n_windows list (default: 1 2 4 8 16 30)
#   K1_MAX_WINDOWS       last span that uses K=1 (default: 2)
#   K_LATE               n_components for later spans (default: 10)
#   MAX_NEW_ATOMS        cap atoms added per stage (default: 10; early spans
#                        often cannot reach min_coverage=0.9)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

LIBRARY_ROOT="/home/yfreund/whale_processing"
DB="${CATALOG_DB:-$LIBRARY_ROOT/result/dict_learning_catalog.sqlite}"
WAV="$LIBRARY_ROOT/LeiData/SOCAL_H_76_230427_003345.x.wav"
OUT_DIR="$LIBRARY_ROOT/result/LeiData/dict_learning_templates/wav1_extend_chain"
mkdir -p "$OUT_DIR"

# shellcheck disable=SC2206
SPANS=(${WINDOW_SPANS:-1 2 4 8 16 30})
K1_MAX_WINDOWS="${K1_MAX_WINDOWS:-2}"
K_LATE="${K_LATE:-10}"
MAX_NEW_ATOMS="${MAX_NEW_ATOMS:-10}"
PREV_ID="${INPUT_DICTIONARY_ID:-}"
PREV_DIR=""

echo "Chained extend on wav1: spans=${SPANS[*]}  db=$DB" >&2
echo "K=1 for n_windows<=${K1_MAX_WINDOWS}; K=${K_LATE} after that; max_new_atoms=${MAX_NEW_ATOMS}" >&2
echo "out_dir=$OUT_DIR" >&2

for N in "${SPANS[@]}"; do
  STAGE_JSON="$OUT_DIR/stage_w${N}.json"
  if (( N <= K1_MAX_WINDOWS )); then
    K=1
  else
    K="$K_LATE"
  fi

  EXTRA=(--n-components "$K" --max-new-atoms "$MAX_NEW_ATOMS")
  if (( K == 1 )); then
    if [[ -n "$PREV_ID" ]]; then
      EXTRA+=(--input-dictionary-id "$PREV_ID")
    fi
  else
    # Prefer on-disk parent so K=10→K=10 chains across stamped run dirs.
    if [[ -n "$PREV_DIR" ]]; then
      EXTRA+=(--input-dictionary-dir "$PREV_DIR")
    elif [[ -n "$PREV_ID" ]]; then
      EXTRA+=(--input-dictionary-id "$PREV_ID")
    fi
  fi

  echo "=== stage n_windows=$N K=$K parent_id=${PREV_ID:-none} parent_dir=${PREV_DIR:-none} ===" >&2
  STAGE_OUT="$(mktemp)"
  python -u -m analysis.dict_learning.code.learn.extend_dictionary \
    --db "$DB" \
    --wav "$WAV" \
    --out-dir "$OUT_DIR" \
    --name "leidata_wav1_extend_w${N}_k${K}" \
    --n-windows "$N" \
    --window-offset 0 \
    --min-coverage 0.9 \
    --eps-step-fraction 0.2 \
    --min-repeats 3 \
    --min-template-width 12 \
    --max-template-width 24 \
    --refine-iters 1 \
    "${EXTRA[@]}" \
    >"$STAGE_OUT"
  cp "$STAGE_OUT" "$STAGE_JSON"
  cat "$STAGE_OUT"
  rm -f "$STAGE_OUT"

  PREV_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["out_dir"])' "$STAGE_JSON")"
  DID="$(python -c 'import json,sys; v=json.load(open(sys.argv[1])).get("dictionary_id"); print("" if v is None else v)' "$STAGE_JSON")"
  if [[ -n "$DID" ]]; then
    PREV_ID="$DID"
  fi
  COVERAGE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("coverage"))' "$STAGE_JSON")"
  ADDED="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("n_atoms_added"))' "$STAGE_JSON")"
  STOP="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("stop_reason"))' "$STAGE_JSON")"
  echo "stage_w${N}_k${K}: dictionary_id=${DID:-none} out_dir=$PREV_DIR coverage=$COVERAGE n_atoms_added=$ADDED stop_reason=$STOP" >&2
done

echo "Final out_dir=$PREV_DIR dictionary_id=${PREV_ID:-none}" >&2
if [[ -n "$PREV_ID" ]]; then
  echo "$PREV_ID"
else
  echo "$PREV_DIR"
fi
