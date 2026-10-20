#!/usr/bin/env bash
# Incremental extend on LeiData wav1 for window prefixes 1 and 1–2, both at K=1.
#
# Stage 0: cold start on window 1  → catalog seed dictionary + matchings
# Stage 1: extend on windows 1–2 with parent = stage 0 → catalog refinement + matchings
#
# Env overrides:
#   CATALOG_DB      catalog sqlite path
#   MAX_NEW_ATOMS   atoms added per stage (default: 10)
#   INPUT_DICTIONARY_ID  optional parent for stage 0 (else cold start)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

LIBRARY_ROOT="/home/yfreund/whale_processing"
DB="${CATALOG_DB:-$LIBRARY_ROOT/result/dict_learning_catalog.sqlite}"
WAV="$LIBRARY_ROOT/LeiData/SOCAL_H_76_230427_003345.x.wav"
OUT_DIR="$LIBRARY_ROOT/result/LeiData/dict_learning_templates/wav1_extend_w1_w2_k1"
mkdir -p "$OUT_DIR"

MAX_NEW_ATOMS="${MAX_NEW_ATOMS:-10}"
PREV_ID="${INPUT_DICTIONARY_ID:-}"

echo "Incremental K=1 extend: n_windows=1 → 2  db=$DB" >&2
echo "out_dir=$OUT_DIR  max_new_atoms=$MAX_NEW_ATOMS" >&2

verify_catalog() {
  local dict_id="$1"
  local label="$2"
  python - "$DB" "$dict_id" "$label" <<'PY'
import sys
from database import catalog

db, dict_id, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with catalog.connect(db) as conn:
    d = catalog.get_dictionary(conn, dict_id)
    if d is None:
        raise SystemExit(f"{label}: dictionary_id={dict_id} missing from catalog")
    tmpls = catalog.list_dictionary_templates(conn, dict_id)
    wins = catalog.list_dictionary_windows(conn, dict_id)
    matches = catalog.list_matchings(conn, dict_id)
    print(
        f"{label}: dictionary_id={dict_id} source={d.get('source')} "
        f"parent={d.get('parent_dictionary_id')} "
        f"templates={len(tmpls)} windows={len(wins)} matchings={len(matches)}",
        flush=True,
    )
    if not tmpls:
        raise SystemExit(f"{label}: no templates registered")
    if not matches:
        raise SystemExit(f"{label}: no matchings registered (required)")
PY
}

for N in 1 2; do
  STAGE_JSON="$OUT_DIR/stage_w${N}.json"
  EXTRA=(--n-components 1 --max-new-atoms "$MAX_NEW_ATOMS")
  if [[ -n "$PREV_ID" ]]; then
    EXTRA+=(--input-dictionary-id "$PREV_ID")
  fi

  echo "=== stage n_windows=$N K=1 parent=${PREV_ID:-none} ===" >&2
  STAGE_OUT="$(mktemp)"
  python -u -m analysis.dict_learning.code.learn.extend_dictionary \
    --db "$DB" \
    --wav "$WAV" \
    --out-dir "$OUT_DIR" \
    --name "leidata_wav1_extend_w${N}_k1" \
    --n-windows "$N" \
    --window-offset 0 \
    --min-coverage 0.9 \
    --eps-step-fraction 0.5 \
    --min-repeats 3 \
    --min-template-width 12 \
    --max-template-width 24 \
    --refine-iters 1 \
    "${EXTRA[@]}" \
    >"$STAGE_OUT"
  cp "$STAGE_OUT" "$STAGE_JSON"
  cat "$STAGE_OUT"
  rm -f "$STAGE_OUT"

  DID="$(python -c 'import json,sys; v=json.load(open(sys.argv[1])).get("dictionary_id"); print("" if v is None else v)' "$STAGE_JSON")"
  if [[ -z "$DID" ]]; then
    echo "ERROR: stage w=$N did not return a catalog dictionary_id (K=1 must register)" >&2
    exit 1
  fi
  verify_catalog "$DID" "stage_w${N}"
  PREV_ID="$DID"

  COVERAGE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("coverage"))' "$STAGE_JSON")"
  ADDED="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("n_atoms_added"))' "$STAGE_JSON")"
  ATOMS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("n_atoms"))' "$STAGE_JSON")"
  STOP="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("stop_reason"))' "$STAGE_JSON")"
  echo "stage_w${N}_k1: dictionary_id=$DID n_atoms=$ATOMS n_atoms_added=$ADDED coverage=$COVERAGE stop_reason=$STOP" >&2
done

echo "Final dictionary_id=$PREV_ID" >&2
echo "$PREV_ID"
