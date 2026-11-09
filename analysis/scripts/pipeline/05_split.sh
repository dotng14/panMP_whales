#!/usr/bin/env bash
# Step 5 — split: replace one atom of the dictionary (SOURCE=pan or randomized) by
# two, from the patches step 3 placed it on (or the build's own, before step 3).
#   in : $GPL_DIR, $DICT_DIR, and $MP_DIR when it exists
#   out: $OUT_ROOT/split_<SOURCE>_t<TEMPLATE>/{atoms/, summary.json, matchings.json}
# Usage: SOURCE=pan bash analysis/scripts/pipeline/05_split.sh TEMPLATE_ID [extra flags]
#   Exit 1 means the split was rejected (two atoms did not cut the error by
#   --min-relative-reduction); summary.json says by how much.
set -euo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 TEMPLATE_ID [extra flags]" >&2
  exit 2
fi
TEMPLATE="$1"; shift
source "$(dirname "$0")/config.sh"
require_dir "$GPL_DIR" 01_gpl.sh
require_dir "$DICT_DIR" "02_${SOURCE}_build.sh"
MATCHINGS_ARGS=()
if [[ -d "$MP_DIR" ]]; then
  MATCHINGS_ARGS=(--matchings "$MP_DIR")
fi

run_logged "05_split_${SOURCE}_t$TEMPLATE" -m analysis.dict_learning split \
  --gpl-dir "$GPL_DIR" \
  --dictionary "$DICT_DIR" \
  --template "$TEMPLATE" \
  --out "$OUT_ROOT/split_${SOURCE}_t$TEMPLATE" \
  "${MATCHINGS_ARGS[@]}" \
  --epsilon 0.1 \
  --n-trials 10 \
  --max-refine 10 \
  --min-relative-reduction 0.1 \
  --seed 0 \
  "${CATALOG_ARGS[@]}" \
  "$@"
