#!/usr/bin/env bash
# Step 6 — register every output folder of this recording in the catalog
# (for folders made without CATALOG set). Registering again updates in place.
#   in : $CATALOG, and whichever of the step folders exist
#   out: rows in $CATALOG
# Usage: CATALOG=catalog.sqlite bash analysis/scripts/pipeline/06_register_catalog.sh [--label TEXT]
set -euo pipefail
source "$(dirname "$0")/config.sh"
if [[ -z "$CATALOG" ]]; then
  echo "error: set CATALOG to the catalog SQLite file" >&2
  exit 2
fi
FOLDERS=()
for dir in "$RANDOMIZED_DIR" "$PAN_BUILD_DIR" "$OUT_ROOT"/matching_pursuit_* "$OUT_ROOT"/split_*; do
  [[ -f "$dir/summary.json" ]] && FOLDERS+=("$dir")
done
if [[ ${#FOLDERS[@]} -eq 0 ]]; then
  echo "error: no dictionary, matching-pursuit or split folder under $OUT_ROOT" >&2
  exit 2
fi

run_logged 06_register_catalog -m database register --db "$CATALOG" "${FOLDERS[@]}" "$@"
