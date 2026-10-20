#!/usr/bin/env bash
# Build an averaged-templates dictionary for any catalog-indexed WAV.
#
# Applies same-logic MP efficiency defaults (threads + CPU Numba) unless you
# already exported overrides. Learning knobs are CLI flags (unchanged math).
#
# Prerequisites: WAV indexed in the catalog, e.g.
#   python -m database.index_existing \
#     --db /path/to/dict_learning_catalog.sqlite \
#     --wav-root /path/to/library
#
# Usage (from repo root):
#   bash analysis/dict_learning/run_build_dictionary.sh \
#     --db /path/to/dict_learning_catalog.sqlite \
#     --wav /path/to/clip.wav \
#     [--out-dir DIR] [--name NAME] [build_dictionary flags…]
#
# Fast recipe (same logic as the ~19-template / ~130-match LeiData tune):
#   bash analysis/dict_learning/run_build_dictionary.sh \
#     --db "$DB" --wav "$WAV" --out-dir "$OUT" --name my_run \
#     --n-windows 4 --n-rounds 20 --eps-step-fraction 0.5 \
#     --min-repeats 3 --min-template-width 12 --max-template-width 24 \
#     --template-method pca --refine-iters 0
#
# Efficiency env (set before invoking, or rely on defaults below):
#   DICT_LEARNING_MP_CANDIDATE_POOL=threads   # default; use process to A/B
#   DICT_LEARNING_MP_COMPUTE=cpu             # default here; auto/cuda also ok
#   DICT_LEARNING_MP_WORKERS=8               # try 8–16 if nproc oversubscribes
#   DICT_LEARNING_LOOP_STATS=1               # optional timing counters
#
# Docs: analysis/dict_learning/README.md §Matching Pursuit performance
#       documentation/Docs/mp-performance.md
#       documentation/plans/13-same-logic-mp-speedups.md

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

if [[ $# -lt 1 ]]; then
  cat <<'USAGE' >&2
Usage: run_build_dictionary.sh --db DB --wav WAV [options…]

Required (also accepted as env CATALOG_DB / BUILD_WAV):
  --db PATH          Catalog SQLite
  --wav PATH|stem    Indexed WAV

Common options (passed through to build_dictionary):
  --out-dir DIR      Template output base (UTC-stamped subdir per run)
  --name NAME        Dictionary / learning-run label
  --n-windows N      --n-rounds N --eps-step-fraction F
  --min-repeats K    --min-template-width W --max-template-width W
  --template-method average|pca --refine-iters N
  --no-progress

See: python -m analysis.dict_learning.code.learn.build_dictionary --help
USAGE
  exit 2
fi

# Same-logic speed defaults (do not override an already-exported value).
export DICT_LEARNING_MP_CANDIDATE_POOL="${DICT_LEARNING_MP_CANDIDATE_POOL:-threads}"
export DICT_LEARNING_MP_COMPUTE="${DICT_LEARNING_MP_COMPUTE:-cpu}"
# Mid/full benches on this host prefer 8 over nproc (16/32 oversubscribe).
if [[ -z "${DICT_LEARNING_MP_WORKERS:-}" ]]; then
  export DICT_LEARNING_MP_WORKERS=8
fi

# Allow CATALOG_DB / BUILD_WAV env to supply required paths without repeating flags.
args=()
have_db=0
have_wav=0
for a in "$@"; do
  case "$a" in
    --db) have_db=1 ;;
    --wav) have_wav=1 ;;
  esac
done
if [[ "$have_db" -eq 0 && -n "${CATALOG_DB:-}" ]]; then
  args+=(--db "$CATALOG_DB")
fi
if [[ "$have_wav" -eq 0 && -n "${BUILD_WAV:-}" ]]; then
  args+=(--wav "$BUILD_WAV")
fi
args+=("$@")

echo "MP efficiency: CANDIDATE_POOL=${DICT_LEARNING_MP_CANDIDATE_POOL} COMPUTE=${DICT_LEARNING_MP_COMPUTE} WORKERS=${DICT_LEARNING_MP_WORKERS:-default}" >&2

exec python -m analysis.dict_learning.code.learn.build_dictionary "${args[@]}"
