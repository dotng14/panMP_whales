#!/usr/bin/env bash
# Section 8: the two-stage experiment (sections 6 and 7) with 5-component atoms.
# Stage 1 has no overlap; stage 2 continues from it with nesting and a little
# partial overlap. Same settings as the rank-1 runs except --n-components 5.
set -u
cd "$(dirname "$0")/../../../.."
G=/home/yfreund/whale_processing/result/LeiData/SOCAL_H_76_230427_014845.x.gpl/output
R=${R:-$HOME/whale_runs/014845_overlap/pc5}
K=${K:-5}
N_WINDOWS=${N_WINDOWS:-10}
mkdir -p "$R"
build() {  # name, extra args...
  local name=$1; shift
  /usr/bin/time -f "%e s wall" python -m analysis.dict_learning pan-build --gpl-dir $G --n-windows $N_WINDOWS \
    --max-outer 10 --min-coverage 0.5 --refine-iters 1 --workers 2 --pan-threads 2 --n-min 3 \
    --n-components $K --overlap limited "$@" --verbose --loop-stats --out "$R/$name" > "$R/$name.log" 2>&1
}
# Round 1 (ROUND2 unset): stage 1 with no overlap at 5 and 10 seeds a pass, then
# stage 2 from each at partial 0.05 / 0.10, nested 0.5, 10 seeds a pass.
if [ "${ROUND2:-0}" != 1 ]; then
  [ -d "$R/s1_m5" ] || build s1_m5 --m-per-outer 5 --max-partial-overlap 0 --max-nested-ratio 0
  build s1_m10 --m-per-outer 10 --max-partial-overlap 0 --max-nested-ratio 0
  for s1 in s1_m5 s1_m10; do
    for p in 0.05 0.10; do
      build "${s1}_p${p#0.}" --m-per-outer 10 --max-partial-overlap $p --max-nested-ratio 0.5 --init-dictionary "$R/$s1"
    done
  done
  echo done > "$R/done.txt"
fi

# Round 2: the 0.5 target stopped stage 2 after one pass. Target 0.7, up to 20 passes,
# for 5 components and for rank 1 (K=1), so both stages get to run.
if [ "${ROUND2:-0}" = 1 ]; then
  rm -f "$R/done2.txt"
  for k in 5 1; do
    K=$k build t70_k${k}_s1 --m-per-outer 5 --max-partial-overlap 0 --max-nested-ratio 0 --max-outer 20 --min-coverage 0.7
    for p in 0.05 0.10; do
      K=$k build "t70_k${k}_p${p#0.}" --m-per-outer 10 --max-partial-overlap $p --max-nested-ratio 0.5 \
        --max-outer 20 --min-coverage 0.7 --init-dictionary "$R/t70_k${k}_s1"
    done
  done
  echo done > "$R/done2.txt"
fi
