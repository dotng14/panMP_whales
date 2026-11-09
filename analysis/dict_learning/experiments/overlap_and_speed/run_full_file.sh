#!/usr/bin/env bash
# Section 7: the two-stage recipe on all 30 windows of 014845.
# Stage 1: no overlap, n_min 3 or 5. Stage 2 from each: partial 0.05 or 0.10, nesting 0.5, 10 seeds a pass.
set -u
cd "$(dirname "$0")/../../../.."
G=/home/yfreund/whale_processing/result/LeiData/SOCAL_H_76_230427_014845.x.gpl/output
R=${R:-$HOME/whale_runs/014845_overlap/full}
mkdir -p $R
BASE="--gpl-dir $G --max-outer 20 --min-coverage 0.5 --refine-iters 1 --workers 2 --pan-threads 2 --overlap limited --verbose --loop-stats"
run() { local name=$1; shift; /usr/bin/time -v python -m analysis.dict_learning pan-build $BASE "$@" --out $R/$name > $R/$name.log 2>&1; echo "$name exit $?" >> $R/done.txt; }
for nmin in 3 5; do
  run s1_n$nmin --m-per-outer 5 --n-min $nmin --max-partial-overlap 0 --max-nested-ratio 0
  for p in 05 10; do
    run s2_n${nmin}_p$p --m-per-outer 10 --n-min $nmin --max-partial-overlap 0.$p --max-nested-ratio 0.5 --init-dictionary $R/s1_n$nmin
  done
done
echo ALL_DONE >> $R/done.txt
