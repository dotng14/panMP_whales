#!/usr/bin/env bash
# To the normal stopping rule (max_outer 10, min_coverage 0.5), now that a run takes seconds.
set -u
cd "$(dirname "$0")/../../../.."
G=/home/yfreund/whale_processing/result/LeiData/SOCAL_H_76_230427_014845.x.gpl/output
R=${R:-$HOME/whale_runs/014845_overlap/grid}
mkdir -p $R
BASE="--gpl-dir $G --n-windows 10 --m-per-outer 5 --max-outer 10 --min-coverage 0.5 --refine-iters 1 --workers 2 --verbose --loop-stats"
run() {  # name, args...
  local name=$1; shift
  /usr/bin/time -v python -m analysis.dict_learning "$@" $BASE --out $R/$name > $R/$name.log 2>&1
  echo "$name exit $?" >> $R/done.txt
}
for spec in "free 10" "limited 10" "limited 5" "limited 3"; do
  set -- $spec; mode=$1; nmin=$2
  run pan_${mode}_n${nmin} pan-build --pan-threads 2 --overlap $mode --n-min $nmin
  for seed in 0 1 2; do
    run rand_${mode}_n${nmin}_s${seed} randomized-build --seed $seed --overlap $mode --n-min $nmin
  done
done
echo ALL_DONE >> $R/done.txt
