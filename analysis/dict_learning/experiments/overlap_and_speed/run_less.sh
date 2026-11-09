#!/usr/bin/env bash
# Tighter overlap limits, n_min 3, to the normal stop.
set -u
cd "$(dirname "$0")/../../../.."
G=/home/yfreund/whale_processing/result/LeiData/SOCAL_H_76_230427_014845.x.gpl/output
R=${R:-$HOME/whale_runs/014845_overlap/less}
mkdir -p $R
BASE="--gpl-dir $G --n-windows 10 --m-per-outer 5 --max-outer 10 --min-coverage 0.5 --refine-iters 1 --workers 2 --verbose --loop-stats --n-min 3 --overlap limited"
run() { local name=$1; shift; /usr/bin/time -v python -m analysis.dict_learning "$@" $BASE --out $R/$name > $R/$name.log 2>&1; echo "$name exit $?" >> $R/done.txt; }
for spec in "p10_n50 0.1 0.5" "p00_n50 0 0.5" "p00_n00 0 0"; do
  set -- $spec; tag=$1; p=$2; n=$3
  run pan_$tag pan-build --pan-threads 2 --max-partial-overlap $p --max-nested-ratio $n
  for seed in 0 1 2; do run rand_${tag}_s$seed randomized-build --seed $seed --max-partial-overlap $p --max-nested-ratio $n; done
done
echo ALL_DONE >> $R/done.txt
