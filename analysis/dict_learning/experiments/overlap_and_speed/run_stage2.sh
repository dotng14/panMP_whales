#!/usr/bin/env bash
# Section 6: continue the no-overlap pan-build (pmp #7) with looser overlap limits.
set -u
cd "$(dirname "$0")/../../../.."
G=/home/yfreund/whale_processing/result/LeiData/SOCAL_H_76_230427_014845.x.gpl/output
D=${D:-$HOME/whale_runs/014845_overlap/less/pan_p00_n00}
R=${R:-$HOME/whale_runs/014845_overlap/stage2}
mkdir -p $R
for spec in "p00_n00 0 0" "p00_n50 0 0.5" "p10_n50 0.1 0.5" "p25_n50 0.25 0.5"; do
  set -- $spec
  /usr/bin/time -v python -m analysis.dict_learning pan-build --gpl-dir $G --n-windows 10 --m-per-outer 5 \
    --max-outer 10 --min-coverage 0.5 --refine-iters 1 --workers 2 --pan-threads 2 --n-min 3 \
    --overlap limited --max-partial-overlap $2 --max-nested-ratio $3 --init-dictionary $D \
    --verbose --loop-stats --out $R/$1 > $R/$1.log 2>&1
done
