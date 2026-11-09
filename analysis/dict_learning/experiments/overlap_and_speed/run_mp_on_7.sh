#!/usr/bin/env bash
# Section 5: re-place the no-overlap pan-build dictionary (pmp #7) with looser
# overlap limits, with and without the build's 0.9 placement gate.
set -u
cd "$(dirname "$0")/../../../.."
G=/home/yfreund/whale_processing/result/LeiData/SOCAL_H_76_230427_014845.x.gpl/output
D=${D:-$HOME/whale_runs/014845_overlap/less/pan_p00_n00}
R=${R:-$HOME/whale_runs/014845_overlap/mp_on_7}
mkdir -p $R
for spec in "none 0 0" "p10 0.1 0" "p25 0.25 0"; do
  set -- $spec
  for gate in "" "--max-relative-distance 0.9"; do
    name=$1${gate:+_gated}
    /usr/bin/time -v python -m analysis.dict_learning matching-pursuit --gpl-dir $G --dictionary $D --out $R/$name \
      --n-windows 10 --workers 2 --overlap limited --max-partial-overlap $2 --max-nested-ratio $3 $gate > $R/$name.log 2>&1
  done
done
