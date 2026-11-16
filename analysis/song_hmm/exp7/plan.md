# Experiment 7: dictionary evaluation gate on wav23

**Goal:** Decide which dict_learning method/settings produce a motif alphabet that can support **phrases** before any further theme HMM work.

**Recording:** LeiData `wav23.wav` (200 kHz, 1.25 h), GPL `wav23.gpl/output` (60 × ~75 s windows).

**Arms**

1. `randomized-build` + `matching-pursuit` parameter grid  
2. `learn_from_motifs` (matrix-profile seeds → MP) on the full file  

**Metrics:** phrase-readiness (period-3 repeats, multi-motif phrases, gate) over raw coverage.  

**Out of scope:** theme–slot HMM retrain.

See [pipeline.md](pipeline.md), [inventory.md](inventory.md), report [`../reports/exp7_dict_evaluation.md`](../reports/exp7_dict_evaluation.md).
