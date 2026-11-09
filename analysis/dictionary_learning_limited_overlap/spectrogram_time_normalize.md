# Shift and scale of the dot-product sequence

Dictionary learning in [ten_motif_plan.md](ten_motif_plan.md) uses the GPL whitened spectrogram directly. This shift and scale is applied to each motif's dot-product sequence before specificity is measured, in `analysis/evaluation/core.py`.

Smooth the dot-product sequence along time by a Gaussian with standard deviation `w`. The default `w` is 1 minute. On this recording a column is 0.12288 seconds, so the default is 488 columns. Weights that would fall past either end of the recording are dropped and the remaining weights are renormalized. `mean(t)` is that smoothed value.

`std(t)` is the standard deviation of the original sequence inside the same Gaussian window, with those same edge weights dropped and renormalized. It is the square root of the smoothed second moment minus `mean(t)` squared.

The sequence passed to the specificity measurement is

`(d(t) - mean(t)) / std(t)`.

A time whose standard deviation is 0 is left unchanged. Specificity then finds local maxima of this sequence and forms each peak-to-baseline ratio from it.
