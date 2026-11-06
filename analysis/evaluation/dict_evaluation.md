# Dictionary evaluation

Add `evaluate_dictionary` in `analysis/evaluation/core.py`. The input is a wave file. The function builds the GPL whitened spectrogram of that file and evaluates the dictionary on it. Detections are the kept matching-pursuit placements on that spectrogram. Time is one column per spectrogram bin. A placement of width `W` at bin `t` covers `[t, t+W)`.

**Coverage.** Fraction of columns covered by at least one placement. This is `n_bins_covered / n_bins_total` from `coverage_from_matchings`.

**Over-coverage.** Sum of the placement lengths, divided by the number of covered columns. Overlaps count in the numerator and once in the denominator. The value is the average number of detections on each covered column, so 1 means no overlap. No covered columns makes this undefined.

**Specificity.** One distribution per motif. A local maximum is a column where the motif fits and the dot product is strictly larger than both neighboring columns. Specificity at that local maximum `t*` is the smaller relative drop of the shifts that stay inside the recording:

`(dot(t*) - dot(t*+d)) / dot(t*)`

`d` is `+k` and `-k`. The distribution collects these ratios over all local maxima. The filtered distribution removes ratios smaller than the largest ratio divided by `m`. The largest ratio is always kept, so a negative largest ratio is not removed by the division. The one-number summary of the filtered distribution is its median.

Parameters of `evaluate_dictionary`:

1. Dictionary. Default: dictionary 276.
2. Wave file. Default: `/home/yfreund/whale_processing/LeiData/SOCAL_H_76_230427_014845.x.wav`, the third LeiData recording. Dictionary 276 was learned on the whitened spectrogram of this file.
3. `k`, the specificity shift in spectrogram columns. For a motif of width `W`, the default is `k = W // 2`, so each motif has its own shift.
4. `m`, the positive factor in the filter cutoff. Default: 2.

The function returns coverage, over-coverage, and, for each motif, the filtered specificity distribution and its median. Report coverage, over-coverage, and those medians for dictionary 276 on the default wave file in `analysis/dict_learning/notebooks/dict276_014845_convolution.ipynb`.
