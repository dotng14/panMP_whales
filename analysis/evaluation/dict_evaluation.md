# Dictionary evaluation

Add `evaluate_dictionary` in `analysis/evaluation/core.py`. The input is a wave file. The function builds the GPL whitened spectrogram of that file and evaluates the dictionary on it. Detections are the kept matching-pursuit placements on that spectrogram. Time is one column per spectrogram bin. A placement of width `W` at bin `t` covers `[t, t+W)`.

**Coverage.** Fraction of columns covered by at least one placement. This is `n_bins_covered / n_bins_total` from `coverage_from_matchings`.

**Over-coverage.** Sum of the placement lengths, divided by the number of covered columns. Overlaps count in the numerator and once in the denominator. The value is the average number of detections on each covered column, so 1 means no overlap. No covered columns makes this undefined.

**Maximum cover count.** The largest number of placements on any one column. A 1-cover dictionary has maximum 1.

**Residual.** For a kept placement, score a motif on the patch that starts at the same column and has that motif's width, when the patch stays inside the window. With a stored mean, the reconstruction is the mean plus the components. Without a stored mean, it is the projection of the patch onto the components. The residual is the unexplained fraction

`‖patch − reconstruction‖² / ‖patch‖²`.

A motif's residual is the average of this fraction over its own placements. The median of those fractions is reported beside the average.

**Residual gap.** The runner-up at a placement is the other motif with the smallest residual there. The gap is the runner-up residual minus the assigned motif's residual. A motif's gap is the average of those differences over its placements, and the median of the same differences is reported beside it. The dictionary's gap is the smallest of the motif averages. The usual runner-up of a motif is the other motif that is the runner-up on the most of its placements.

**Coverage ratio.** A motif's coverage is the number of distinct time columns its placements cover. The coverage ratio is the largest of these counts divided by the smallest. The ratio is undefined when a motif covers no columns.

**Placement ratio.** The number of kept placements of the motif with the most placements, divided by the number kept by the motif with the fewest. This is the balance check used with the coverage ratio. The ten-motif dictionary was balanced on column coverage. The twenty-motif search was balanced on placement counts. Both ratios are reported.

**Motif count and components.** The number of motifs, and the number of components stored in each atom.

**Specificity.** One distribution per motif. Before the ratios are formed, shift and scale the dot-product sequence as in [spectrogram_time_normalize.md](../dictionary_learning_limited_overlap/spectrogram_time_normalize.md). A local maximum is a column where the motif fits and this shifted and scaled sequence is strictly larger than both neighboring columns. Specificity at that local maximum `t*` is the ratio of the shifted and scaled value at `t*` to the same sequence smoothed by a Gaussian with standard deviation `k`:

`shifted(t*) / smoothed(t*)`

`smoothed` is the shifted and scaled sequence convolved with the Gaussian. The distribution collects these ratios over all local maxima. The one-number summary is the 90th percentile of that distribution.

Parameters of `evaluate_dictionary`:

1. Dictionary. Default: dictionary 276.
2. Wave file. Default: `/home/yfreund/whale_processing/LeiData/SOCAL_H_76_230427_014845.x.wav`, the third LeiData recording. Dictionary 276 was learned on the whitened spectrogram of this file.
3. `k`, the standard deviation of the smoothing Gaussian, in spectrogram columns. For a motif of width `W`, the default is `k = W // 2`, so each motif has its own standard deviation.

The function returns coverage, over-coverage, the maximum cover count, the coverage ratio, the placement ratio, the minimum gap, and, for each motif, the mean and median residual, the mean runner-up residual, the mean and median gap, the usual runner-up, the column coverage, the placement count, the component count, and the specificity distribution with its 90th percentile. Report coverage, over-coverage, and those 90th percentiles for dictionary 276 on the default wave file in `analysis/dict_learning/notebooks/dict276_014845_convolution.ipynb`.
