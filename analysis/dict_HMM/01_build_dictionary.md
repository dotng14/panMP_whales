# Build a dictionary from a wav file

Learn motifs from one recording. The reference run is `SOCAL_H_76_230427_014845.x.limited_overlap_k1_w40_pc5_tau04_m8x32_refine2` (wav 3). A correct port of this step reproduces that dictionary.

The input is the GPL output of the wav, not the raw samples. GPL is not part of this step. Each window is a whitened spectrogram plus a JSON sidecar and a detection list.

## Spectrogram constants

These values are fixed by the GPL sidecars of the SOCAL recordings. A C program reads them from the sidecar of each window.

| Name | Default | Meaning |
|---|---:|---|
| `sample_freq` | 200000 | Audio samples per second |
| `fftl` | 32768 | FFT length used to build the spectrogram |
| `overlap` | 24576 | Overlap of successive spectrogram columns, in samples |
| `start_freq` | 10 | Lowest stored frequency, Hz |
| `end_freq` | 10000 | Highest stored frequency, Hz |
| `window_length_sec` | 75 | Audio length of one GPL window |
| `n_freq` | 1637 | Frequency rows in a stored window |
| `n_time` | 609 | Time columns in a full window |

`hop = overlap / sample_freq` = 0.12288 seconds per column.

A placement that starts at column `t_start` of a window whose first audio sample is `start_sample` (1-based) begins at

```
((start_sample - 1) / sample_freq) + t_start * hop
```

seconds from the start of the wav.

## Learning constants

Defaults are the values that produced the 35-motif dictionary. Where the command-line default in `learn.py` differs, it is noted.

| Name | Default | Meaning |
|---|---:|---|
| `k` | 1 | Maximum number of placements that may cover one time column |
| `eps` | 0.01 | Stop pursuit when the best explained energy is below this fraction of the first accepted placement |
| `w_min` | 10 | Minimum seed-crop width, in columns |
| `w_max` | 40 | Maximum seed-crop width, in columns |
| `seeds_per_pass` | 10 | Seeds drawn each outer pass. The saved `summary.json` records `m` = 10. Each pass log records `n_sampled` = 32. A port must try both and keep the value that reproduces the saved atoms |
| `n_min` | 10 | Drop a motif with fewer placements than this |
| `refine_iters` | 2 | Rebuild rounds inside one pass. The command-line default is 1 |
| `tau` | 0.4 | Drop a placement whose unexplained fraction exceeds this. The command-line default is 0.9 |
| `max_outer` | 50 | Maximum outer passes |
| `min_coverage` | 0.7 | Stop when the fraction of covered columns exceeds this. The command-line default is 0.5 |
| `n_components` | 5 | Principal components stored for each motif. The command-line default is 1 |
| `seed` | 0 | Seed of the NumPy `default_rng` generator (PCG64). The same generator is required to draw the same crops |

## Inputs

- One whitened spectrogram per window, shape `(n_freq, n_time)`, float64.
- For each window: `window_id`, `start_sample`, `sample_freq`, `overlap`.
- Detections. Each detection has a start time, an end time, and a boundary flag. A detection that overlaps a window becomes a crop of that window. Empty crops and crops whose energy is zero are dropped. The crop's column range is the detection interval mapped into the window with hop `overlap`.

## Algorithm

Sort windows by `(start_sample, window_id)`. Cover counts start at zero for every column.

Repeat up to `max_outer` passes.

1. Eligible seeds are detection crops whose width is between `w_min` and `w_max` and whose columns are all covered fewer than `k` times. If none remain, stop with reason `no_seeds`.
2. Draw `min(seeds_per_pass, number of eligible crops)` crops without replacement. The generator is NumPy PCG64 with `seed`.
3. Each drawn crop becomes a seed motif. Divide the crop by its Euclidean norm. Motif ids start at 0 and increase by one.
4. For each of `refine_iters` rounds, using a copy of the cover counts:
   - Pursuit. At every step, among all motifs and all starts whose width is entirely covered fewer than `k` times, take the start with the largest explained fraction. The explained fraction is specified below. Accept it, increment the cover counts on its columns, and record the placement. The first accepted placement sets `first_energy` to its explained energy. Later placements are accepted only while their explained energy is at least `eps * first_energy`.
   - Append the new placements to this pass's list.
   - Drop motifs with fewer than `n_min` placements, and drop those placements.
   - If any motifs remain, rebuild each one from the patches at its placements (below), rescore every remaining placement, then drop placements whose unexplained fraction exceeds `tau` and motifs that then have fewer than `n_min` placements.
   - Recompute cover counts from every placement kept in earlier passes plus the placements still alive in this pass.
5. If no motif survives the pass, stop with reason `no_survivors`.
6. Otherwise append the surviving motifs and placements to the dictionary. If the covered fraction of columns exceeds `min_coverage`, stop with reason `coverage`.

The saved run finished after 2 passes. Its `summary.json` records `stop_reason` `refine`. The source as it stands now uses the stop reasons above; a port should match the atoms, not the stop-reason string.

### Explained fraction

Let the patch energy at a start be the sum of squares of the spectrogram over the motif's width.

Without a stored mean, the explained energy is the sum of squared dot products of the patch with each unit-norm component. The explained fraction is explained energy divided by patch energy.

With a stored mean, the residual energy is

```
||patch - mean||^2 - sum_i (dot(patch - mean, component_i))^2
```

floored at zero. Explained energy is patch energy minus that residual. The explained fraction is explained energy divided by patch energy. A zero patch energy scores as negative infinity and is never chosen.

Dot products of one component against every start are a correlation. Reverse the component in time, take the FFT of each frequency row of the spectrogram and of the reversed component, multiply, inverse FFT, and sum over frequency. The valid starts are the alignment that places the component fully inside the window.

### Rebuild

The patches are the spectrogram crops at the motif's current placements, each of the motif's width.

The mean patch is their average. The components are the top `n_components` principal components of the centered patches, at most as many as there are patches. Compute them from the eigendecomposition of the Gram matrix of the centered patches. Orient each component so that its dot product with the mean patch is positive. Each component has unit norm. If the patches do not vary, the single component is the normalized mean.

When `n_components` is 1, store that component and no mean. When `n_components` is greater than 1, store the components and the mean patch.

### Rescore

For a placement, the unexplained fraction is `||patch - reconstruction||^2 / ||patch||^2`.

Without a mean, the reconstruction is the sum of `dot(patch, component_i) * component_i`. With a mean, the coefficients are the dots of `patch - mean` with the components, and the reconstruction is `mean` plus the sum of `coefficient_i * component_i`.

## Outputs for the viewer

Write a directory the catalog can register:

- `atoms/template_XXX.npy`, the components, shape `(n_components, n_freq, width)` when `n_components` > 1, otherwise `(n_freq, width)`. `XXX` is the motif id with three digits.
- `atoms/mean_XXX.npy`, the mean patch, only when `n_components` > 1. The repeat view draws this mean.
- `matchings.json`. One object per placement: `motif_id`, `window_id`, `t_start`, `width`, `amplitude`, `unexplained`, and `coeffs` when there is more than one component.
- `windows.json`. One object per window: `start_sample`, `sample_freq`, `overlap`.
- `summary.json`. The knobs, the stop reason, the coverage, and one row per motif with its width, frequency count, component count, and placement count.

The viewer computes a placement's start time from `windows.json` and `t_start` with the formula at the top.

## Acceptance

On wav 3, with these constants:

- 35 motifs
- 1336 placements
- 13901 of 18270 columns covered
- coverage 0.7608648056923919
- maximum cover count 1
- two outer passes, the second ending at that coverage
