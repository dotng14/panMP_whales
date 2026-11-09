# Analysis CLIs

Run from the repo root. Exit code 0 is success. Exit code 2 is bad input, printed on stderr as `error: …`. `randomized-build` exits 1 when no atom survives.

## `python -m analysis.dict_learning`

Four commands. All read a GPL output folder: `spectrogram_npy/spwhiten_*.npy` with JSON sidecars, windows in time order. `randomized-build` also requires `CSV_*.csv` detections. `--n-windows` and `--window-offset` (default 0) take the slice `[offset, offset + n)`. `pan-build` and `split` are described in the root [`README.md`](../README.md#2b-pan-build) and [`dict_learning/README.md`](dict_learning/README.md); every file they read and write is in [`FILES.md`](../FILES.md).

```bash
python -m analysis.dict_learning randomized-build --gpl-dir GPL_OUT --out DICT_DIR
python -m analysis.dict_learning pan-build --gpl-dir GPL_OUT --out DICT_DIR
python -m analysis.dict_learning matching-pursuit --gpl-dir GPL_OUT --dictionary DICT_DIR --out MP_DIR
python -m analysis.dict_learning split --gpl-dir GPL_OUT --dictionary DICT_DIR --template 3 --out SPLIT_DIR
```

`--catalog DB` registers the output folder in that SQLite catalog after a successful run.

### Shared matching-pursuit knobs

| Flag | Default | Meaning |
|---|---|---|
| `--eps-step-fraction` | `0.01` | Stop when a step removes less than this fraction of the first step's energy. |
| `--max-relative-distance` | off (`none`) | Reject a placement that leaves more than this fraction of the patch energy unexplained. In `[0, 1]`. |
| `--mp-mode` | `per_window` | `per_window`: one Mallat loop per window, atoms do not cross windows. `global`: one best atom per step over the stack. Env `DICT_LEARNING_MP_MODE`. |
| `--workers` | all CPUs | Worker processes for `per_window`. `1` is serial. Env `DICT_LEARNING_MP_WORKERS`. |
| `--compute` | `auto` | `score_shifts` backend. `auto` uses CUDA when CuPy works and the problem is at least `DICT_LEARNING_MP_GPU_MIN_OPS` operations (default 2e6); otherwise numba, then NumPy. `cuda` forces the GPU. `cpu` stays on the CPU. Env `DICT_LEARNING_MP_COMPUTE`, then `GPL_COMPUTE`. |
| `--progress` | off | Progress bars on stderr. |
| `--loop-stats` | off | Write `loop_stats.json` (loop counts and wall time). |

### `randomized-build`

Sample unexplained detection patches, turn each into a unit-norm atom, refine by matching pursuit and top-k PCA, and repeat on the residual until time coverage reaches `--min-coverage` or `--max-outer` passes run.

| Flag | Default | Meaning |
|---|---|---|
| `--m-per-outer` | `5` | New seeds sampled each outer pass. |
| `--min-coverage` | `0.2` | Stop when this fraction of time bins is covered. |
| `--max-outer` | `10` | Outer-pass cap. |
| `--refine-iters` | `1` | MP-and-rebuild rounds inside one pass. |
| `--n-components` | `1` | PCA components kept when an atom is rebuilt. `1` is a rank-1 atom. |
| `--n-min` | `10` | Drop an atom with fewer than this many patches. |
| `--epsilon` | `0.9` | A detection crop is a candidate only when its residual energy fraction is above this. |
| `--min-template-width` | `10` | Shortest atom, in time bins. `none` lifts the limit. |
| `--max-template-width` | `20` | Longest atom, in time bins. `none` lifts the limit. |
| `--post-refine-max-relative-distance` | `0.9` | After refine, drop placements that leave more than this fraction unexplained. |
| `--seed` | unset | NumPy random seed. |
| `--verbose` | off | One stderr line per outer pass. |

**Writes** under `--out`:

| File | Content |
|---|---|
| `atoms/template_NNN.npy` | One atom, shape `(F, W)` or `(k, F, W)`. |
| `summary.json` | Knobs, `stop_reason`, coverage, energies, and the atom list (`template_id`, `width`, `n_freq`, `n_components`, `n_patches`, `source_segment_id`, `path`). |
| `matchings.json` | One row per placement: `template_id`, `window_id`, `t_start`, `amplitude`, `energy`, and `coeffs` when the atom has more than one component. |
| `outer_logs.jsonl` | One JSON object per outer pass. |
| `loop_stats.json` | Only with `--loop-stats`. |

Stdout is one JSON object: `out`, `stop_reason`, `n_atoms`, `n_matchings`, `time_coverage_fraction`.

### `matching-pursuit`

Place the atoms listed in a `randomized-build`, `pan-build` or `split` `summary.json` on the GPL windows. Rank-1 dictionaries are matched with normalized patches.

| Flag | Default | Meaning |
|---|---|---|
| `--dictionary` | required | Folder that contains `summary.json` and `atoms/`. |
| `--max-repeats` | off | Stop after this many accepted placements. |
| `--disallow-overlap` | off | A new placement may not cover a frame already taken. |

**Writes** `summary.json` (energies, `stop_reason`, time and window coverage) and `matchings.json` (same row shape as above). Stdout: `out`, `stop_reason`, `n_matchings`.

### `python -m analysis.dict_learning.code.learn.learn_from_motifs`

Stitch the whitened spectrogram, find the strongest motif of the given lengths, seed an atom, refine, and repeat on the residual until frame coverage reaches `--min-coverage`.

```bash
python -m analysis.dict_learning.code.learn.learn_from_motifs --gpl-dir GPL_OUT --out OUT
```

| Flag | Default | Meaning |
|---|---|---|
| `--gpl-dir` | the 014845 LeiData GPL output | GPL folder or its `spectrogram_npy/` directory. |
| `--lengths` | `5,10` | Motif lengths in seconds. Width is `round(length × frame rate)`. |
| `--min-coverage` | `0.5` | Stop when this fraction of frames is covered. |
| `--refine-iters` | `1` | Averaging rounds after each new motif. |
| `--eps-step-fraction` | `0.01` | Same energy step stop as above. |
| `--max-relative-distance` | `0.859` | Reject a placement above this unexplained-energy fraction. `0` is exact, `1` accepts all. |
| `--nthreads` | `8` | Matrix-profile threads. |
| `--max-rounds` | `100` | Round cap. |
| `--no-progress` | off | Hide the progress bar. |

**Writes** `detection_map.npy` (frame label, `-1` uncovered), `residual.npy`, `template_NNN.npy`, `matchings.json`, `summary.json` (`stop_reason`, coverage, rounds). Exit code 1 if `--gpl-dir` is missing.

## `python -m analysis.matrix_profile pan`

Matrix profile over a sweep of motif lengths. Exactly one source.

```bash
python -m analysis.matrix_profile pan --series x.npy --out OUT
python -m analysis.matrix_profile pan --wav clip.wav --rate 100 --signal bands --out OUT
python -m analysis.matrix_profile pan --gpl-dir GPL_OUT --rate 10 --signal full --out OUT
```

**Input.**

- `--series`: a `.npy` array, 1-D or `(rows, time)`.
- `--wav`: a WAV. `--rate` is required and must divide the file's sample rate. `--signal rms` (default) is one RMS row. `--signal bands` is `analysis.bands` (five 100 Hz bands, 0–500 Hz).
- `--gpl-dir`: whitened windows. `--signal energy` (default) sums frequency. `bands` sums within Hz edges, default every 100 Hz from the spectrogram's start to its end; `--band-edges 10,100,200` overrides that. `full` keeps every frequency row. With `--rate`, columns are averaged into bins of `1/rate` seconds by STFT-frame centre time. Without it, the native STFT hop is kept.

Lengths `--min-len`, `--max-len`, `--step` are in samples of that signal. The series must be at least `max-len + max(max-len − 2, 4)` samples long. `--min-len` is at least 4.

| Flag | Default | Meaning |
|---|---|---|
| `--min-len` | `32` | Shortest motif, samples. |
| `--max-len` | `128` | Longest motif, samples. |
| `--step` | `32` | Length increment. Lengths are `min, min+step, …` through `max`. |
| `--top-k` | `10` | How many pairs to keep. A later pair's start must fall outside every copy already chosen. |
| `--engine` | `fast` | `fast` is `mpx_fast`, or `mpx_channels` when there is more than one row. `reference` is the MATLAB-parity `mpxForPan` and accepts one row only. |
| `--threads` | `1` | Threads for the `fast` engine. |
| `--disallow-overlap` | off | Exclusion zone equals the motif length, so the two copies cannot overlap. Reported pairs also cannot overlap a pair already chosen. |
| `--progress` | off | Progress bar over the motif lengths. |
| `--plot` | off | Also write `pan.png`. |
| `--out` | required | Output folder. |

**Writes**

| File | Content |
|---|---|
| `pan.npy` | Score surface, shape `(n_lengths, time)`. Each length is divided by `sqrt(length)`, z-scored, then scaled so the strongest motif in the sweep is 1 and the weakest is 0. Positions past the last window of a length are NaN. |
| `pan.json` | `lengths`, `rows`, `n_samples`, `rate`, `engine`, and how the signal was built. |
| `pairs.json` | Up to `--top-k` pairs: `length`, `start_a`, `start_b`, `distance`, `zscore`. |
| `pan.png` | The surface, only with `--plot`. |

Stdout: `out`, `rows`, `n_pairs`.
