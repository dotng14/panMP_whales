# panMP_whales

## Command-line tools

Every command, in the order a pipeline uses them. Run them from the repo root
(so `analysis` and `database` import from this checkout, not another one in the
same environment). `--help` on any command prints its full usage.

| Step | Command | Reads | Writes |
|---|---|---|---|
| 1. Spectrogram + detections | `python -m analysis.GPL_Python.process_v4` | WAV (or a folder of WAVs) | GPL output folder |
| 2. Build a dictionary | `python -m analysis.dict_learning randomized-build` | GPL output folder | dictionary folder |
| 3. Place a dictionary | `python -m analysis.dict_learning matching-pursuit` | GPL output folder + dictionary folder | MP folder |
| — Find repeats (no dictionary) | `python -m analysis.matrix_profile pan` | `.npy` series, WAV or GPL output folder | pan folder |
| — Catalog for the web viewer | `python -m database index` / `register` | WAV library, run folders, CSVs | catalog SQLite file |

A whole run, end to end:

```bash
python -m analysis.GPL_Python.process_v4 recording.wav --output-dir GPL_OUT
python -m analysis.dict_learning randomized-build --gpl-dir GPL_OUT --out DICT --seed 0 --catalog catalog.sqlite
python -m analysis.dict_learning matching-pursuit --gpl-dir GPL_OUT --dictionary DICT --out MP --catalog catalog.sqlite
python -m analysis.matrix_profile pan --gpl-dir GPL_OUT --rate 10 --out PAN --plot
```

### 1. GPL: `process_v4`

```bash
python -m analysis.GPL_Python.process_v4 recording.wav --output-dir GPL_OUT
```

| Option | Default | Meaning |
|---|---|---|
| `input_path` | required | A WAV file, or a folder of WAV files |
| `--output-dir` | | Where `spectrogram_npy/` and the detection CSV go |
| `--freq-lo` / `--freq-hi` | 10 / 10000 Hz | Frequency range for spectrograms and detection |
| `--intensity-clip-hi-percentile` | 100 (no trim) | Upper clip for exported spectrograms, as a percentile of `sp_whiten` |

Details: [`analysis/GPL_Python/README.md`](analysis/GPL_Python/README.md).

### 2. `randomized-build`

```bash
python -m analysis.dict_learning randomized-build --gpl-dir GPL_OUT --out DICT \
  --m-per-outer 5 --n-min 10 --min-coverage 0.5 --seed 0 --verbose
```

Each outer pass samples poorly explained detection patches as atoms, refines
them with matching pursuit, keeps the ones enough patches support, and
subtracts them from the residual. It stops at the first of: coverage reached,
`--max-outer` passes, or no patches left to sample.

| Option | Default | Meaning |
|---|---|---|
| `--gpl-dir`, `--out` | required | Input GPL folder, output folder |
| `--seed` | random | Fix it to make a run repeatable |
| `--m-per-outer` | 5 | Atoms sampled per outer pass |
| `--min-coverage` | 0.5 | Stop once this fraction of time bins is explained |
| `--max-outer` | 100 | Stop after this many outer passes |
| `--refine-iters` | 3 | MP + rebuild rounds per pass |
| `--n-min` | 10 | Drop an atom matched to fewer patches than this |
| `--n-components` | 1 | PCA modes per atom; `>1` is subspace MP (not stored in the catalog schema) |
| `--epsilon` | 0.1 | Only sample patches that keep more than this fraction of their energy unexplained |
| `--min-template-width` / `--max-template-width` | 10 / 20 columns | Detections outside this width are not sampled (`none` = no limit) |
| `--post-refine-max-relative-distance` | 0.9 | After refining, drop placements leaving more than this fraction of energy unexplained (`none` = off) |
| `--verbose` | off | One line per outer pass on stderr |
| `--n-windows`, `--window-offset` | all, 0 | Use a slice of the windows |
| `--catalog DB` | off | Also register the output in the catalog |

Plus the [shared MP options](#shared-mp-options). Exit codes: 0 success,
1 no atoms survived, 2 bad input. Output files:
[`analysis/dict_learning/README.md`](analysis/dict_learning/README.md#randomized-build).

### 3. `matching-pursuit`

```bash
python -m analysis.dict_learning matching-pursuit --gpl-dir GPL_OUT --dictionary DICT --out MP
```

Places a saved dictionary's atoms on the windows and writes the placements
(`matchings.json`) and the energy left (`summary.json`).

| Option | Default | Meaning |
|---|---|---|
| `--gpl-dir`, `--dictionary`, `--out` | required | GPL folder, a `randomized-build` output folder, output folder |
| `--max-repeats` | no limit | Stop after this many placements |
| `--disallow-overlap` | off | Don't let placements overlap in time within a window |
| `--n-windows`, `--window-offset` | all, 0 | Use a slice of the windows |
| `--catalog DB` | off | Also register the output (and its dictionary, if needed) |

Plus the shared MP options below.

### Shared MP options

Both `randomized-build` and `matching-pursuit` take these.

| Option | Default | Meaning |
|---|---|---|
| `--mp-mode` | `per_window` | `per_window`: each window on its own, in parallel. `global`: one best atom per step over all windows |
| `--workers` | all CPUs | Processes for `per_window`; `1` runs serially |
| `--compute` | `auto` | `cpu`, `cuda`, or `auto` (GPU only when it works and the problem is large) |
| `--eps-step-fraction` | 0.01 | Stop when a step removes less than this fraction of the first step's energy |
| `--max-relative-distance` | off | Reject placements leaving more than this fraction of energy unexplained |
| `--progress` | off | Progress bars on stderr |
| `--loop-stats` | off | Write MP counters to `loop_stats.json` |

`--mp-mode`, `--workers` and `--compute` also read `DICT_LEARNING_MP_MODE`,
`DICT_LEARNING_MP_WORKERS` and `DICT_LEARNING_MP_COMPUTE` (then `GPL_COMPUTE`).
`--workers` defaults to every CPU, so on a shared server pass a small number.

### `pan` (pan matrix profile)

```bash
python -m analysis.matrix_profile pan --gpl-dir GPL_OUT --rate 10 --signal bands \
  --min-len 10 --max-len 50 --step 10 --out PAN --plot
```

| Option | Default | Meaning |
|---|---|---|
| `--series FILE.npy` / `--wav FILE` / `--gpl-dir DIR` | one required | Input: a 1-D or `(rows, T)` array, a mono WAV, or a GPL folder |
| `--signal` | WAV: `rms`; GPL: `energy` | WAV: `rms` or `bands`. GPL: `energy`, `bands` or `full` (every frequency row) |
| `--rate` | GPL: native columns | Samples per second of the signal; required with `--wav` |
| `--band-edges` | 100 Hz bands | Band edges in Hz for `--signal bands`, e.g. `10,100,200` |
| `--min-len` / `--max-len` / `--step` | 32 / 128 / 32 | Motif lengths to scan, in samples of the signal |
| `--top-k` | 10 | Motif pairs to report |
| `--engine` | `fast` | `fast` (numba) or `reference` (MATLAB port, 1 row, slow) |
| `--threads` | 1 | Threads for the fast engine |
| `--plot` | off | Also write `pan.png` |
| `--out` | required | Output folder |

Writes `pan.npy`, `pairs.json`, `pan.json` (and `pan.png`). The signal
options and output format in detail: [`analysis/README.md`](analysis/README.md#pan-matrix-profile).

### Catalog: `python -m database`

```bash
# Scan everything (defaults come from WAV_LIBRARY_ROOTS, CATALOG_RUNS_ROOTS, CATALOG_CSV_ROOTS)
python -m database index --db catalog.sqlite --wav-root /data/recordings --runs-root runs --csv-root csvs

# Add specific run folders or *.motif_segments.csv files
python -m database register --db catalog.sqlite runs/build_a my.motif_segments.csv --label "first try"
```

The web viewer's backend does the same scan on `POST /api/catalog/reindex`.
Details: [`database/README.md`](database/README.md).

### Tests

```bash
python -m pytest analysis --continue-on-collection-errors
```

The suites, the coverage gate and the parameters the CLI tests use:
[`analysis/test/README.md`](analysis/test/README.md).

## Notes

Deploying
```bash
sudo deploy-webserver --pmp
```

Bug
Chrome has a minimum sampling frequency required in order to play a wav file.
