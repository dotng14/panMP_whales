# song_exp3: Experiment 3

This experiment tries several ways to process the recording and build model inputs, and trains four model approaches on the best inputs:
- **M1:** one theme-level HMM;
- **M2:** song-boundary-first;
- **M3:** unit-based;
- **M4:** six theme HMMs chained into a song model.

All of them are compared with Experiment 2 against a human reference.

## Documents

| File | What it is |
|---|---|
| [plan.md](plan.md) | The approved plan |
| [pipeline.md](pipeline.md) | Every processing option (S, N) and input (X1–X5): what it is, how it is built, its shape and units, what it means, and which models use it |
| [metrics.md](metrics.md) | Every metric and its caveats |
| [reports/exp3_inputs_and_models.md](../../reports/exp3_inputs_and_models.md) | Results. Required sections: previous experiment, current experiment, comparison |
| [notebooks/detect_songs_best_model.ipynb](notebooks/detect_songs_best_model.ipynb) | Runs the best model on the wav, lists every detected song and shows each song's spectrogram (helpers in `detect.py`) |
| [reports/exp3_best_combination.md](../../reports/exp3_best_combination.md) | The best setup (`S2_N1_X2_20s` + M1) on its own: how it works, what it does well and not well |
| [reference/README.md](reference/README.md) | How to hand-label the two 30-minute clips in Audacity |
| `pairs/page/` | The blind phrase-pair judging page (published as an Artifact) |

## Run, in order, from the repository root

```bash
python -m analysis.song_exp3.inputs.screen
```

```bash
python -m analysis.song_exp3.inputs.units_screen
```

```bash
python -m analysis.song_exp3.run_models
```

```bash
python -m analysis.song_exp3.evaluate
```

```bash
python -m analysis.song_exp3.report
```

- `screen` runs the window-input grid: 3 spectrograms × 4 noise options × {X1, X2} × {12 s, 20 s} × {PCA-20, PCA-40}, plus X5.
- `units_screen` runs unit detection (X4) for each spectrogram × noise setting.
- `run_models` picks the inputs from the screening and trains M1–M4.
- `evaluate` scores Exp 2, Exp 2b, M1–M4 and the CSV.
- `report` writes `reports/exp3_inputs_and_models.md` (at the repository root) and `figures/`.

Everything that is computed lands in `work/`, which is git-ignored. To rebuild the pair page: `python -m analysis.song_exp3.pairs.build`.

## The human reference

- **Hand labels:** follow `reference/README.md` and save the Audacity exports as `reference/labels/clip1_labels.txt` and `clip2_labels.txt`. `evaluate` then fills in metric group H.
- **Pair judgements:** made on the published pair page. Claude reads them back into `work/pairs/judgments.json`, and `report` fills in metric group P.

## Code

| Path | Role |
|---|---|
| `inputs/spectrogram.py`, `inputs/pcen.py` | Spectrogram options S1–S3 and noise options N1–N4 |
| `inputs/window_features.py` | X1 window statistics, X2 2D-Fourier patch, standardise + PCA |
| `inputs/frames.py` | X3 frame vectors (0.25 s) |
| `inputs/units.py` | X4 unit detection, measurements and unit types |
| `inputs/screen.py`, `inputs/units_screen.py` | Screening |
| `models/graph.py` | Shared HMM engine: sparse graph with tied product parameters, Gaussian-mixture and categorical emissions, log-domain kernels from Experiment 1 |
| `models/m1_theme.py`, `m2_songfirst.py`, `m3_units.py`, `m4_theme_hmms.py` | The four models |
| `run_models.py`, `evaluate.py`, `figures.py`, `report.py` | Phases 2 and 3 |

Tests: `python -m pytest analysis/test/unit/song_exp3`.
