# analysis/

GPL **analysis** science: reference detection (WAV → whitened spectrograms + detections) and dictionary learning on those products. This tree is **not** “offline only.”

The product’s **offline** path is **preprocessing**: [`backend/preprocess.py`](../backend/preprocess.py) (usually via [`backend/scripts/preprocess_cron.sh`](../backend/scripts/preprocess_cron.sh)), which batch-builds spectrogram / `.gpl` caches for the library. Production interactive detection uses the CUDA-capable fork in [`backend/`](../backend/). FastAPI does not import this package directly.

## Song analysis pipeline

Three stages. Payne and McVay (1971) supply the hierarchy the model is meant to recover: **subunit < unit < phrase < theme < song**. A song of type A is six themes in a fixed cycle. Phrase-repeat count varies; theme order does not, and no theme is dropped. See [`../papers/payne_mcvay_1971_song_structure.md`](../papers/payne_mcvay_1971_song_structure.md).

A **motif** in the dictionary sketch below is a convolution kernel. In the fitted song model, a motif is one of 12 panMP prototypes, and each second is a 12-dimensional score against those prototypes. The labeled `motif` column (`U1`…`U11`) in [`../backend/data/song_structure/`](../backend/data/song_structure/) is a reference annotation, loaded by [`../backend/app/services/song_structure.py`](../backend/app/services/song_structure.py) for display. Slots that share one of those names do not share an emission.

```
whitened sequence → 10 Hz energy → panMP scores → Baum–Welch HMM
```

The fit lives in [`HMM/`](HMM/). Decoding a new recording is not in this tree. The handoff is `HMM/song_soft_hmm_t6.npz`.

### Stage 1 — Motif dictionary

**Input.** Whitened window spectrograms in `spectrogram_npy/`, written by [`GPL_Python/gpl_outputs.py`](GPL_Python/gpl_outputs.py).

**Method.** Two existing pieces to combine:

- Matrix Profile over motif length: `panMP` / `pan_profile` in [`matrix_profile/`](matrix_profile/). Finds repeated subsequences without a fixed length.
- Matching Pursuit: the Mallat loop and the randomized dictionary build in [`dict_learning/`](dict_learning/). See [Baseline pipelines](#baseline-pipelines) below.

**Combination (sketch).** Use pan-MP peaks to propose candidate kernels (location and length). Use Matching Pursuit to place, average, and deflate them into a small dictionary.

**Output.** A dictionary of convolution kernels, plus the score of each kernel convolved along the sequence.

**Quality.**

- **Coverage.** Fraction of the sequence where at least one motif score is more than 2 standard deviations above that motif’s mean.
- **Specificity.** For each motif, the ratio of the score at a peak to the scores obtained when that motif is shifted by half its length to either side.

Repeat count, residual energy, and time-bin coverage stay internal stopping rules of the dictionary build. The stage score is coverage and specificity above. The combined design is in [`../plans/stage1_panmp_to_mp_design.md`](../plans/stage1_panmp_to_mp_design.md).

### Stage 2 — Train the HMM

Implemented in [`HMM/`](HMM/). The training recording is the 10 kHz, 10-hour file `SOCAL_H_76_230426_193500_df100.x.x`. GPL at 10–1000 Hz, collapsed to 10 Hz, then 12 panMP prototype scores per second. The motif CSV cuts 53 songs and initializes the Gaussians. It is not the observation. Matching Pursuit is not in this fit.

**States.** Baum–Welch on the graph in [`HMM/song_soft_hmm_t6.dot`](HMM/song_soft_hmm_t6.dot):

- **6 themes**, 28 phrase slots (3+3+6+3+5+8). A song may open on the first slot of any theme, and it may end in any theme. After theme 6, the next theme is theme 1.
- **140 states.** Each slot has five duration states, 1–5 seconds, sharing one diagonal Gaussian over the 12 scores.
- Only the last slot of a theme chooses repeat versus next. Other edges are fixed by the drawing.

**Output.** [`HMM/song_soft_hmm_t6.npz`](HMM/song_soft_hmm_t6.npz): slot means and variances, duration exits, theme-repeat probabilities, the start distribution, the 12 prototypes, and the score normalization. The write-up is [`HMM/song_soft_hmm_t6_report.pdf`](HMM/song_soft_hmm_t6_report.pdf).

**Quality.** Reported in [`HMM/work/train_report.json`](HMM/work/train_report.json):

- Sequence log-likelihood, including nats per second against a 12-dimensional standard normal.
- Posterior-weighted complete-data log-likelihood, split into emission, transition, and start.
- Per-song log-likelihood per second.
- Forward–backward probability on the CSV theme, and expected versus labeled seconds in each theme.
- Likelihood drop when one theme-end probability or one slot mean is nudged. No refit.

### Stage 3 — Inference

Not in this tree. A later decoder loads `HMM/song_soft_hmm_t6.npz` and scores a new recording with the saved prototypes. This folder does not choose a state path.

## Layout

```
analysis/
├── GPL_Python/         # CPU-only Python reference detector (parity baseline)
├── HMM/                # six-theme song model, Baum–Welch fit
├── bands.py            # band energy from raw audio (5 × 100 Hz)
├── matrix_profile/     # matrix profile, pan matrix profile, motif pairs; `pan` CLI
├── dict_learning/      # Mallat MP and the randomized dictionary build; CLI
│   ├── code/           # library (core, io, mp, tools, cli, learn/)
│   ├── experiments/    # compare / bench harnesses (many target removed builders)
│   ├── scripts/        # LeiData bash runners (target removed builders)
│   └── notebooks/      # result / explore notebooks
└── test/               # unit/ and integration/ tests; see test/README.md
```

| Directory                            | Language | Role                                                                                     |
| ------------------------------------ | -------- | ---------------------------------------------------------------------------------------- |
| [`GPL_Python/`](GPL_Python/)         | Python   | Reference detector (STFT, whitening, detect, exports); CLI + tests                       |
| [`matrix_profile/`](matrix_profile/) | Python   | Matrix profile (1 row or many), pan matrix profile over motif length, top motif pairs   |
| [`HMM/`](HMM/)                       | Python   | Six-theme song HMM: panMP motif scores, Baum–Welch, saved parameters                     |
| [`dict_learning/`](dict_learning/)   | Python   | Mallat MP, randomized dictionary build, `learn_from_motifs` trial ([README](dict_learning/README.md)) |
| [`test/`](test/)                     | Python   | Unit and CLI integration tests, coverage gate ([README](test/README.md))                 |

## Responsibilities

| Subtree          | Owns                                                                 | Does not own                          |
| ---------------- | -------------------------------------------------------------------- | ------------------------------------- |
| `GPL_Python/`    | Reference numerical detector + exports (`CSV_*`, `spectrogram_npy/`) | Web API, CUDA, cron preprocess caches |
| `HMM/`           | Baum–Welch fit of the six-theme song graph on panMP scores | Decoding a recording into a state path |
| `matrix_profile/` | Matrix profile and pan matrix profile on any 1-row or multi-row signal | Choosing the dictionary |
| `dict_learning/` | MP and dictionary builds on those exports | Writing library `result/*.gpl` caches |

## Relationship to the web stack

```
GPL_Python ──port science──► backend/gpl_* (+ CUDA) ──┬──► backend/app/ (interactive)
     │                                                 └──► preprocess.py / cron (offline spectrograms)
     └── CSV + sp_whiten ──► dict_learning (MP, randomized build), matrix_profile (pan)
```

When detector math or exports change for correctness, land and validate them in `GPL_Python/`, then port into `backend/` so the web app **and** the preprocess cron pick them up. See [`../documentation/Docs/architecture.md`](../documentation/Docs/architecture.md) (repo layout), [`../documentation/Docs/analysis.md`](../documentation/Docs/analysis.md) (dictionary / peel design), and plans `08` / `10` under [`../documentation/plans/`](../documentation/plans/).

## Baseline pipelines

Three commands, each reading files directly (no catalog database). Tests and
the coverage gate: [`test/README.md`](test/README.md).

### Randomized dictionary build

```bash
python -m analysis.dict_learning randomized-build --gpl-dir GPL_OUT --out DICT \
  --m-per-outer 5 --n-min 10 --min-coverage 0.5 --seed 0
```

Each outer pass samples `--m-per-outer` poorly explained GPL detection patches
as unit-norm atoms, refines them (`--refine-iters` rounds of whole-dictionary
Mallat MP, then a top-`K` PCA rebuild of each atom from its patches, dropping
atoms with fewer than `--n-min` patches), drops placements that leave more than
`--post-refine-max-relative-distance` of their energy unexplained, merges the
survivors, and subtracts them from the residual. It stops once time coverage
exceeds `--min-coverage`. Details: [`dict_learning/README.md`](dict_learning/README.md).

### Mallat matching pursuit

```bash
python -m analysis.dict_learning matching-pursuit --gpl-dir GPL_OUT --dictionary DICT --out MP
```

Places a saved dictionary's atoms on the windows; writes the placements and the
energy left.

### Pan matrix profile

```bash
python -m analysis.matrix_profile pan --gpl-dir GPL_OUT --rate 10 --signal bands \
  --min-len 10 --max-len 50 --step 10 --out PAN --plot
```

| Input | Signal options | Rows |
|---|---|---|
| `--series FILE.npy` | — | 1-D → 1 row; `(rows, T)` as is |
| `--wav FILE --rate HZ` | `--signal rms` (default): RMS per frame | 1 |
| | `--signal bands`: energy in 5 × 100 Hz bands, 0–500 Hz | 5 |
| `--gpl-dir DIR [--rate HZ]` | `--signal energy` (default): whitened spectrogram summed over frequency (the HMM's signal) | 1 |
| | `--signal bands [--band-edges 10,100,…]`: summed within bands (default 100 Hz bands over GPL's range) | bands |
| | `--signal full`: every frequency row | ~100 |

With `--rate`, GPL columns are averaged into `1/rate` s bins by frame centre;
without it, the native STFT columns are kept. Lengths (`--min-len`,
`--max-len`, `--step`) are in samples of that signal. `--engine fast` (default)
uses numba (`mpx_fast` for 1 row, `mpx_channels` for several, the sum of per-row
z-normalized distances); `--engine reference` is the MATLAB port `mpxForPan`,
1 row only and slow.

Output: `pan.npy` (motif length × time; per length the distance ÷ √length,
z-scored, flipped and scaled so the strongest motif is 1; NaN past the last
window), `pairs.json` (most unusual pairs across lengths), `pan.json` (lengths
and how the signal was built), and `pan.png` with `--plot`.

## Setup

From the repo root, in a virtual environment used only for this checkout (see
[`test/README.md`](test/README.md#setup) for why):

```bash
pip install -e '.[test]'      # analysis.* + FastAPI/uvicorn stack (CPU) + pytest
pip install -e '.[gpu]'       # + CuPy for CUDA (GPU servers only)
```

All runtime deps (science + webserver) live in the root [`pyproject.toml`](../pyproject.toml).

## Quick start

```bash
# GPL_Python (module invocation: internal imports are relative)
python3 -m analysis.GPL_Python.process_v4 /path/to/input.wav --output-dir GPL_OUT

# Dictionary, then MP with it
python3 -m analysis.dict_learning randomized-build --gpl-dir GPL_OUT --out DICT --seed 0
python3 -m analysis.dict_learning matching-pursuit --gpl-dir GPL_OUT --dictionary DICT --out MP

# Pan matrix profile on the whitened energy at 10 Hz
python3 -m analysis.matrix_profile pan --gpl-dir GPL_OUT --rate 10 --out PAN

# Tests
python3 -m pytest analysis
```

MATLAB parity helpers (optional): see [`GPL_Python/README.md`](GPL_Python/README.md).
