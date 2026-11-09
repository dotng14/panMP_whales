# song_hmm: humpback song-session HMM

This folder holds all the work on detecting humpback song sessions and the songs inside them. Every experiment follows Payne & McVay's song structure: themes always run 1 → 6, and a new song starts at the Theme 6 → Theme 1 transition.

The recording is `SOCAL_H_76_230426_193500_df100.x.x.wav` (2 kHz, 10 h). The first 5 h are used for training and the last 5 h for testing.

## Layout

| Folder / file | What it is |
|---|---|
| [reports/](reports/README.md) | **All reports**, one per experiment, plus the best setup on its own |
| [exp1_slot_hmm/](exp1_slot_hmm/README.md) | **Experiment 1.** Per-second slot HMM: 28 phrase slots from the six-theme drawing, using an 18-number input per second |
| [exp2_phrase_hmm/](exp2_phrase_hmm/README.md) | **Experiment 2.** Phrase-window theme HMM: 12 s windows, 295 statistics, PCA-20. Also holds `pipeline_check/` (is the wav → matrix processing correct, and what does it lose) and `diagnostics/` (why Experiment 1 failed; song-boundary checks) |
| [exp3/](exp3/README.md) | **Experiment 3.** 120 input combinations and four models: M1 theme HMM, M2 song-first, M3 units, M4 six theme HMMs. Also holds the human-reference tools (`reference/`, `pairs/`) and the notebook that shows each detected song (`notebooks/detect_songs_best_model.ipynb`) |
| `report_paths.py` | Shared helper: writes every report into `reports/` and fixes its links |

Each experiment folder has its own `plan.md`. Experiments 2 and 3 also have a `pipeline.md` (every input: what it is, how it is built, what it means) and a `metrics.md`. Computed files go in each folder's `work/`, which is git-ignored.

**Tests:** `python -m pytest analysis/test/unit/song_hmm`, organised by experiment in `exp1/`, `exp2/` and `exp3/`.

## Best result so far

Input `S2_N1_X2_20s` (short-FFT spectrogram, median floor, 2D-Fourier phrase patch on 20 s windows) with model M1. On the test half against the CSV: balanced accuracy 0.313, κ 0.233. Details: [reports/exp3_best_combination.md](reports/exp3_best_combination.md).

These numbers measure agreement with the CSV, which is unreliable. The human reference is still pending:
- hand labels, made by following `exp3/reference/README.md`;
- pair judgements, made on the pair-judging page described in `exp3/pairs/README.md`.
