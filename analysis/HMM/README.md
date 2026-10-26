# HMM

The six-theme song model: the drawing, the Baum–Welch fit, the saved parameters, and the products of the 10-hour run.

## The model and the write-up

- `song_soft_hmm_t6.dot` is the architecture. Twenty-eight phrase slots in six themes. Only the last slot of a theme can repeat or move on. After theme 6, that move returns to theme 1. A song may end in any theme.
- `song_soft_hmm_t6.npz` is the fitted model. It stores the slot means and variances, the duration-exit and theme-repeat probabilities, the start distribution, the 12 motif prototypes, and the score normalization.
- `song_soft_hmm_t6_report.pdf` is the training report: sequence likelihood, nats per second, posterior mass on the labeled themes, theme occupancy, and the fitted branches.
- `plan.md` is the plan that was executed. `instructions.md` is how to read the metrics and how to train again.

## Code

- `__init__.py` marks the folder as a Python package.
- `topology.py` turns the drawing into 140 states: 28 slots, each with five duration states of 1–5 seconds. It builds the allowed transitions. A song may begin at the first slot of any theme, and it may end in any theme. Theme 6 can return to theme 1.
- `observations.py` turns GPL windows into a 10 Hz energy series, runs panMP, keeps the 12 motif prototypes, and scores each second against them. It also reads the motif CSV to cut the 53 songs and to initialize the slots.
- `hmm.py` is Baum–Welch. The five duration states of a slot share one diagonal Gaussian. `fit_metrics` reports the sequence likelihood, the complete-data split, and the forward–backward mass on the labeled theme.
- `run_train.py` is the training entry point (`python -m analysis.HMM.run_train`). `refresh_metrics` recomputes those metrics for the saved parameters and does not refit.
- `train_song_soft_hmm_t6.ipynb` calls that training and prints the metrics.
- `test_soft_hmm.py` checks that the transitions sum to one, that theme 6 can return to theme 1, and that Baum–Welch raises the likelihood on synthetic songs.
- `write_report.py` rebuilds the PDF from the saved parameter file, `work/train_report.json`, and the figures.

## `work/`

Products of the run, not source.

- `work/gpl_193500/` is the GPL output for the 10 kHz file: 480 whitened spectrogram windows, plus the detection `.mat`, `.csv`, `.tlab`, and `panel.bin`.
- `work/series_10hz.npy` is the 10 Hz energy series (360,000 samples).
- `work/motif_scores.npz` is the 12 prototypes and the z-scored score for every second.
- `work/figures/` holds `loglik.png`, `song_loglik.png`, `theme_occupancy.png`, and `branch_sensitivity.png`.
- `work/train_report.json` is the numeric summary of the fit.

`__pycache__` is the compiled-Python cache and is not part of the model.
