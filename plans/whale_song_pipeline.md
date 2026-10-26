---

name: Whale song pipeline
overview: "The fitted path is GPL windows to a 10 Hz series, 12 panMP scores, then Baum–Welch in analysis/HMM. A matching-pursuit dictionary exists and is not part of that fit. Decoding a new recording is not in this repository."
todos:

- id: stage1-motifs
content: Score motifs by coverage and half-length specificity. learn_from_motifs already grows a dictionary; it does not feed the HMM.
status: pending
- id: stage2-hmm
content: Baum–Welch fit of the six-theme song graph on 12 panMP scores, in analysis/HMM.
status: completed
- id: stage3-decode
content: A later decoder loads analysis/HMM/song_soft_hmm_t6.npz. Not in this repository.
status: pending
isProject: false

---

# Whale song analysis pipeline

The fit is in [analysis/HMM/](analysis/HMM/). Module map for motifs and dictionaries: [analysis/analysis.md](analysis/analysis.md). How to read the fit: [analysis/HMM/instructions.md](analysis/HMM/instructions.md).

Payne and McVay (1971): **subunit < unit < phrase < theme < song**. A type-A song is six themes in a fixed order. Phrase-repeat count varies; theme order does not, and no theme is dropped. See [papers/payne_mcvay_1971_song_structure.md](papers/payne_mcvay_1971_song_structure.md). The graph has no edge from theme 6 back to theme 1. Each of the 53 training songs is its own sequence; they share one parameter set.

Two different things are called a motif. In the fit, a motif is one of 12 panMP prototypes, and each second is a 12-dimensional score against them. In [analysis/dict_learning/](analysis/dict_learning/), a motif is a time–frequency patch grown by matching pursuit. The labeled `motif` column (`U1`…`U11`) in [backend/data/song_structure/](backend/data/song_structure/) names slots on the drawing. It is not an emitted symbol. Slots that share a name do not share a Gaussian.

```mermaid
flowchart LR
  gpl[Whitened windows] --> hz[10 Hz energy]
  hz --> scores[12 panMP scores]
  scores --> hmm[Baum-Welch]
  hmm --> saved[song_soft_hmm_t6.npz]
```

## Fitted path

**Recording.** `SOCAL_H_76_230426_193500_df100.x.x_10kHz.wav`: 10 kHz, mono, 36,000 s, upsampled from 2 kHz, so GPL is limited to 10–1000 Hz. 480 windows of 75 s under `analysis/HMM/work/gpl_193500/`.

**Scores.** [analysis/HMM/observations.py](analysis/HMM/observations.py) sums each whitened window over frequency and averages to 10 Hz (360,000 samples). panMP, via [analysis/matrix_profile/](analysis/matrix_profile/), searches lengths 1–5 s and keeps the 12 strongest peaks. Each second is the mean z-normalized correlation with each prototype, then each of the 12 scores is z-scored over the 10 hours. The saved run's peaks are all 1 s. Retraining reuses `analysis/HMM/work/motif_scores.npz` when it is present.

**Labels.** [backend/data/song_structure/SOCAL_H_76_230426_193500_df100.x.x.motif_segments.csv](backend/data/song_structure/SOCAL_H_76_230426_193500_df100.x.x.motif_segments.csv) cuts the 53 songs and initializes the Gaussians. It is not the observation.

**States.** Baum–Welch on [analysis/HMM/song_soft_hmm_t6.dot](analysis/HMM/song_soft_hmm_t6.dot), in [analysis/HMM/hmm.py](analysis/HMM/hmm.py) and [analysis/HMM/topology.py](analysis/HMM/topology.py):

- **6 themes**, 28 phrase slots (3+3+6+3+5+8). A song may open on the first slot of any theme. Theme 6 repeats until the song ends.
- **140 states.** Each slot has five duration states, 1–5 s, sharing one diagonal Gaussian over the 12 scores.
- Only the last slot of a theme chooses repeat versus next.

**Output.** [analysis/HMM/song_soft_hmm_t6.npz](analysis/HMM/song_soft_hmm_t6.npz): slot means and variances, duration exits, theme-repeat probabilities, the start distribution, the 12 prototypes, and the score normalization. Report: [analysis/HMM/song_soft_hmm_t6_report.pdf](analysis/HMM/song_soft_hmm_t6_report.pdf). Entry point: `python -m analysis.HMM.run_train`.

**Quality** in [analysis/HMM/work/train_report.json](analysis/HMM/work/train_report.json):

- Sequence log-likelihood, total and in nats per second, against a 12-dimensional standard normal.
- Posterior-weighted complete-data log-likelihood, split into emission, transition, and start.
- Per-song log-likelihood per second.
- Forward–backward probability on the CSV theme, and expected versus labeled seconds in each theme.
- Likelihood drop when one theme-end probability or one slot mean is nudged. No refit.

This repository does not choose a state path.

## Motif dictionary, not in the fit

[analysis/dict_learning/code/learn/learn_from_motifs.py](analysis/dict_learning/code/learn/learn_from_motifs.py) searches the stitched whitened spectrogram with the matrix profile, seeds a dictionary from the strongest 5 s or 10 s pair, and runs matching pursuit on the residual. It stops when matchings cover half the frames. That frame coverage is a stopping rule. It is not a score the HMM sees.

Still open, and not implemented:

- **Coverage.** Fraction of time where at least one motif score is more than 2 standard deviations above that motif's mean.
- **Specificity.** For each motif, the score at a peak divided by the gaussian average using a gaussian whose sigma is the length of the motif.

## Inference

Not in this repository. A later decoder loads `analysis/HMM/song_soft_hmm_t6.npz` and scores a new recording with the prototypes stored there.

## Deferred

Left for a later pass: the coverage and specificity scores above, using the motif dictionary as the HMM observation, and a decoder for a new recording.
