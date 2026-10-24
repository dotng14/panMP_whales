---
name: Whale song pipeline
overview: "Stage 2 is the Baum–Welch fit in analysis/HMM. Stage 1 (Matching Pursuit combined with panMP) is still a sketch. Decoding a new recording is not in this repository."
todos:
  - id: stage1-motifs
    content: Combine pan-MP and Matching Pursuit into a motif dictionary, and score it by coverage and half-length specificity.
    status: pending
  - id: stage2-hmm
    content: Baum–Welch fit of the six-theme song graph on panMP motif scores, in analysis/HMM.
    status: completed
  - id: stage3-decode
    content: A later decoder loads analysis/HMM/song_soft_hmm_t6.npz. Not in this repository.
    status: pending
isProject: false
---

# Whale song analysis pipeline

Barebones plan. The writeup lives in [analysis/README.md](analysis/README.md) under **Song analysis pipeline**. Stage 2 is the fit in [analysis/HMM/](analysis/HMM/). Stage 1, combining panMP with Matching Pursuit, is still a sketch.

Dictionary learning stays in [analysis/dict_learning/](analysis/dict_learning/). The copies that had been placed on the `analysis/` root are already gone.

Payne and McVay (1971) supply the hierarchy this model is meant to recover: **subunit < unit < phrase < theme < song**. A song of type A is six themes in a fixed cycle. Phrase-repeat count varies; theme order does not, and no theme is dropped. See [papers/payne_mcvay_1971_song_structure.md](papers/payne_mcvay_1971_song_structure.md). The fitted graph follows that order and does not add an edge from theme 6 back to theme 1. Each training song is its own sequence.

In the fitted model a **motif** is one of 12 panMP prototypes, and the observation at each second is the 12-dimensional score against them. The labeled `motif` column (`U1`…`U11`) in [backend/data/song_structure/](backend/data/song_structure/) names the slots on the drawing. It is not an emitted symbol, and slots that share a name do not share a Gaussian.

```mermaid
flowchart LR
  seq[Whitened sequence] --> motifs[Stage1 Motif dictionary]
  motifs --> hmm[Stage2 Train HMM]
  hmm --> saved[Parameter file]
```

## Stage 1 — Motif dictionary

**Input.** A whitened sequence from the GPL path: window spectrograms in `spectrogram_npy/` written by [analysis/GPL_Python/gpl_outputs.py](analysis/GPL_Python/gpl_outputs.py).

**Method.** Two existing pieces, not yet combined:

- Matrix Profile over motif length: [analysis/panMP.py](analysis/panMP.py) (`panMP`) on [analysis/mpxForPan.py](analysis/mpxForPan.py). Finds repeated subsequences without a fixed length.
- Matching Pursuit dictionary: the Mallat loop in [analysis/dict_learning/](analysis/dict_learning/) (seed a patch, average its matches, re-pursue, deflate the residual). Documented in [analysis/README.md](analysis/README.md).

**Combination (sketch only).** Use pan-MP peaks to propose candidate kernels (location and length). Use Matching Pursuit to place, average, and deflate them into a small dictionary. Details of the handoff come later.

**Output.** A dictionary of convolution kernels, each a short time–frequency patch, plus the score of each kernel convolved along the sequence.

**Quality.**

- **Coverage.** Fraction of the sequence where at least one motif score is more than 2 standard deviations above that motif’s mean.
- **Specificity.** For each motif, the ratio of the score at a peak to the scores obtained when that motif is shifted by half its length to either side.

These two measures are not in the repo today. Current dictionary gates (repeat count, residual energy, time-bin coverage near 0.9) stay as internal stopping rules, not as the stage score.

## Stage 2 — Train the HMM

Implemented in [analysis/HMM/](analysis/HMM/). The training recording is the 10 kHz, 10-hour file. GPL at 10–1000 Hz is collapsed to 10 Hz, then scored with 12 panMP prototypes. The motif CSV cuts 53 songs and initializes the Gaussians. It is not the observation.

**States.** The graph in [analysis/HMM/song_soft_hmm_t6.dot](analysis/HMM/song_soft_hmm_t6.dot), fit with Baum–Welch:

- **6 themes**, 28 phrase slots. A song may open on the first slot of any theme. Theme 6 repeats until the song ends and does not return to theme 1.
- **140 states.** Each slot has five duration states, 1–5 seconds, sharing one diagonal Gaussian over the 12 motif scores. Slots that share a unit name do not share a Gaussian.
- Only the last slot of a theme chooses repeat versus next.

**Output.** [analysis/HMM/song_soft_hmm_t6.npz](analysis/HMM/song_soft_hmm_t6.npz) and the report [analysis/HMM/song_soft_hmm_t6_report.pdf](analysis/HMM/song_soft_hmm_t6_report.pdf).

**Quality.** In [analysis/HMM/work/train_report.json](analysis/HMM/work/train_report.json):

- Sequence log-likelihood, in total and in nats per second, against a 12-dimensional standard normal.
- Posterior-weighted complete-data log-likelihood, split into emission, transition, and start.
- Per-song log-likelihood per second.
- Forward–backward probability on the CSV theme, and expected versus labeled seconds in each theme.
- Likelihood drop when one theme-end probability or one slot mean is nudged.

## Stage 3 — Inference

Not in this repository. A later decoder loads `analysis/HMM/song_soft_hmm_t6.npz` and scores a new recording with the prototypes stored there. This repository does not choose a state path.

## What is reused vs new

- Reuse: GPL whitened windows, `panMP` / `mpxForPan`, the Mallat dictionary loop in [analysis/dict_learning/](analysis/dict_learning/), and the motif-segment CSV as the song cuts and the Gaussian initialization.
- Done in [analysis/HMM/](analysis/HMM/): panMP prototype scores and the Baum–Welch fit. Matching Pursuit is not part of that fit.
- Not in this repository: a decoder for a new recording.

## Deferred

Left for a later pass: how pan-MP candidates enter Matching Pursuit, and a decoder that loads the saved parameter file for a new recording.
