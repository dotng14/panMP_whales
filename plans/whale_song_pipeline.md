---
name: Whale song pipeline
overview: "A three-stage pipeline, written in analysis/README.md: learn a motif dictionary with Matching Pursuit and Matrix Profile, fit a hierarchical HMM of themes, phrases, and units with Baum–Welch, then decode a sequence with Viterbi. Each stage has an explicit quality measure. Algorithmic detail is deferred."
todos:
  - id: stage1-motifs
    content: Combine pan-MP and Matching Pursuit into a motif dictionary, and score it by coverage and half-length specificity.
    status: pending
  - id: stage2-hmm
    content: Define and train a Baum–Welch HMM of 6 cyclic themes, repeating phrases, and motif-emitting units; score posterior-weighted likelihood and single-parameter sensitivity.
    status: pending
  - id: stage3-viterbi
    content: Decode a sequence with Viterbi and score sequence likelihood plus sensitivity to theme-boundary shifts.
    status: pending
isProject: false
---

# Whale song analysis pipeline

Barebones plan. The writeup lives in [analysis/README.md](analysis/README.md) under **Song analysis pipeline**. Each stage has an input, a method, an output, and a quality measure. How the methods are combined, and the exact HMM topology, stay open until a later pass.

Dictionary learning stays in [analysis/dict_learning/](analysis/dict_learning/). The copies that had been placed on the `analysis/` root are already gone.

Payne and McVay (1971) supply the hierarchy this model is meant to recover: **subunit < unit < phrase < theme < song**. A song of type A is six themes in a fixed cycle. Phrase-repeat count varies; theme order does not, and no theme is dropped. See [papers/payne_mcvay_1971_song_structure.md](papers/payne_mcvay_1971_song_structure.md).

In this pipeline a **motif** is a convolution kernel (acoustic atom). A **unit** is a latent state that emits a distribution over those motifs. That is finer than the labeled `motif` column (`U1`…`U11`) in [backend/data/song_structure/](backend/data/song_structure/), which is a reference annotation, not the model.

```mermaid
flowchart LR
  seq[Whitened sequence] --> motifs[Stage1 Motif dictionary]
  motifs --> hmm[Stage2 Train HMM]
  hmm --> decode[Stage3 Viterbi]
  motifs --> decode
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

**Input.** The motif dictionary and its score sequence from stage 1.

**States.** One hierarchical HMM, trained with Baum–Welch:

- **6 themes** in a cycle (1→2→3→4→5→6→1). Each theme is present; only the number of repeats changes.
- **Phrases** repeat inside a theme. A phrase is a short chain of units.
- **Units** are the emitting states. Each unit has a distribution over motifs.

No HMM code exists. [backend/app/services/song_structure.py](backend/app/services/song_structure.py) only loads a hand-labeled CSV (theme, phrase id, phrase count, motif label) for display.

**Output.** Fitted transition and emission parameters, and the posterior over states on the training sequence.

**Quality.**

- **Posterior-weighted likelihood** of the sequence.
- **Parameter sensitivity:** how much that likelihood moves when any single parameter is perturbed.

## Stage 3 — Inference

**Input.** A trained HMM and a new sequence scored with the same motif dictionary.

**Method.** Viterbi: the most likely state path (theme, phrase, unit).

**Output.** That path, aligned to the sequence.

**Quality.**

- **Likelihood** of the sequence.
- **Boundary sensitivity:** how much that likelihood moves when the boundaries between themes are shifted.

## What is reused vs new

- Reuse: GPL whitened windows, `panMP` / `mpxForPan`, the Mallat dictionary loop in [analysis/dict_learning/](analysis/dict_learning/), Payne’s six-theme cycle, and the motif-segment CSV as a later reference.
- New: the MP–Matrix-Profile combination, the three quality measures, and the HMM (Baum–Welch and Viterbi).

## Deferred

Left for a later pass: how pan-MP candidates enter Matching Pursuit; whether observations are continuous kernel scores or discrete motif hits; the exact phrase/unit topology inside a theme; where the new code lives under `analysis/`; and any comparison against the labeled CSV.
