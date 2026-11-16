# Experiment 4: theme boundaries from motif sequences — plan

**Scope for this run: `014845` only.**

Companion documents:
- [pipeline.md](pipeline.md)
- [metrics.md](metrics.md)

## 1. Context

Experiments 1–3 fed hand-made spectrogram features into an HMM, so the model had to learn both *what* the sounds are and *how* they are ordered. The CSV reference was not usable. Experiment 4 separates the two problems:

1. Detect motifs first (GPL → dictionary → matching pursuit).
2. Train an HMM on the motif / phrase sequence.

## 2. Goal

On **`SOCAL_H_76_230427_014845`**, find **theme boundaries**: times where one repeated phrase stops and another starts.

1. Split the file into a **train half** and an **eval half**.
2. Build the dictionary and train the HMM on the **train half only**.
3. Decode the **eval half**.
4. Show the eval spectrogram with **themes** and **boundaries** in notebooks.
5. Write a **report** that says what works, what fails, and where on the recording.

Two options:

| | **Option 1: motif-vocabulary HMM** | **Option 2: repetition HMM** |
|---|---|---|
| Observes | Motif ids + gap tokens | Phrase distances (`d1`, `d2`, …) — no motif ids |
| Output | Theme labels + boundaries | Boundaries / segments only |

## 3. Data

| Item | Value |
|---|---|
| Wav | `/home/yfreund/whale_processing/LeiData/SOCAL_H_76_230427_014845.x.wav` (also `_10kHz.wav`) |
| Working rate | 2 kHz (decimated), band 0–1 kHz |
| GPL | `exp4/work/gpl_014845_2kHz/` — 30 windows × 75 s (127×779), matching the ensemble 193500 GPL shape |
| Length | ≈ 37.5 min (0.62 h) |
| **Train half** | windows 0–14 (≈ 0–18.75 min) |
| **Eval half** | windows 15–29 (≈ 18.75–37.5 min) |

Results: `exp4/work/results/014845/`. Figures: `exp4/work/figures/014845/`.

*(193500 is out of scope for this run; it may return later for multi-file tests.)*

## 4. Reference

Prefer hand marks on the eval spectrogram. Until then, provisional marks from B1 novelty on each half are used to wire training/scoring — **scores against provisional marks are not final truth**.

## 5. Processing (014845 only)

1. Decimate to 2 kHz; run GPL (10–1000 Hz) → 30 windows.
2. **Dictionary:** `randomized-build` on the **train-half windows only**.
3. **Matching pursuit** with that dictionary on **all** windows.
4. Placements → events → gap tokens → sessions; cut phrases at `GAP_S` / `GAP_L`.
5. Train option 1 / option 2 / B1 / B2 on the train half; decode eval half.

## 6–8. Models and baselines

As before: option 1 (`pair-v0`, `1-full`), option 2 (repetition HMM), baselines B1 (token-histogram novelty) and B2 (`d1` threshold). All trained/tuned on the train half and scored on the eval half only.

## 9. Evaluation, notebooks, report

**Scores** on eval half: recall / precision / F1 @ ±5/10/30 s, error, false alarms / h ([metrics.md](metrics.md)).

**Notebooks** (`option1_themes.ipynb`, `option2_themes.ipynb`) for 014845 only:
- eval spectrogram, coloured themes, boundary lines, theme table, scores.

**Report** `reports/exp4_motif_boundary.md` must answer:
- Overall: does each model do well or badly?
- Where it does well (time ranges / behaviours).
- Where it fails (misses, false alarms, collapsed themes, over-segmentation).
- What to try next.

## 10. Stages

| Stage | Work |
|---|---|
| 0 | 2 kHz wav + GPL for 014845 |
| 1 | Dictionary on train half; MP on all windows |
| 2 | Motif sequences + marks |
| 3 | Train / decode / figures for eval half |
| 4 | Notebooks + report |

## 11. Deliverables

- `plan/`, `exp4/` code and notebooks
- `exp4/work/results/014845/`, `exp4/work/figures/014845/`
- `reports/exp4_motif_boundary.md`

## 12. Open questions

1. Hand-mark the eval half.
2. Dictionary size / coverage on this short file (risk of under-fitting option 1).
3. Transitional phrases: §4 rule vs option 2 `X` state.
