# Exp5: theme boundaries on 193500 (train 0–7 h / test 7–10 h)

**Status:** first full pipeline run complete. Boundary scores vs marks are **provisional** (B1 novelty peaks), not hand marks.

Docs: [`../exp5/plan.md`](../exp5/plan.md) · [`../exp5/model_inputs.md`](../exp5/model_inputs.md) · [`../exp5/pipeline.md`](../exp5/pipeline.md)

## Setup

| Item | Value |
|---|---|
| Wav / GPL | `SOCAL_H_76_230426_193500` @ 2 kHz; 480 × 75 s windows, shape 127×779 |
| Train | windows `[0, 336)` → 0–25 200 s |
| Test | windows `[336, 480)` → 25 200–36 000 s |
| Topology | Theme–slot HMM: left-to-right slots, **repeat** / **next**, **no skip** |
| Dict (A) | `randomized-build` on 336 train windows → **5 atoms** (stopped at coverage 0.278) |
| Features (B) | GPL log-band + centroid/bw, hop 1 s, 19-D, train mean/std frozen |

## Motif gate (train)

| Metric | Value |
|---|---|
| Phrases | 2931 |
| Median `d1` consecutive | 0.75 |
| Median `d1` at JS peaks | 1.00 |
| Gate hint | pass (`d1` lower inside runs than at peaks) |

Alphabet is still small (5 motifs + gaps); phrases do not match tightly (`d1` 0.75). Treat HMM boundaries as exploratory.

## Models selected on train

| | Approach A (categorical) | Approach B (Gaussian) |
|---|---|---|
| `T` | 8 (BIC grid 3–8) | 6 (BIC grid 3–6) |
| Slot counts | 8,2,2,2,10,2,2,8 | 3,3,6,3,5,8 (soft-HMM-like default lengths) |

## Test results (provisional marks)

| | A | B |
|---|---|---|
| Predicted boundaries | 18 | 5 |
| Theme segments | 21 | 6 |
| Mean JS @ pred / @ random | **0.217** / 0.111 | **0.103** / 0.042 |
| F1 @ 10 s (vs provisional) | 0.032 | 0.0 |
| False alarms / h | 4.67 | 1.33 |

JS at model boundaries beats random for both approaches (good qualitative signal). Agreement with provisional novelty marks is weak — expected until hand marks exist and/or the motif alphabet improves.

## Where to look

| Artifact | Path |
|---|---|
| Summary JSON | `exp5/work/results/summary.json` |
| A / B eval | `exp5/work/results/{A,B}/eval.json` |
| Spectrograms | `exp5/work/figures/{A,B}/slice_*.png` |
| Notebooks | `exp5/notebooks/approachA_motifs.ipynb`, `approachB_signal.ipynb` |
| Architecture | `exp5/figures/architecture/approachA_theme_slot_hmm.png` |

## Good / bad / where

**Good**
- End-to-end Exp5 pipeline runs on the planned 7 h / 3 h split.
- Theme–slot topology (no skip; repeat/next) is wired for A and B.
- JS at predicted boundaries > JS at random times for both A and B.
- Gate hint passes on train (phrases slightly more similar within runs than at novelty peaks).

**Bad**
- Dictionary stopped early with only **5 atoms** — weak alphabet for song structure.
- Median phrase `d1` still high (0.75); themes will not look like soft-HMM units.
- Provisional F1 vs novelty marks is near zero; do **not** treat as final accuracy.
- Approach B with default soft-HMM slot lengths may be under-segmenting (5 boundaries / 3 h).

**Where next**
1. Hand-mark test 3 h (or a dense subset) for real F1.
2. Revisit dictionary (more atoms / coverage / call-rich subsample) before trusting A.
3. Search slot lengths for B from data (not only soft-HMM defaults).
4. Inspect `work/figures/{A,B}/` by eye for both themes around each boundary.

## How to re-run

```bash
cd /home/hdn008/projects/panMP_whales
python -m analysis.song_hmm.exp5.run_pipeline all --workers 8
# or: 0 | 1a | 1b | 2 | 3
```
