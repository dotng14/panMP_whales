# Exp8: theme–slot HMM on Exp7 wav23 dictionaries

**Recording:** `wav23.wav` (200 kHz, 1.25 h), GPL 60 windows.

**Arms:** `R_defaults` and `R_defaults_cov04` (Exp7 randomized-build dictionaries).

**Split:** train `[0, 3600)` s (1 h), test `[3600, end)` (~15 min).

## HMM architecture (Approach A theme–slot)

Categorical HMM (`hmmlearn.CategoricalHMM`) over motif/gap tokens.

- Each theme = **one phrase type** = left-to-right slot chain `slot1 → … → L` (no skips, no self-loops).
- At last slot: **phrase-repeat** → `slot1` (same theme), or **next-theme** → next theme’s `slot1` (never revisit).
- Final theme: phrase-repeat only.
- `T ∈ {3..8}` chosen by **BIC** on train; per-theme `L_t` from phrase medoids (clipped [2,10]).

```text
Theme1 ──next──► Theme2 ──next──► … ──next──► ThemeT
  ↑ phrase_repeat    ↑ phrase_repeat              ↑ phrase_repeat
```

## Dict provenance (Exp7)

| run_id | atoms | cov | period3 | gate | phrase_ready |
|---|---:|---:|---:|---|---|
| `R_defaults` | 10 | 0.281 | 0.03936 | True | False |
| `R_defaults_cov04` | 30 | 0.4025 | 0.01246 | False | False |

## Results (test JS vs random)

| run_id | n_themes | slot_counts | n_bounds | mean JS pred | mean JS random | pred/random | null trial mean | null p90 | looks_like_random | pred_below_p90 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `R_defaults` | 8 | `[2, 2, 6, 4, 2, 10, 7, 2]` | 1 | 0.6931 | 0.1415 | 4.898 | 0.2434 | 0.3771 | False | False |
| `R_defaults_cov04` | 8 | `[2, 2, 2, 2, 3, 2, 6, 6]` | 1 | 0.5843 | 0.3112 | 1.878 | 0.3136 | 0.4673 | False | False |

## Comparison

- `R_defaults`: mean JS pred=0.6931, pred/random=4.898, T=8.
- `R_defaults_cov04`: mean JS pred=0.5843, pred/random=1.878, T=8.
- Higher mean JS at predicted boundaries: **`R_defaults`** (larger = sharper motif-histogram change at boundary).

## Artifacts

- Arms: `analysis/song_hmm/exp8/work/arms/<run_id>/`
- Motif CSVs: `…/motifs.csv`
- Model: `…/models/theme_slot.npz`
- Eval: `…/results/eval.json`
- Notebooks: `analysis/song_hmm/exp8/notebooks/`

## Note

Train and test are time-adjacent segments of the same 1.25 h file (no separate recording).
Exp9 (many themes + slot self-loops) is planned after Exp8 — not run here.
