# Exp10: R_defaults pipeline on 193500

**Recording:** `SOCAL_H_76_230426_193500` (~10 h, 2 kHz GPL df100).

**Arm:** R_defaults only (`min_coverage=0.2`, `max_outer=10`, overlap allowed, `n_min=10`, `epsilon=0.9`).

**Split:** train 0–7 h, test 7–10 h.

## HMM architecture

Exp8 Approach A theme–slot categorical HMM (`T∈{3..8}`, no self-loops).
Phrase-repeat at last slot → slot1; next-theme → next theme slot1 (never revisit).

## Dictionary

- atoms=5, coverage=0.2783, stop=coverage

## Results (test JS vs random)

| metric | value |
|---|---|
| n_themes (selected) | 8 |
| slot_counts | `[8, 2, 2, 2, 10, 2, 2, 8]` |
| n_boundaries | 18 |
| n_theme_visits | 21 |
| mean JS pred | 0.2166 |
| mean JS random | 0.1941 |
| pred/random | 1.116 |
| null trial mean | 0.1586 |
| null p90 | 0.1983 |
| looks_like_random | False |
| pred_below_random_p90 | False |

## Artifacts

- Dict: `analysis/song_hmm/exp10/work/dict/R_defaults/`
- Sequences: `…/sequences/193500_h*.json`
- Motifs CSV: `…/work/motifs.csv`
- Model: `…/models/theme_slot.npz`
- Eval: `…/results/eval.json`
- Notebooks: `analysis/song_hmm/exp10/notebooks/`

## Note vs Exp8

Same R_defaults knobs and theme–slot model family as Exp8 on wav23, but different recording (193500, 10 h, 7/3 split).
