# Exp9: many-theme HMM (self-loops) on wav23

**Recording:** wav23 (1.25 h). **Arms:** `R_defaults`, `R_defaults_cov04`.

**Split:** train `[0, 3600)` s, test remainder.

## Architecture vs Exp8

```text
slot1 → slot2 → … → slot_Lmax   (L_max=10, shared)
  each slot: self-loop allowed
  slot_Lmax → slot1          (phrase repeat)
  slot_Lmax → next.slot1     (next theme; never revisit)
```

- Theme grid `T ∈ [8, 10, 12, 14, 16]` (BIC on train).
- Themes never revisit; one phrase type per theme; phrase may repeat inside theme.

## Exp9 results (test JS vs random)

| run_id | n_themes | L_max | n_bounds | mean JS pred | mean JS random | pred/random | null trial mean | null p90 | looks_like_random | pred_below_p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `R_defaults` | 16 | 10 | 1 | 0.2414 | 0.1415 | 1.706 | 0.2434 | 0.3771 | False | True |
| `R_defaults_cov04` | 10 | 10 | 1 | 0.5843 | 0.3112 | 1.878 | 0.3136 | 0.4673 | False | False |

## Comparison vs Exp8

| run_id | Exp8 T | Exp8 JS pred | Exp8 pred/rand | Exp9 T | Exp9 JS pred | Exp9 pred/rand |
|---|---:|---:|---:|---:|---:|---:|
| `R_defaults` | 8 | 0.6931 | 4.898 | 16 | 0.2414 | 1.706 |
| `R_defaults_cov04` | 8 | 0.5843 | 1.878 | 10 | 0.5843 | 1.878 |

## Artifacts

- `analysis/song_hmm/exp9/work/arms/<run_id>/`
- Model: `…/models/theme_slot.npz`
- Eval: `…/results/eval.json`
- Notebooks: `analysis/song_hmm/exp9/notebooks/`
