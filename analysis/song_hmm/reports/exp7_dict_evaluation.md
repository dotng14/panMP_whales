# Exp7: dictionary evaluation on wav23

**Recording:** `wav23.wav` (200 kHz, 1.25 h), GPL 60 windows.

Phrase-readiness gate before any theme HMM. Methods: `randomized-build` grid + `learn_from_motifs`.

## Leaderboard

| Rank | run_id | method | atoms | cov | period3 | same | multi-phrase | gate | ready |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | `R_defaults_hicov` | randomized-build | 172 | 0.560 | 0.000 | 0.087 | 0.659 | False | no |
| 2 | `R_mod_overlap` | randomized-build | 73 | 0.476 | 0.006 | 0.128 | 0.630 | False | no |
| 3 | `R_wide_w` | randomized-build | 53 | 0.493 | 0.009 | 0.143 | 0.592 | False | no |
| 4 | `R_defaults` | randomized-build | 10 | 0.281 | 0.039 | 0.286 | 0.589 | True | no |
| 5 | `R_strict_eps` | randomized-build | 58 | 0.491 | 0.020 | 0.177 | 0.559 | False | no |
| 6 | `R_defaults_cov04` | randomized-build | 30 | 0.403 | 0.012 | 0.194 | 0.557 | False | no |
| 7 | `R_mod_noov` | randomized-build | 49 | 0.451 | 0.018 | 0.190 | 0.548 | False | no |
| 8 | `R_narrow_w` | randomized-build | 43 | 0.430 | 0.028 | 0.214 | 0.532 | False | no |
| 9 | `R_fewer_atoms` | randomized-build | 31 | 0.417 | 0.017 | 0.181 | 0.479 | False | no |
| 10 | `R_exp6_like` | randomized-build | 112 | 0.580 | 0.002 | 0.151 | 0.470 | False | no |

## Go / no-go

**No-go** — no run is phrase-ready (gate fail and/or coverage < 0.4 and/or too many atoms). Do **not** advance to theme HMM. Next: dict redesign (e.g. panMP→MP batch), not another Exp5-style theme model.

## Arms: good / bad

- `R_defaults_hicov` (randomized-build): atoms=172, cov=0.560, period3=0.000, flags=['n_atoms_gt_80', 'gate_fail']
- `R_mod_overlap` (randomized-build): atoms=73, cov=0.476, period3=0.006, flags=['gate_fail']
- `R_wide_w` (randomized-build): atoms=53, cov=0.493, period3=0.009, flags=['gate_fail']
- `R_defaults` (randomized-build): atoms=10, cov=0.281, period3=0.039, flags=['coverage_lt_0.4']
- `R_strict_eps` (randomized-build): atoms=58, cov=0.491, period3=0.020, flags=['gate_fail']
- `R_defaults_cov04` (randomized-build): atoms=30, cov=0.403, period3=0.012, flags=['gate_fail']
- `R_mod_noov` (randomized-build): atoms=49, cov=0.451, period3=0.018, flags=['gate_fail']
- `R_narrow_w` (randomized-build): atoms=43, cov=0.430, period3=0.028, flags=['gate_fail']
- `R_fewer_atoms` (randomized-build): atoms=31, cov=0.417, period3=0.017, flags=['gate_fail']
- `R_exp6_like` (randomized-build): atoms=112, cov=0.580, period3=0.002, flags=['n_atoms_gt_80', 'gate_fail']

## Artifacts

- Runs: `analysis/song_hmm/exp7/work/runs/`
- Leaderboard: `analysis/song_hmm/exp7/work/leaderboard.json`
- Panels: `…/runs/<id>/figures/motifs_by_5min/`
- Notebook: `analysis/song_hmm/exp7/notebooks/dict_compare.ipynb`
- Inventory: `analysis/song_hmm/exp7/inventory.md`
