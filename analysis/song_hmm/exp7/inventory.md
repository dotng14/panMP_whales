# Exp7 inventory: dict_learning methods and knobs

## Methods that still work

| Method | CLI / entry | Seed source | Notes |
|---|---|---|---|
| **randomized-build** | `python -m analysis.dict_learning randomized-build` | Random poorly-explained detection patches | Baseline Create→Refine outer loop |
| **matching-pursuit** | `python -m analysis.dict_learning matching-pursuit` | — | Places a saved dictionary |
| **learn_from_motifs** | `python -m analysis.dict_learning.code.learn.learn_from_motifs` | Matrix-profile pairs (5 s / 10 s) → MP refine | Slow; one pair per round |

Removed (2026-10-25): `build_dictionary`, `extend_dictionary`, greedy seed trials, DP/subspace solvers, etc. See `analysis/dict_learning/README.md` §Removed.

## randomized-build — important knobs

| Knob | Default | Exp6-on-193500 | Meaning |
|---|---|---|---|
| `--min-coverage` | 0.2 | 0.95 | Stop when time coverage ≥ this |
| `--max-outer` | 10 | 40 | Cap outer Create→Refine passes |
| `--m-per-outer` | 5 | 5 | Seeds added per outer |
| `--n-min` | 10 | 10 | Drop atoms with fewer patches |
| `--epsilon` | 0.9 | 0.9 | Keep seed patch if residual/orig > ε |
| `--disallow-overlap` | off | **on** | Strict non-overlap in refine MP |
| `--min/max-template-width` | 10–20 | 10–20 | Atom width in bins |
| `--n-components` | 1 | 1 | PCA subspace rank after refine |
| `--refine-iters` | 1 | 1 | Refine rounds per outer |

Follow-up: `R_defaults_hicov` keeps defaults (overlap on, ε=0.9, n_min=10) but sets `min_coverage=0.85`, `max_outer=40` to push coverage without Exp6-like no-overlap.

## learn_from_motifs — important knobs

| Knob | Exp7 retune | Library default | Meaning |
|---|---|---|---|
| `--lengths` | `2,3,5` | `5,10` | Motif lengths (seconds) for MP search |
| `--min-coverage` | `0.85` | `0.5` | Stop when frame coverage ≥ this |
| `--max-rounds` | 30 (cap) | 100 | Cap atoms / rounds |
| `--max-relative-distance` | `0.5` | `0.859` (014845) | Placement acceptance; lower = stricter |
| `--refine-iters` | 1 | 1 | Average/PCA refine iters |

`L_motifs_wav23_v1_loose` used library defaults (`5,10` / `0.5` / `0.859`) and stopped after **1** atom with dissimilar patches all labeled M0.

## Historical contrast (not a numeric baseline)

Exp6 on **193500** (2 kHz GPL): no-overlap randomized-build → **199** atoms, cov **0.878**, phrase gate **fail**, no clear `ABCABC`. Different wav and GPL rate than wav23 — do not compare atom counts directly.

## What “good” means for phrases

motifs → **repeatable multi-motif phrases** → themes.

Coverage alone is **not** enough (Exp6 lesson). Implementation: [`metrics_dict.py`](metrics_dict.py) + [`../exp5/gate.py`](../exp5/gate.py).

A run is **`phrase_ready`** only if **none** of the skepticism flags fire.

| Flag key | Criterion | Meaning |
|---|---|---|
| `n_atoms_gt_80` | `n_atoms > 80` | Dictionary too fragmented / unlikely to be phrase-like |
| `gate_fail` | `gate_pass_hint` is not true | Motif-sequence gate failed (see below) |
| `coverage_lt_0.4` | `time_coverage_fraction < 0.4` | Too little of the file covered by placements |
| `degenerate_single_motif_alphabet` | `n_unique_motifs ≤ 1` | Single-id alphabet → fake-perfect same-label stats; demoted |

**Phrase gate (`gate_pass_hint`):** from `motif_gate` —

- Cut phrases (gap cuts); compute normalised edit distance `d1` between consecutive phrases (gap < 30 s).
- At novelty / JS peaks (`b1_boundaries`), compute `d1` between the phrase just before vs just after the peak.
- **Pass** iff `median_d1_consecutive < median_d1_at_js_peaks` (within-theme phrases more similar than across novelty peaks). Missing either median → hint is `None` → treated as fail.

**Supporting metrics (reported, not hard-fail):** `frac_phrases_len_ge2`, `frac_consecutive_same_label`, `n_phrase_types_period_edit05`, `frac_atoms_rare_le5`, `median_phrase_len`. `frac_local_period3_repeat` may still be logged for curiosity but is **not** a gate — phrase length / period is unknown a priori, so a fixed period-3 check does not apply.

**Leaderboard order:** `phrase_ready` first; demote degenerate alphabets; then multi-motif phrase frac, consecutive-same, coverage.
