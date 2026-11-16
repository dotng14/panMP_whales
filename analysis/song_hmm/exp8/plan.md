# Exp8 plan

Theme–slot HMM on Exp7 wav23 dictionaries `R_defaults` and `R_defaults_cov04`.

- Sync motif CSVs + sequences from Exp7
- 5-min motif notebooks (GPL display)
- Train Approach A on first 1 h; decode + JS on remainder
- Theme-boundary notebooks
- Report with architecture + JS vs random

Exp9 (many themes + slot self-loops) is **after** Exp8 — not part of this run.

## Exp7 phrase-readiness flags (provenance)

Canonical write-up: [`../exp7/inventory.md`](../exp7/inventory.md) § “What good means for phrases”. Source: [`../exp7/metrics_dict.py`](../exp7/metrics_dict.py) + [`../exp5/gate.py`](../exp5/gate.py).

Arms used here:

- `R_defaults` — 10 atoms, cov~0.28, gate pass; flags: `coverage_lt_0.4`
- `R_defaults_cov04` — 30 atoms, cov~0.40, gate fail; flags: `gate_fail`

A run is **`phrase_ready`** only if **none** of these fire:

| Flag key | Criterion | Meaning |
|---|---|---|
| `n_atoms_gt_80` | `n_atoms > 80` | Dictionary too fragmented / unlikely to be phrase-like |
| `gate_fail` | `gate_pass_hint` is not true | Motif-sequence gate failed |
| `coverage_lt_0.4` | `time_coverage_fraction < 0.4` | Too little coverage |
| `degenerate_single_motif_alphabet` | `n_unique_motifs ≤ 1` | Fake-perfect same-label; demoted |

**Gate pass:** `median_d1_consecutive < median_d1_at_js_peaks` (consecutive phrases more similar than across novelty peaks). Missing either median → fail.

`period3` / `frac_local_period3_repeat` is **not** a readiness flag (phrase motif count unknown a priori).

Exp8 keeps one **gate-pass / low-cov** arm and one **cov-ok / gate-fail** arm even though neither was `phrase_ready` in Exp7.
