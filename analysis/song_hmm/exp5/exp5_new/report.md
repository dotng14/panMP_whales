# Report: Exp5 no-overlap dictionary and 10 h motif map

**Recording:** `SOCAL_H_76_230426_193500` (2 kHz, 10 h).  
**Front end:** GPL → randomized-build (train 0–7 h) → matching pursuit (all 10 h) → cleaned motif sequences.

## Summary

| Item | Previous Exp5 dict | This revision (`train_w336_s0_noov`) |
|---|---|---|
| Overlap | Allowed | **Disallowed** (`disallow_overlap=True`) |
| Atoms | 5 | **199** |
| Time coverage (train build) | 0.278 | **0.878** |
| Stop reason | `coverage` (at 0.25 target) | `max_outer` (40 passes; target was 0.95) |
| Cleaned placements (10 h) | — | **9666** |
| Hour-1 MP overlap check | — | **0** strict overlaps |

Coverage did not reach the 0.95 target: with no overlap and this detection mask, ~0.88 is the plateau after 40 outer passes.

## Dictionary build

| Knob | Value |
|---|---|
| Windows | 336 (hours 0–7) |
| `--disallow-overlap` | yes |
| `--min-coverage` | 0.95 |
| `--max-outer` | 40 |
| `--m-per-outer` | 5 |
| Result | 199 atoms, 49 760 build matchings, coverage 0.878 |

## Matching pursuit (no overlap)

Placed the 199-atom dictionary on hours 1–10 with `--disallow-overlap`. After energy/overlap cleanup → sessions:

| Hour file | Cleaned motif events | Sessions |
|---|---:|---:|
| h01 | 1084 | 2 |
| h02 | 941 | 1 |
| h03 | 887 | 1 |
| h04 | 957 | 1 |
| h05 | 1181 | 1 |
| h06 | 1287 | 1 |
| h07 | 916 | 2 |
| h08 | 845 | 1 |
| h09 | 695 | 6 |
| h10 | 873 | 1 |
| **Total (seq tokens)** | **9666** placements exported | |

Sanity: hour-1 raw MP events had **0** pairwise time overlaps under the no-overlap rule.

## Motif-sequence gate (train)

| Metric | Value |
|---|---|
| Phrases | 2874 |
| Median `d1` consecutive | 1.0 |
| Median `d1` at JS peaks | 1.0 |
| Gate hint | **fail** (no separation) |

Larger alphabet + no overlap improved coverage, but consecutive phrases still do not repeat cleanly by edit distance. Theme HMM should be treated as exploratory until the gate improves or phrases are inspected by eye.

## Exports and figures

- CSVs (packaged Exp6): `../../exp6/work/exports/` (9666 placements; origin unix 1682537700).
- Spectrograms: **20 × 30 min** — `../work/figures/motifs_by_30min/panel_01.png` … `panel_20.png` (also copied under `exp6/work/figures/`).
- Notebook: `../notebooks/approachA_motifs.ipynb`.
- Full write-up: [`../../reports/exp6_motif_segment_exports.md`](../../reports/exp6_motif_segment_exports.md).

## Theme–slot HMM (retrained)

Stages 2–3 on the no-overlap alphabet: Approach A selected **T=8**, slots `[2,2,2,2,2,2,3,2]`; mean JS pred **0.546** vs random 0.414; 20 test boundaries. Details in the Exp6 report.

## Good / bad / next

**Good**
- Strict non-overlap enforced in build and place.
- Large jump in atoms (5 → 199) and train coverage (0.28 → 0.88).
- Half-hour motif map + HMM retrain + Exp6 CSVs shipped.

**Bad**
- Coverage target 0.95 not reached (max_outer / residual structure).
- Phrase-repeat gate still fails (`d1` ≈ 1).
- Boundary scores vs provisional marks remain weak.

**Next**
1. By-eye review of `motifs_by_30min` panels.
2. Optionally raise `max_outer` or tune MP τ / widths if more coverage is needed without allowing overlap.
