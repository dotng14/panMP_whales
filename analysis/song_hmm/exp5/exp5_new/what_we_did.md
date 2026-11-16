# What we did (Exp5 revised front end → Exp6 package)

## Motivation

The first Exp5 dictionary (`train_w336_s0`) stopped early at **5 atoms** / coverage **~0.28** and allowed overlapping placements. That alphabet was too weak for song structure. We rebuilt Approach A’s front end for **highest coverage with strictly no overlap**, mapped motifs on a **20 × 30-minute** timeline, retrained the theme–slot HMM, and packaged results as **Exp6**.

## Changes to tooling

- Added `--disallow-overlap` to `python -m analysis.dict_learning randomized-build` (threaded through `refine_dictionary` / `learn_randomized_outer`).
- Matching pursuit already supported `--disallow-overlap`; Exp5 pipeline passes it for both build and place.
- Motif plots: `plot_motifs_twenty_half_hours` → `work/figures/motifs_by_30min/`.

## Pipeline (commands)

```bash
cd /home/hdn008/projects/panMP_whales

# Dict + MP + sequences (no overlap)
python -m analysis.song_hmm.exp5.run_pipeline 1a --force --workers 8

# Theme–slot HMM + eval
python -m analysis.song_hmm.exp5.run_pipeline 2
python -m analysis.song_hmm.exp5.run_pipeline 3

# Package CSVs / figures docs as Exp6
python -m analysis.song_hmm.exp6.run_pipeline all
```

Notebook: [`../notebooks/approachA_motifs.ipynb`](../notebooks/approachA_motifs.ipynb).

## Artifacts

| Artifact | Path |
|---|---|
| Dictionary | [`../work/dict/train_w336_s0_noov/`](../work/dict/train_w336_s0_noov/) |
| MP (per hour) | [`../work/mp/193500_h01`](../work/mp/) … `h10` |
| Sequences | [`../work/sequences/193500_h*.json`](../work/sequences/) |
| HMM eval A | [`../work/results/A/eval.json`](../work/results/A/) |
| 30-min panels | [`../work/figures/motifs_by_30min/`](../work/figures/motifs_by_30min/) |
| Exp6 package | [`../../exp6/`](../../exp6/) — exports, copied figures, report |

### CSV schemas (`exp6/work/exports/`)

- `placements.txt` — `motif start_sec end_sec` (wav-relative, full 10 h)
- `placements_<stem>.csv` — `motif,start_sec,end_sec,unexplained`
- `<stem>.motif_segments.csv` — catalog columns with song/theme from Approach A test decode
- `segments.csv` / `segments_motifs_only.csv` — typed intervals with unix times

Stem: `SOCAL_H_76_230426_193500_df100.x.x`. Origin unix **1682537700**.

## Status

Theme–slot HMM A/B **retrained** on the no-overlap alphabet (Exp5 stages 2–3). Exp6 docs and CSVs reflect that run. See [`../../reports/exp6_motif_segment_exports.md`](../../reports/exp6_motif_segment_exports.md).
