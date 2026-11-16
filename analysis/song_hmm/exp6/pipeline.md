# Experiment 6 pipeline

Plan: [plan.md](plan.md). Exp5 work tree is the compute source; Exp6 writes under `exp6/work/`.

## Upstream (Exp5)

```bash
# Strict no-overlap dict + MP + sequences (default)
python -m analysis.song_hmm.exp5.run_pipeline 1a
# Theme–slot HMM A (+ B)
python -m analysis.song_hmm.exp5.run_pipeline 2
python -m analysis.song_hmm.exp5.run_pipeline 3
# 20 × 30-min panels
python -c "from analysis.song_hmm.exp5.plot_motifs import plot_motifs_twenty_half_hours as p; p()"
# or execute notebooks/approachA_motifs.ipynb
```

Artifacts used:

| Input | Path |
|---|---|
| Dictionary | `exp5/work/dict/train_w336_s0_noov/` |
| Sequences | `exp5/work/sequences/193500_h*.json` |
| Eval A | `exp5/work/results/A/eval.json` |
| Summary | `exp5/work/results/summary.json` |
| 30-min figures | `exp5/work/figures/motifs_by_30min/` |

## Exp6 stages

```bash
python -m analysis.song_hmm.exp6.run_pipeline 0
python -m analysis.song_hmm.exp6.run_pipeline export-placements
python -m analysis.song_hmm.exp6.run_pipeline export-structure
python -m analysis.song_hmm.exp6.run_pipeline all
```

| Stage | Action |
|---|---|
| `0` | Check noov dict / sequences / `eval.json`; write `work/file_manifest.json` |
| `export-placements` | Motif tokens → `placements.txt`, `placements_<stem>.csv`, `segments_motifs_only.csv` |
| `export-structure` | Approach A themes + placements → `<stem>.motif_segments.csv`, `segments.csv` |

## Outputs (`exp6/work/`)

| File | Role |
|---|---|
| `exports/placements.txt` | `motif start_sec end_sec` (full 10 h) |
| `exports/placements_*.csv` | same + unexplained column |
| `exports/segments_motifs_only.csv` | motif rows with unix times |
| `exports/*.motif_segments.csv` | catalog song-structure columns (test themes) |
| `exports/segments.csv` | motif + theme + song rows (test) |
| `figures/motifs_by_30min/` | 20 half-hour spectrogram panels |
| `figures/A/` | theme-decode slice plots |
| `dict_summary.json` / `eval_A.json` / `hmm_summary.json` | copied metrics |

`start_unix` = **1682537700** + wav-relative seconds.

**Note:** Approach A decode is on the **test** 3 h. Structure CSVs only include placements inside those theme spans. Full-file placements are always in `placements.*`.
