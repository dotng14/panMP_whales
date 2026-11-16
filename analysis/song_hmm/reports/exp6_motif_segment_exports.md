# Exp6: no-overlap motifs, 30-minute map, theme–slot HMM

**Recording:** `SOCAL_H_76_230426_193500` (2 kHz, 10 h).  
**Package:** `analysis/song_hmm/exp6/` (docs + `work/`). Compute: `exp5/work/`.

This report **replaces** the earlier Exp6 CSV-only note. It documents the Approach A revision: no-overlap dictionary, 20 × 30-minute motif panels, HMM retrain/decode, and catalog CSVs.

## Summary

| Item | Value |
|---|---|
| Dictionary | `train_w336_s0_noov`, **199** atoms, coverage **0.878**, `disallow_overlap=True` |
| Cleaned placements (10 h) | **9666** |
| Motif map | **20 × 30 min** panels (`panel_01` … `panel_20`) |
| HMM A | **T=8**, slots `[2,2,2,2,2,2,3,2]`, vocab used in fit **164** |
| Test boundaries (A) | **20** |
| Theme visits decoded (A) | **28** spans (`T1`…`T8`); **21** with positive duration used in structure export |
| Songs (theme-1 rule) | **3** |
| Motif rows in structure CSV | **2411** (placements inside test theme spans) |
| Origin unix | **1682537700** |

## Dictionary and placements

| Knob | Value |
|---|---|
| Train windows | 336 (0–7 h) |
| `--disallow-overlap` | build + MP |
| `--min-coverage` / `--max-outer` | 0.95 / 40 → stopped at `max_outer`, cov 0.878 |
| MP | hours 1–10, no overlap |
| Sequences | `exp5/work/sequences/193500_h01.json` … `h10.json` |

Phrase-repeat gate on train still fails (median consecutive `d1` ≈ 1); treat theme decode as exploratory.

## Theme–slot HMM (Approach A)

Retrained on no-overlap motif sequences (Exp5 stages 2–3).

| Metric | Value |
|---|---|
| Selected model | 8 themes, slots 2/2/2/2/2/2/3/2 |
| Train score (LL) | −36751.17 |
| Mean JS (pred / random) | **0.546** / 0.414 |
| Median JS (pred / random) | 0.476 / 0.413 |
| vs provisional marks | recall@10s 0.077, precision@10s 0.15, F1@10s 0.102 |
| FA / h | 5.67 (n_pred 20, n_marks 39) |

Approach B was also refit in the same Exp5 run (features unchanged by the dict); primary Exp6 deliverable is Approach A.

## Figures

- Half-hour motif spectrograms: [`../exp6/work/figures/motifs_by_30min/`](../exp6/work/figures/motifs_by_30min/) (also under `exp5/work/figures/motifs_by_30min/`)
- Theme decode slices: [`../exp6/work/figures/A/`](../exp6/work/figures/A/)
- Notebook: [`../exp5/notebooks/approachA_motifs.ipynb`](../exp5/notebooks/approachA_motifs.ipynb)

## CSV exports

Regenerate:

```bash
python -m analysis.song_hmm.exp6.run_pipeline all
```

| File | Rows / role |
|---|---|
| `placements.txt` / `placements_*.csv` | 9666 motifs, full 10 h |
| `segments_motifs_only.csv` | same, unix times |
| `*.motif_segments.csv` | 2411 rows with song/theme |
| `segments.csv` | motifs in structure + theme + song intervals |

## Paths

| | |
|---|---|
| Plan / pipeline | `../exp6/plan.md`, `../exp6/pipeline.md` |
| Dict | `../exp5/work/dict/train_w336_s0_noov/` |
| Eval A | `../exp5/work/results/A/eval.json` |
| Packaged copies | `../exp6/work/dict_summary.json`, `eval_A.json`, `hmm_summary.json` |
