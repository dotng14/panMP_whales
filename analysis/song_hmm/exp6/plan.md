# Experiment 6: no-overlap motifs, 30-minute map, theme–slot HMM (Approach A)

**This replaces** the earlier Exp6 “CSV-only” package. Exp6 is now the documented revision of Approach A on `193500`:

1. Strict **no-overlap** dictionary (train 0–7 h) + matching pursuit (full 10 h)
2. Motif timeline as **20 × 30-minute** spectrogram panels
3. Theme–slot **categorical HMM** retrained on that alphabet; test decode + JS / provisional scores
4. Viewer CSVs (`placements`, `motif_segments`, `segments`) packaged under `exp6/work/`

Compute still lives in `../exp5/work/` (dict / MP / sequences / models / eval). This folder is the **results package + docs** for that stack. Do not confuse with the first Exp5 dual-approach report (`../reports/exp5_theme_boundaries.md`, 5-atom overlapping dict).

## Data

| | |
|---|---|
| Wav | `SOCAL_H_76_230426_193500` @ 2 kHz, 10 h |
| Train / test | windows `[0,336)` / `[336,480)` (7 h / 3 h) |
| Stem | `SOCAL_H_76_230426_193500_df100.x.x` |
| Origin unix | **1682537700** |

## Approach

**Approach A only** for motif CSVs and the primary HMM (categorical emissions on motif ids). Approach B (Gaussian on STFT features) is still produced by the Exp5 runner; features are unchanged by the dictionary, so B is secondary here.

Topology: theme–slot chains with **repeat / next**, no skip inside a theme (`../exp5/figures/architecture/`).

## Deliverables

| Artifact | Location |
|---|---|
| Dictionary summary | `work/dict_summary.json` (copy of `train_w336_s0_noov`) |
| HMM eval (A) | `work/eval_A.json`, `work/hmm_summary.json` |
| 20 × 30 min panels | `work/figures/motifs_by_30min/panel_01.png` … `panel_20.png` |
| Theme decode slices | `work/figures/A/` |
| Placements / structure CSVs | `work/exports/` |
| Notebook (panels) | `../exp5/notebooks/approachA_motifs.ipynb` |
| Report | [`../reports/exp6_motif_segment_exports.md`](../reports/exp6_motif_segment_exports.md) |
