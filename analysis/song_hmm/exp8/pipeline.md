# Exp8 pipeline

```bash
python -m analysis.song_hmm.exp8.run_pipeline sync
python -m analysis.song_hmm.exp8.run_pipeline motifs
python -m analysis.song_hmm.exp8.run_pipeline hmm
python -m analysis.song_hmm.exp8.run_pipeline themes
python -m analysis.song_hmm.exp8.run_pipeline report
# or: all
```

| Stage | Action |
|---|---|
| `sync` | Copy Exp7 sequences + `motifs.csv` / `placements.csv` into `work/arms/<id>/` |
| `motifs` | 15×5-min GPL display panels + `notebooks/motifs_*.ipynb` |
| `hmm` | Split at 3600 s; `select_theme_slot` on train; decode + JS on test → `results/eval.json` |
| `themes` | Theme-boundary panels + `notebooks/themes_*.ipynb` |
| `report` | `reports/exp8_theme_hmm_wav23.md` |
