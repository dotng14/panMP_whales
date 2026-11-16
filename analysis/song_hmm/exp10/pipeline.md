# Exp10 pipeline

```bash
python -m analysis.song_hmm.exp10.run_pipeline 0
python -m analysis.song_hmm.exp10.run_pipeline dict
python -m analysis.song_hmm.exp10.run_pipeline mp
python -m analysis.song_hmm.exp10.run_pipeline motifs
python -m analysis.song_hmm.exp10.run_pipeline hmm
python -m analysis.song_hmm.exp10.run_pipeline themes
python -m analysis.song_hmm.exp10.run_pipeline report
```

| Stage | Action |
|---|---|
| `0` | Check wav + GPL |
| `dict` | R_defaults randomized-build on train 336 windows |
| `mp` | Matching pursuit hours 1–10 + sequences + motif CSVs |
| `motifs` | 5-min panels (train sample + all test) + notebook |
| `hmm` | Theme–slot train 7 h / decode test 3 h + JS |
| `themes` | Theme-boundary panels on test + notebook |
| `report` | `reports/exp10_r_defaults_193500.md` |
