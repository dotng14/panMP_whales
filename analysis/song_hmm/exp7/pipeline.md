# Exp7 pipeline

```bash
python -m analysis.song_hmm.exp7.run_pipeline 0
python -m analysis.song_hmm.exp7.run_pipeline inventory
python -m analysis.song_hmm.exp7.run_pipeline grid      # randomized-build runs
python -m analysis.song_hmm.exp7.run_pipeline motifs    # learn_from_motifs
python -m analysis.song_hmm.exp7.run_pipeline score
python -m analysis.song_hmm.exp7.run_pipeline panels
python -m analysis.song_hmm.exp7.run_pipeline report
python -m analysis.song_hmm.exp7.run_pipeline all
```

| Stage | Action |
|---|---|
| `0` | Check wav23 + GPL; write `work/file_manifest.json` |
| `inventory` | Copy/confirm `inventory.md` into work |
| `grid` | Run randomized-build configs → MP → sequences |
| `motifs` | `learn_from_motifs` on full wav23 → sequences |
| `export` | Motif placement CSVs → `work/exports/<run_id>/motifs.csv` |
| `score` | Phrase-readiness metrics + `leaderboard.json` |
| `panels` | 5-min motif panels per scored run |
| `report` | `work/report.md` + `reports/exp7_dict_evaluation.md` |

Outputs under `work/runs/<run_id>/`.
