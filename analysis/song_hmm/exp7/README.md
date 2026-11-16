# Experiment 7 — dictionary evaluation on wav23

Compare **randomized-build** vs **learn_from_motifs** for phrase-readiness before any theme HMM.

| Doc | |
|---|---|
| [plan.md](plan.md) | Scope |
| [pipeline.md](pipeline.md) | Stages |
| [inventory.md](inventory.md) | Methods + knobs |
| Report | [`../reports/exp7_dict_evaluation.md`](../reports/exp7_dict_evaluation.md) |

```bash
cd /home/hdn008/projects/panMP_whales
python -m analysis.song_hmm.exp7.run_pipeline all
# stages: 0 | inventory | grid | motifs | score | panels | export | report | all
# Motif CSVs: work/exports/<run_id>/motifs.csv  (motif,start_sec,end_sec,energy)
```

Kernel: **Python (panMP / ensemble)**.

Notebooks:
- `notebooks/dict_compare.ipynb` — randomized-build leaderboard + top runs  
- `notebooks/learn_from_motifs.ipynb` — full-file panels + params for `L_motifs_wav23`
