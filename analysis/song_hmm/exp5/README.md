# Experiment 5 — theme boundaries (193500)

New experiment (not a revision of Exp4).

| Doc | |
|---|---|
| [plan.md](plan.md) | Goal, data split, approaches A/B, stages |
| [model_inputs.md](model_inputs.md) | Every input, model, parameters, expected results |
| [pipeline.md](pipeline.md) | Processing steps |
| [metrics.md](metrics.md) | JS + boundary scores |

**Data:** `193500` — train 0–7 h, test 7–10 h.  
**Notebooks:** [`notebooks/approachA_motifs.ipynb`](notebooks/approachA_motifs.ipynb), [`notebooks/approachB_signal.ipynb`](notebooks/approachB_signal.ipynb).  
**Report (theme boundaries, first run):** [`../reports/exp5_theme_boundaries.md`](../reports/exp5_theme_boundaries.md)  
**Revised no-overlap front end + motif map:** [`exp5_new/`](exp5_new/) — [`what_we_did.md`](exp5_new/what_we_did.md), [`report.md`](exp5_new/report.md)

```bash
cd /home/hdn008/projects/panMP_whales
python -m analysis.song_hmm.exp5.run_pipeline all --workers 8
# stages: 0 | 1a | 1b | 2 | 3 | dict | mp | seq | gate
```
