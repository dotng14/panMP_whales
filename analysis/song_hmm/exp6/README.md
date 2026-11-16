# Experiment 6 — no-overlap motifs, 30-min map, HMM, CSVs

Revised **Approach A** package for `193500`: no-overlap dictionary, **20 × 30-minute** motif panels, theme–slot HMM re-run, and aa32a7f-compatible exports.

| Doc | |
|---|---|
| [plan.md](plan.md) | Scope and deliverables |
| [pipeline.md](pipeline.md) | How to regenerate |
| Report | [`../reports/exp6_motif_segment_exports.md`](../reports/exp6_motif_segment_exports.md) |

```bash
cd /home/hdn008/projects/panMP_whales
# Exp5 compute (dict → MP → sequences → models → eval)
python -m analysis.song_hmm.exp5.run_pipeline all --force   # or stages 1a / 2 / 3
# Exp6 packaging (CSVs + manifest from Exp5 artifacts)
python -m analysis.song_hmm.exp6.run_pipeline all
```

**Primary figures:** `work/figures/motifs_by_30min/` (20 panels).  
**Notebook:** `../exp5/notebooks/approachA_motifs.ipynb`.
