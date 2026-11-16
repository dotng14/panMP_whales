# Experiment 9 — many-theme HMM with slot self-loops

Same Exp7 dictionaries / 1h–rest split as Exp8. Differences: `T ∈ {8,10,12,14,16}`, shared `L_max=10`, **self-loops** on every slot.

```bash
python -m analysis.song_hmm.exp9.run_pipeline all
```

Report: `../reports/exp9_varlen_many_theme.md`.
