# Experiment 8 — theme–slot HMM on Exp7 wav23 dictionaries

Retrain Approach A theme–slot HMM on Exp7 `R_defaults` and `R_defaults_cov04` motif sequences (wav23). Train 1 h / test rest. Evaluate predicted theme boundaries with Jensen–Shannon vs a random-position null.

```bash
cd /path/to/panMP_whales
python -m analysis.song_hmm.exp8.run_pipeline all
# or: sync | motifs | hmm | themes | report
# optional: --only R_defaults
```

Kernel for notebooks: **Python (panMP / ensemble)**.

See `plan.md`, `pipeline.md`, and `../reports/exp8_theme_hmm_wav23.md`.
