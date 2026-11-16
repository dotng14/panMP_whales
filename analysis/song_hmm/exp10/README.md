# Experiment 10 — R_defaults on 193500

Rebuild dictionary with Exp7 **R_defaults** knobs on `SOCAL_H_76_230426_193500`, place motifs, train Approach A theme–slot HMM (7 h / 3 h), evaluate JS vs random.

```bash
python -m analysis.song_hmm.exp10.run_pipeline all
# stages: 0 | dict | mp | motifs | hmm | themes | report | all
```

Kernel: **Python (panMP / ensemble)**. Report: `../reports/exp10_r_defaults_193500.md`.
