# Experiment 4 — `014845` only

Plan: [`../plan/plan.md`](../plan/plan.md). Report: [`../reports/exp4_motif_boundary.md`](../reports/exp4_motif_boundary.md).

## Split

| Half | Windows | Time |
|---|---|---|
| Train | 0–14 | ≈ 0–18.75 min |
| Eval | 15–29 | ≈ 18.75–37.5 min |

Dictionary is learned on the **train half** of this file. Results: `work/results/014845/`.

## Run

```bash
python -m analysis.song_hmm.exp4.run_014845 --workers 8
```

Kernel for notebooks: **Python (panMP / ensemble)**.

- `notebooks/option1_themes.ipynb`
- `notebooks/option2_themes.ipynb`
- `notebooks/spectrogram_gallery.html`
