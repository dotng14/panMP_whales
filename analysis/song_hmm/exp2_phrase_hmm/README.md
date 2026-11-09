# exp2_phrase_hmm: Experiment 2

This folder holds a phrase-window theme HMM. It detects humpback song sessions, splits them into songs at the Theme 6 → Theme 1 transition, and labels the theme being sung. It is the second experiment; the first is in [`analysis/song_hmm/exp1_slot_hmm/`](../exp1_slot_hmm/README.md), and it stays unchanged so the two can be compared.

## Documents

| File | What it is |
|---|---|
| [plan.md](plan.md) | The approved plan: inputs, architecture, metrics, report layout |
| [pipeline.md](pipeline.md) | Exactly how the wav becomes the matrix the model sees, for both experiments |
| [metrics.md](metrics.md) | Definition and caveats of every metric |
| [reports/exp2_phrase_hmm.md](../reports/exp2_phrase_hmm.md) | The results. Sections: previous experiment, current experiment, comparison |
| [reports/exp2_pipeline_check.md](../reports/exp2_pipeline_check.md) | The check of this pipeline: is it correct, and what does it lose |
| [diagnostics/](diagnostics/) | `exp1_feature_checks.py` and its stored output, quoted in the report |

## Run

From the repository root:

```bash
python -m analysis.song_hmm.exp2_phrase_hmm.train --seed audio
```

```bash
python -m analysis.song_hmm.exp2_phrase_hmm.train --seed csv
```

```bash
python -m analysis.song_hmm.exp2_phrase_hmm.report
```

- **`--seed audio`** is Experiment 2: k-means seeding, where the CSV only names Theme 1.
- **`--seed csv`** is Experiment 2b: the theme Gaussians are seeded from the CSV.
- **`report`** recomputes every metric for Experiments 1, 2 and 2b and for the CSV. It writes `reports/exp2_phrase_hmm.md` (at the repository root), `figures/` and `work/report.json`. It regenerates Experiment 1's outputs if they are missing, and refuses to write a report that lacks any of the three required sections.

**Files:**
- **Models:** `exp2_model.npz` and `exp2b_model.npz`.
- **Decoded output:** `work/<run>/<half>/sessions.csv`, `songs.csv` and `decoded.npz`, plus `posterior.npz`, in recording seconds.
- **Feature cache:** `work/phrase_features.npz`.
- The `work/` folder is git-ignored.

The default wav path is `C:\Users\hoang\Downloads\SOCAL_H_76_230426_193500_df100.x.x.wav`; override it with `--wav`.

## Code

| File | Role |
|---|---|
| `features.py` | wav → 0.05 s band spectrogram → noise whitening → 12 s windows → 295 statistics → PCA to 20 |
| `topology.py` | 81-state theme graph: theme chains of at least 20 s, breaths, background of at least 30 s |
| `hmm.py` | Emissions, M-step, Baum–Welch, sampling. The log-domain kernels are shared with Experiment 1 |
| `init.py` | k-means seeding with cyclic ordering, and the CSV seeding |
| `decode.py` | Converts every labelling (both experiments and the CSV) into per-second themes, sessions and songs |
| `metrics.py` | CSV-agreement and label-free metrics |
| `train.py`, `report.py`, `report_text.py` | Entry points and report text |

Tests: `python -m pytest analysis/test/unit/song_hmm/exp2`.
