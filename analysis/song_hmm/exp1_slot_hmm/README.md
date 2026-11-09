# exp1_slot_hmm: Experiment 1

An HMM that finds **song sessions** in a long humpback recording, and inside each session the **songs** (cut at the Theme 6 → Theme 1 transition, following Payne & McVay) and the theme being sung each second. Built from scratch from the wav. The design is in [plan.md](plan.md), and the results are in [reports/exp1_slot_hmm.md](../reports/exp1_slot_hmm.md).

## Run

From the repository root:

```bash
python -m analysis.song_hmm.exp1_slot_hmm.train
```

```bash
python -m analysis.song_hmm.exp1_slot_hmm.report
```

`train` reads `C:\Users\hoang\Downloads\SOCAL_H_76_230426_193500_df100.x.x.wav` (override with `--wav`). It builds features and caches them in `work/features.npz`, then fits on the first 5 h. It writes `song_session_hmm.npz`, and for each half (`work/train/`, `work/test/`) it writes `sessions.csv`, `songs.csv` and `posteriors.npz`. `report` writes `reports/exp1_slot_hmm.md` (at the repository root), `figures/` and `work/report.json`.

## Model

- **Observation.** One 18-D vector per second, computed by `features.py`:
  - 16 log band levels from 30 to 1000 Hz, each above a rolling 10-minute noise floor;
  - the broadband level above its floor;
  - the within-second modulation of the broadband level.

  Each column is z-scored with the train half.
- **States (176), built in `topology.py`:**
  - 28 phrase slots from the six-theme drawing, each with 1–5 s duration states, giving 140 states. One Gaussian per slot.
  - One breath state per theme, `P1..P6`. They share the background Gaussian, so a breath does not end the session.
  - A background chain of 30 states, so background lasts at least 30 s. Its last state loops.
- **Edges.** Themes only repeat or move to the next theme: 1→2→…→6→1, with no skips and no going back. The last slot of each theme branches to repeat, next, breath or end session. Background can enter the first slot of any theme. The edge from T6 (or P6) into T1.1 is the song boundary.
- **Fixed priors.** P(end session) at a phrase end is 1e-3, and the background self-loop has a mean of 1 h. The recording has no labeled non-song, so these can't be learned. Everything else is learned with Baum–Welch in `hmm.py` (log-domain, numba).
- **Seeding.** The slot Gaussians and slot lengths are initialized from the motif CSV (train half only). The background Gaussian is initialized from the quietest 10% of train seconds.

## Files

| File | Role |
|---|---|
| `labels.py` | Motif CSV → per-second slot/theme, song spans, units |
| `features.py` | wav → per-second features |
| `topology.py` | States, sparse transition graph, parameters |
| `hmm.py` | Forward–backward, Viterbi, Baum–Welch, initialization, sampling |
| `decode.py` | State path → sessions and songs |
| `train.py` | Fit and decode |
| `report.py` | Metrics, baselines, synthetic recovery, figures |

Tests: `python -m pytest analysis/test/unit/song_hmm/exp1`.
