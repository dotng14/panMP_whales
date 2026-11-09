# Song-session HMM (from scratch) — plan

## Context

The goal is to detect **song sessions** in a long recording and, inside each session, the **boundaries between songs**.

The old model in `analysis/HMM/` cannot do this. It has only song states and is trained on 53 pre-cut song spans. Those spans cover all 36,000 s, so the model has never seen non-song. Its z-normalized motif scores also throw away loudness.

This work starts from scratch:
- It uses no GPL outputs, no motif scores and no other cached products of the recording.
- Everything is computed from the wav.
- The only prior inputs are the phrase/theme architecture from the drawing and the motif-label CSV.

The model follows the Payne & McVay convention:
- **Fixed theme order.** Theme numbers are fixed, and the order is a closed cycle with no reversals: 1 → 2 → 3 → 4 → 5 → 6 → 1. No theme is ever omitted.
- **Fixed song boundary.** A song begins at Theme 1, so **the song boundary is the T6 → T1 transition**.
- **Partial first song.** A session may begin at any theme. Everything before the first T1 is a partial song.

**Where the work lives:**
- New, self-contained folder `analysis/song_hmm/`, with no imports from `analysis/HMM/`.
- Branch `dat-hmm`.
- `analysis/HMM/` stays untouched.

## Data and split

| Item | Value |
|---|---|
| Recording | `C:\Users\hoang\Downloads\SOCAL_H_76_230426_193500_df100.x.x.wav`: 2 kHz, mono, 16-bit PCM, 10 h (36,000 s), 144 MB. Path set as a constant, overridable with `--wav` |
| Reference labels | `backend/data/song_structure/SOCAL_H_76_230426_193500_df100.x.x.motif_segments.csv`: 53 songs, 30,923 units, theme/slot per unit |
| Train | First 5 h, 0–18,000 s |
| Test | Last 5 h, 18,000–36,000 s |

The test half is never used for fitting. The z-score stats, the initialization and Baum–Welch all use the train half only. The test half is only decoded and scored.

## Observations (from the wav, one vector per second)

1. **Read** with `scipy.io.wavfile` (no new dependency) and convert to float32.
2. **STFT:**
   - Hann window, 512 samples (0.256 s);
   - hop 200 samples (0.1 s), giving 10 frames per second;
   - frame power |X|².
3. **Bands.** 16 log-spaced bands from 30 to 1000 Hz. The 2 kHz recording has its Nyquist at 1 kHz.
4. **Noise floor.** For each band, the rolling 10th percentile of frame log-power over a 10-minute window. It is computed separately within each half, so the test half does not leak into train.
5. **Per-second features (K = 18):**
   - 16 band levels above the floor, each averaged over the 10 frames of the second. These capture spectral shape, which tells units, slots and themes apart.
   - Broadband level above the floor: the log of summed 30–1000 Hz power minus its own rolling floor. This separates song from background.
   - Modulation: the std of the broadband frame level within the second. Song units are modulated; steady noise is not.
6. **z-score** every column with the train-half mean and std.
7. **Cache** to `analysis/song_hmm/work/features.npz`, together with the parameters used to build it.

## Changes made while building

- **Background has a minimum duration of 30 s.** It is a chain of 30 states, so the model has 176 states instead of 147. Without the chain, the model used 1-second background gaps to jump between themes outside the 1→6 order.
- **Session on/off rates are a fixed prior, not learned.** These are P(end) = 1e-3 at a phrase end and a background mean of 1 h. Left free, they collapsed into short gap fillers, because the recording has no labeled non-song.
- **Forward–backward runs in the log domain.** A scaled probability-domain pass lost the posterior inside the background chain.
- **Figures are written to `analysis/song_hmm/figures/`, not `work/figures/`.** `work/` is git-ignored, and `report.md` needs its figures committed.

## Model architecture

### States (147)

| Block | States | Emission |
|---|---|---|
| Phrase slots | 28 slots from the drawing (3+3+6+3+5+8) × 5 duration states (1–5 s) = 140 | One diagonal Gaussian per slot, shared by its 5 duration states |
| Breath / pause | 1 per theme = 6 (`P1..P6`) | Tied to the background Gaussian |
| Background | 1 (`BG`) | Its own diagonal Gaussian |

Slot table, with unit and nominal duration taken from the drawing:

| Theme | Slots |
|---|---|
| T1 | U7, U2, U9 |
| T2 | U1, U8, U7 |
| T3 | U9, U11, U9, U6, U8, U7 |
| T4 | U2, U9, U6 |
| T5 | U8, U9, U3, U8, U7 |
| T6 | U9, U3, U8, U2, U7, U9, U11, U2 |

Emissions are per slot, not tied by unit name.

**Breath states.** Whales surface to breathe during a session. `Pk` is a silence between phrases of theme k. Because it remembers which theme the whale is in, a breath does not end the session or lose the position in the song. `Pk` and `BG` share an emission, so the only difference between them is how long they last:
- `Pk` stays short, with a self-loop mean of about 10 s;
- `BG` stays long, from minutes to hours.

### Transitions

All edges sit at phrase boundaries, as the drawing shows:

```mermaid
flowchart LR
  BG((BG)) -- enter session --> T1 & T2 & T3 & T4 & T5 & T6
  subgraph theme_k [Theme k]
    Tk1[Tk.1] --> Tk2[...] --> TkN[Tk.last]
    TkN -- repeat --> Tk1
    TkN -- breath --> Pk((Pk))
    Pk -- resume --> Tk1
  end
  TkN -- next --> Tnext[T(k+1).1]
  Pk -- next --> Tnext
  TkN -- end session --> BG
  T6last[T6.8] -- "next = SONG BOUNDARY" --> T11[T1.1]
```

- **Inside a slot.** From duration state d < 5, the chain either continues to d+1 or leaves the slot. At d = 5 it must leave.
- **Leaving a non-final slot.** The chain goes to the next slot's 1 s state.
- **Leaving the final slot of theme k.** This is a branch with 4 outcomes, learned per theme:
  - `repeat` → Tk.1
  - `next` → T(k+1).1, where theme 6 goes to T1.1; this is the **song-boundary edge**
  - `breath` → Pk
  - `end` → BG
- **`Pk`.** A self-loop, `resume` → Tk.1, or `next` → T(k+1).1. For P6, `next` is also a song boundary.
- **`BG`.** A self-loop, or entry into the first slot of any theme. The entry distribution over the 6 themes is learned. This is the "we don't know where the song starts" freedom, and it only exists at the start of a session.
- **Sequence start.** BG or the first slot of any theme, so a recording or half may open mid-session.
- **Structural zeros.** There are no edges that skip a theme or go backward. Theme order is enforced by the graph, not learned.

### Learning (Baum–Welch on the train half)

The first 18,000 s are treated as one sequence, with no song cuts.

Learned parameters:
- the slot Gaussians and the background Gaussian;
- the per-slot exit probabilities;
- the 4-way theme-end branch for each theme;
- the `Pk` stay / resume / next probabilities;
- BG stay and the theme-entry distribution;
- the start distribution.

Variances are floored, and probabilities are clipped away from 0. Iteration stops on a small relative log-likelihood gain or after 30 iterations.

### Initialization (train half only)

- **Slot Gaussians.** The mean and variance of the train-half feature vectors that the CSV assigns to each slot.
- **Slot exit probabilities.** Set from the CSV unit-length histogram for each slot, clipped to 1–5 s.
- **Theme-end branches.** `repeat` comes from the drawing (T1 0.008, T2 0.984, T3 0.943, T4 0.585, T5 0.922, T6 0.95). `breath` starts at 0.02, `end` at 0.001, and `next` takes the remainder.
- **Background Gaussian.** Fitted on the 10% of train seconds with the lowest broadband level, because the CSV marks nothing as non-song.
- **Stay probabilities.** BG stay starts at 1 − 1/3600 and `Pk` stay at 0.9.

The CSV is a seed, not truth. It covers 100% of the recording, uses whole-second units, and 8 of its songs skip themes. Baum–Welch is free to move away from it.

### Decoding

Each half is decoded as its own sequence.

- **Viterbi path.** From it:
  - **sessions** are maximal runs of non-BG states, with breaths included;
  - **songs** are split at each T6 → T1 transition (T6.last or P6 → T1.1);
  - the first segment is flagged `partial_start` if it does not begin at T1.1, and the last is flagged `partial_end` if it does not end in theme 6.
- **Forward–backward.** P(song) per second (1 − P(BG)) and P(theme) per second.
- **Written to `analysis/song_hmm/work/{train,test}/`:**
  - `sessions.csv` (session, start_s, end_s)
  - `songs.csv` (session, song, start_s, end_s, first_theme, last_theme, partial flags)
  - `posteriors.npz`

## Model report

`python -m analysis.song_hmm.report` writes `analysis/song_hmm/report.md`, with its figures in `analysis/song_hmm/work/figures/` and the numbers in `analysis/song_hmm/work/report.json`. Every metric is shown for **train** and **test** side by side.

The CSV is the only reference, and it looks generated. The report therefore treats CSV agreement as *agreement with the reference*, not as accuracy. Accuracy against true labels comes from the synthetic recording (section 6).

1. **Fit and generalization**
   - Baum–Welch log-likelihood curve.
   - Log-likelihood per second, train vs test. The size of the gap shows whether the model overfits.
   - The same figure compared with two baselines: one Gaussian for every second, and a two-state model with BG and a single "song" Gaussian.
2. **Session detection**
   - Number of sessions, their total time, and the fraction of seconds decoded as song.
   - Time spent in `Pk` (breaths) vs `BG`.
   - The fraction of CSV-song seconds decoded as song. The CSV marks every second as song.
   - A timeline figure over the full 10 h, with the split marked, showing:
     - the spectrogram;
     - the broadband level;
     - P(song);
     - the decoded theme track;
     - sessions and song boundaries.
3. **Song boundaries (T6 → T1)**
   - Number of complete songs vs CSV songs in each half.
   - Boundary precision, recall and F1 against the CSV song starts at ±5 s and ±15 s.
   - Median absolute offset.
   - Decoded vs CSV song-duration histograms.
4. **Theme labeling**
   - Per-second theme accuracy vs the CSV, and a 6×6 confusion matrix.
   - Mean posterior mass on the CSV theme.
   - Decoded vs labeled seconds per theme.
5. **Learned structure**
   - A per-theme table of repeat / next / breath / end probabilities.
   - Expected phrases per theme vs the CSV.
   - Session entry distribution.
   - Slot duration distributions.
6. **Synthetic recovery**
   - Sample 5 h from the fitted model and refit from the same initialization procedure.
   - Score session and boundary F1 and theme accuracy against the **true** path.
   - This shows how well the model can do when its own assumptions hold.

## Files

| File | Purpose |
|---|---|
| `analysis/song_hmm/plan.md` | This plan. Written first |
| `analysis/song_hmm/__init__.py` | Marks the folder as a package |
| `analysis/song_hmm/labels.py` | Parses the CSV into per-second slot/theme, song spans and per-slot unit-length histograms |
| `analysis/song_hmm/features.py` | wav → STFT → band levels → noise floor → 18-D per-second features, z-scored with train stats; caches to `work/features.npz` |
| `analysis/song_hmm/topology.py` | Slot table; builds the 147-state sparse graph (dst, log-prob, edge kind and tie group) from the parameters |
| `analysis/song_hmm/hmm.py` | numba log-space forward, backward/E-step and Viterbi on the sparse graph; M-step; `baum_welch`; `sample` for the synthetic check |
| `analysis/song_hmm/decode.py` | Viterbi/posteriors → sessions, songs, CSVs |
| `analysis/song_hmm/train.py` | `python -m analysis.song_hmm.train [--wav PATH]`: features → init → fit on train → save `song_session_hmm.npz` → decode train and test |
| `analysis/song_hmm/report.py` | `python -m analysis.song_hmm.report`: metrics, figures, `report.md`, `work/report.json` |
| `analysis/song_hmm/README.md` | Architecture and how to run it |
| `analysis/test/unit/song_hmm/test_topology.py`, `test_hmm.py`, `test_features.py` | Tests |

`work/` products should be git-ignored, following the existing `.gitignore` pattern for `analysis/HMM/work`. `report.md` and the model `.npz` are committed.

## Verification

1. **Unit tests** (`pytest analysis/test/unit/song_hmm`).
   - Topology:
     - every row sums to 1;
     - no edge skips a theme or goes backward;
     - the only cross-theme edges into T1.1 start at T6.last or P6;
     - BG reaches only first slots.
   - HMM:
     - forward log-likelihood matches brute-force enumeration on a tiny graph;
     - Viterbi matches brute force;
     - Baum–Welch log-likelihood never decreases on synthetic data.
   - Features:
     - a synthetic wav with a tone burst gives level and modulation peaks at the right seconds;
     - the train z-score stats do not change when the test half is altered.
2. **Synthetic recovery.** Run as part of the report: session boundaries within ±5 s and song boundaries within ±5 s on the sampled data.
3. **Real run.**
   - Run `python -m analysis.song_hmm.train`, then `python -m analysis.song_hmm.report`.
   - Check that `report.md` lists every metric for train and test, and that the figures render.
   - Look over the timeline figure and `songs.csv` by hand.
   - The CSV-agreement numbers are diagnostics, not pass/fail.
