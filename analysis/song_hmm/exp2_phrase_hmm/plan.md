# Experiment 2: phrase-window theme HMM — plan

## Context

**Experiment 1** (`analysis/song_hmm/exp1_slot_hmm/`, built on this branch):
- **Input:** one 18-number vector per second, computed from 16 frequency bands.
- **Model:** 28 one-second phrase slots in the 1→…→6→1 theme cycle.
- **What worked:** it finds sessions and a T6→T1 song cycle.
- **What failed:** its themes agree with the CSV no better than chance on the held-out half (0.27 agreement, against 0.33 for always guessing the most common theme). The diagnostics showed why:
  - per-second band averages cannot recognize units (0.18 balanced accuracy, chance 0.125);
  - they cannot recognize themes either.

**Why change the input.** Payne & McVay define a theme as "an unbroken sequence of similar phrases", each theme with its own phrase type. A test on 40 minutes showed that describing **12-second phrase windows** cut from a **0.05 s spectrogram** makes themes appear as clear repeating blocks in a self-similarity matrix. Those blocks recur about 13 minutes later, in the next song. The same windows still match the CSV themes only weakly (0.28 balanced accuracy, chance 0.17), so the CSV itself is in doubt.

**Experiment 2 changes three things:**
1. the input becomes phrase windows;
2. the model gets one state per theme instead of 28 slots;
3. a new set of **label-free metrics**, measured on the audio itself, joins the CSV-based metrics. They score both experiments and the CSV on the same scale.

**Report.** The report has three required sections: Experiment 1, Experiment 2, and the comparison. Every processing step and metric is written down in stored documents.

The recording and split are unchanged:
- **Recording:** `C:\Users\hoang\Downloads\SOCAL_H_76_230426_193500_df100.x.x.wav`, 2 kHz, 10 h.
- **Train:** the first 5 h.
- **Test:** the last 5 h.

## Where it lives

- Experiment 2 goes in a new package, `analysis/song_hmm/exp2_phrase_hmm/`.
- Experiment 1 stays untouched in `analysis/song_hmm/exp1_slot_hmm/`, so its numbers can be regenerated for the comparison.
- Shared code is imported, not copied:
  - `labels.load_labels`, the CSV reader;
  - the generic sparse-graph numba kernels `_forward`, `_backward`, `_viterbi` from `analysis/song_hmm/exp1_slot_hmm/hmm.py`.

**Stored documents**, all committed:

| File | Content |
|---|---|
| `analysis/song_hmm/exp2_phrase_hmm/plan.md` | This plan |
| `analysis/song_hmm/exp2_phrase_hmm/pipeline.md` | Exact wav → matrix processing for **both** experiments: every parameter, array shape and file |
| `analysis/song_hmm/exp2_phrase_hmm/metrics.md` | Definition, formula and caveat for every metric |
| `analysis/song_hmm/exp2_phrase_hmm/report.md` | Generated report with figures in `figures/` |
| `analysis/song_hmm/exp2_phrase_hmm/README.md` | How to run |

## Inputs in detail (wav → model matrix)

Each input below is listed as: what it is, how it is built, its shape and units, and what it means. `pipeline.md` carries this same text with the real shapes printed by the run.

### I-0. Raw audio `a[n]`

- **What:** the recording as a sequence of pressure samples, mono.
- **How built:** read with `scipy.io.wavfile.read(..., mmap=True)`, then int16 → float32 with no rescaling.
- **Shape and units:** 72,000,000 samples at 2,000 Hz, which is 36,000 s. Units are raw ADC counts; only relative level matters, because later steps work on log-power differences.
- **Meaning:** everything the hydrophone heard. The Nyquist limit is 1 kHz, so the highest frequency anything can describe is 1 kHz.
- **Split:** samples for 0–18,000 s are **train**, and 18,000–36,000 s are **test**.

### I-1. Band spectrogram `S[n, b]`

- **What:** how much energy each frequency band has in each 0.05 s frame.
- **How built:**
  1. Cut frame `n` = `a[100n : 100n + 512]`. The window is 0.256 s long and the hop is 0.05 s.
  2. Multiply by a Hann window and take the FFT. That gives 257 bins, 3.9 Hz apart.
  3. Take the power `|X_n(k)|²`.
  4. Sum the power over the bins in band `b`.
  5. Compute `S[n, b] = ln(Σ_{k∈b} |X_n(k)|² + 1e-6)`.
- **Bands:** 64 bands with log-spaced edges from 30 to 1000 Hz, so the edge ratio is 1.056, about **one semitone per band**. The lowest few bands contain no FFT bin and are dropped, which leaves **B = 59**.
- **Why log-spaced:** whale units are tonal sweeps. Log spacing gives each octave the same resolution, so the same sweep shape at a different pitch looks alike.
- **Shape and units:** (720,000 frames × 59 bands), 20 frames per second. Units are natural log of power; 1 unit ≈ 4.3 dB.
- **Meaning:** a fine-grained spectrogram. A unit, which lasts about 1 s, spans about 20 frames, so its pitch contour is kept.

### I-2. Noise-whitened spectrogram `Ŝ[n, b]`

- **What:** the energy above the background noise in each band.
- **How built:**
  1. For each band, take the median of `S` in each 1-minute block (1,200 frames).
  2. The floor `F[·, b]` for a block is the median of those block medians over 5 blocks (±2 min).
  3. Compute `Ŝ = max(S − F, 0)`.

  This is done separately inside train and inside test, so neither half informs the other.
- **Shape and units:** same as I-1. Units are log-power above the floor: 0 = at or below noise, 1 ≈ 4.3 dB above it.
- **Meaning:** the whale's sound with steady noise removed. Steady noise means shipping hum, a constant hydrophone tone, and the general level of the sea. Because the floor moves with a 5-minute window, a passing ship that raises the floor for minutes is partly absorbed. A loud unit stands out against the floor of its own few minutes.

### I-3. Phrase window `P_i`

- **What:** a 12 s stretch of `Ŝ`, long enough to hold one complete phrase of every theme. The longest phrase in the CSV is about 11 s, in theme 6.
- **How built:** window `i` covers frames `40i … 40i + 239`, i.e. it starts at `t_i = 2i` s and lasts 12 s. A new window starts every 2 s, so consecutive windows overlap by 10 s. Windows are made separately in each half, and a window never crosses 18,000 s.
- **Shape:** (240 frames × 59 bands) per window, about 8,995 windows per half.
- **Time stamp:** the window centre, `t_i + 6 s`.
- **Meaning:** "what one phrase sounds like around this moment". A theme is a run of similar phrases, so windows from the same theme should look alike wherever the phrase happens to start inside the window.

### I-4. Window descriptor `v_i` (295 numbers)

The descriptor summarizes `P_i` so that it does not depend on where the phrase starts inside the window. There are five statistics per band `b`; `n` runs over the window's frames.

| Statistic | Formula | Meaning |
|---|---|---|
| **level** `m_b` | `mean_n P[n,b]` | How much this pitch is used during the phrase. Across bands this is the phrase's **spectral profile**: which pitches the units sit at |
| **variability** `s_b` | `std_n P[n,b]` | Steady (low) or on/off (high). Separates pulsed units and gaps from continuous tones or noise at this pitch |
| **flux** `f_b` | `mean_n |P[n+1,b] − P[n,b]|` | How often energy starts or stops at this pitch, every 0.05 s. Separates choppy, rapid units from long moans |
| **up-sweep** `u_b` | `Σ_n P[n,b]·P[n+1,b+1] / Σ_n P[n,b]·(P[n+1,b−1] + P[n+1,b] + P[n+1,b+1]) + ε` | Of the energy at band b, the fraction that appears one semitone **higher** 0.05 s later. High for **rising** units, about 20 semitones per second or faster. 0 at the top band |
| **down-sweep** `d_b` | Same, with `P[n+1,b−1]` in the numerator | The fraction that moves one semitone **lower**: falling units. 0 at the bottom band |

- **Size:** 5 × 59 = **295 numbers** per window.
- **Units:** level and variability are in log-power above floor; flux is per frame; the sweeps are fractions from 0 to 1.
- **Loudness:** the sweeps are ratios, so they describe contour shape, not loudness. Level, variability and flux still carry loudness, and that is wanted, because it separates song from background.

### I-5. Standardized descriptor `z_i`

- **How built:** `z_i = (v_i − μ) / σ` for each of the 295 dimensions, where `μ` and `σ` are computed over **train** windows only.
- **Meaning:** puts all 295 numbers on one scale, so no statistic dominates just because of its units.

### I-6. Model input `x_i` (20 numbers), the matrix fed to the HMM

- **How built:** fit PCA on the train `z_i`, then set `x_i = Vᵀ z_i`, using the top **D = 20** principal directions.
- **Shape:** `X_train` and `X_test`, each about (8,995 × 20). Row i is window i in time order. One HMM step is 2 s.
- **Meaning:** the 20 most varied combinations of the phrase descriptors. The report prints:
  - the explained variance of the 20 components;
  - for the top 3 components, which band, statistic and pitch they weigh most, so they can be read. For example, "component 1 = level at 200–500 Hz" or "component 2 = up-sweep vs down-sweep".
- **Why 20:** a diagonal Gaussian per theme needs decorrelated inputs, and 20 dimensions keep the 6 theme Gaussians estimable from 5 h.
- **Caveat:** neighbouring rows overlap by 10 s of audio, so they are not independent. The HMM treats them as if they were. The minimum theme length of 10 steps and the learned stay probabilities are set with that in mind.

### I-7. Audio self-similarity matrix `K` (evaluation only, never fed to the model)

- **How built:** `K[i, j] = cos(z_i, z_j)`, the cosine similarity of standardized 295-D descriptors, on the test half.
- **Shape:** about 8,995 × 8,995.
- **Meaning:** how alike two moments sound at the phrase level.
  - A theme appears as a bright square block on the diagonal.
  - The same theme in the next song appears as a bright off-diagonal block about 13 minutes away.
- **Use:** it is the common yardstick for the label-free metrics. It judges Experiment 1, Experiment 2 and the CSV by the same rule, and it does not depend on any of their labels.

### I-8. Reference labels from the CSV

- **Source:** `backend/data/song_structure/SOCAL_H_76_230426_193500_df100.x.x.motif_segments.csv`, one row per unit.
  - **Columns:**
    - `motif`, the unit name;
    - `song`;
    - `theme`, 1–6;
    - `phrase_id`, `phrase_count`;
    - `start_unix`, `end_unix`.
  - **To seconds:** the unix times are converted to seconds from the recording start, 2023-04-26 19:35:00 UTC, which is taken from the file name.
- **Per-second theme:** the theme of the unit that covers the second's midpoint. Every second is covered.
- **Song spans:** the first and last unit time of each song. A song start is a reference song boundary.
- **Use:**
  - to score both experiments (metric group B);
  - to seed the Experiment 2b ablation;
  - in the main Experiment 2 run, only to name which of the 6 learned clusters is Theme 1.

### Experiment 1 inputs, for comparison

`pipeline.md` documents Experiment 1's inputs in the same format:

- **Spectrogram:**
  - 512-sample Hann window, 0.1 s hop;
  - 16 log-spaced bands from 30 to 1000 Hz;
  - per-band floor = rolling 10th percentile over 10 minutes.
- **Per-second vector, 18 numbers:**
  - 16 band levels above the floor, averaged over the second's 10 frames;
  - broadband level above its floor;
  - within-second std of the broadband level (modulation);
  - all z-scored with train statistics.
- **Meaning:** the average pitch profile and loudness of each second. It has no contour, no phrase context and no sweep direction. This is the gap Experiment 2 addresses.

### Cache

`work/phrase_features.npz` (git-ignored) stores:
- `X_train`, `X_test`;
- the window start times;
- the 295-D `z` for the test half, used for `K`;
- `μ`, `σ` and the PCA basis;
- every parameter above.

## Model architecture (Experiment 2)

**One step is 2 s, so one observation is one phrase window.**

| Block | States | Emission |
|---|---|---|
| Themes T1..T6 | Each theme is a chain of `MIN_THEME = 10` steps (20 s minimum, a few phrases), then a self-loop state: 6 × 10 = 60 | One diagonal Gaussian per theme (20-D), shared by its chain |
| Breath P1..P6 | 6 | Tied to background |
| Background | A chain of 15 steps (30 s minimum), then a self-loop | Its own Gaussian |

That is about 81 states in total.

**Edges:**
- **Inside a theme:** the chain moves forward. The theme's self-loop state can:
  - stay, i.e. keep repeating phrases;
  - move to the next theme's chain start, where theme 6 → theme 1 is the **song boundary**;
  - go to the breath state P_k;
  - end the session by going to background.
- **P_k:** stay, resume theme k, or move to the next theme.
- **Background:** after its minimum length it stays, or enters the start of any theme. Where a song starts is unknown, so entry is allowed at every theme.
- **Not allowed:** skipping a theme or going backward.

**Learned by Baum–Welch on the train half:**
- the theme and background Gaussians;
- the stay / next / breath split for each theme;
- the breath stay / resume / next probabilities;
- which theme a session enters at;
- the start distribution.

**Fixed priors, as in Experiment 1:**
- P(end session) = 1e-4 per step;
- background lasts 1 h on average.

**Initialization from the audio alone (main run):**
1. Run k-means with k = 6 on the train windows.
2. Smooth the cluster sequence with a 20 s majority filter.
3. Count transitions between clusters, then choose the cyclic order of the 6 clusters that best explains those transitions, trying all 120 cyclic orders.
4. The only thing left is which cluster is "Theme 1". That single choice is made by agreement with the CSV on the train half, and is documented as the only use of the CSV in this run.
5. The background Gaussian starts from the quietest 10% of windows.

**Ablation run (Experiment 2b):** the same model with the theme Gaussians seeded from the CSV themes on the train half. This separates the effect of the features from the effect of the seeding.

**Decoding:** Viterbi on the windows, then each second gets the state of the window whose centre is nearest. This gives per-second themes, sessions and songs, written in the same CSV formats as Experiment 1.

## Metrics (defined in `metrics.md`)

All metrics are reported for train and test, with **test as the headline**. They are computed the same way for Experiment 1, Experiment 2, Experiment 2b and, where it applies, **the CSV itself**.

**A. Fit, within each experiment only**
- Log-likelihood per second on held-out data, compared with two baselines on the same features: one Gaussian, and a 2-state HMM.
- Log-likelihoods of different feature sets **cannot** be compared across experiments, and the report says so.

**B. Agreement with the CSV** (a reference, not ground truth)
- Per-second theme accuracy, balanced accuracy and Cohen's κ.
- Majority-theme and chance baselines.
- Song-boundary precision, recall and F1 at ±15, ±30 and ±60 s.
- Number of complete songs, and song lengths compared with the CSV.
- Fraction of seconds in a session.

**C. Label-free audio structure**, measured with one common yardstick: the phrase-window self-similarity matrix (cosine similarity of the 295-D descriptors, test half).
- **Theme coherence:** mean similarity of window pairs in the same decoded theme segment, minus pairs in adjacent different segments. Higher means each decoded theme is acoustically uniform and different from its neighbours.
- **Theme identity:** across consecutive songs, similarity of theme k in song n to theme k in song n+1, minus its similarity to other themes. Higher means a theme label means the same phrase type every song.
- **Boundary–novelty alignment:** precision and recall of decoded theme changes against peaks of the novelty curve, at ±10 s. The novelty curve is a checkerboard kernel slid along the self-similarity diagonal, which peaks where the sound changes character.
- **Song regularity:** coefficient of variation of complete-song lengths, and the fraction of songs whose 6 theme lengths fall within the train interquartile range.

**D. Synthetic recovery**
- Sample from the fitted model, refit with the same initialization, and score against the true path.
- Report theme accuracy, plus session and song-boundary F1 at ±10 s.
- This measures what the model can do when its assumptions hold.

## Report structure (`report.md`)

**The report must contain these three sections. They are required, not optional:**

| Required section | Heading in `report.md` | Must cover |
|---|---|---|
| **Previous experiment** | `## Experiment 1 (previous): per-second slot HMM` | Its inputs, architecture, how it was trained, and its results under the same metrics. Also why it fell short |
| **Current experiment** | `## Experiment 2 (current): phrase-window theme HMM` | Why it changed, its inputs, architecture, initialization and results, plus the 2b ablation |
| **Comparison** | `## Comparing Experiment 1 and Experiment 2` | Side-by-side metrics, the same figures for both, and what improved, what did not, and why |

`report.py` asserts that all three headings are present before it writes the file. Other sections may be added around them.

Full order:

1. **Summary.** One table of headline test metrics: Experiment 1 vs Experiment 2 vs Experiment 2b vs the CSV.
2. **Data and split.** The recording, the split and the CSV caveats.
3. **Experiment 1 (previous): per-second slot HMM** (required).
   - Processing, summarized with a link to `pipeline.md`.
   - Architecture, with its diagram.
   - The changes made while building it: background minimum length, fixed session priors, log-domain forward–backward.
   - Results (metric groups A–D) and figures.
   - Diagnosis of why it failed: the unit and theme classification checks, and the lag test.
4. **Experiment 2 (current): phrase-window theme HMM** (required).
   - Motivation: the definition of a theme, and the self-similarity evidence.
   - Processing and architecture.
   - Initialization by clustering.
   - Results (A–D), figures, and the 2b ablation.
5. **Comparing Experiment 1 and Experiment 2** (required).
   - Side-by-side tables for groups B, C and D.
   - Self-similarity matrices with each experiment's and the CSV's theme boundaries drawn on top.
   - Timelines over the same 30 minutes.
   - What changed, what improved, what did not, and why.
6. **Limitations and next steps.**

## Files

| File | Purpose |
|---|---|
| `exp2_phrase_hmm/features.py` | Steps 1–8 above; cache |
| `exp2_phrase_hmm/topology.py` | The ~81-state theme graph and its parameters |
| `exp2_phrase_hmm/hmm.py` | Emissions, M-step, Baum–Welch, sampling (kernels imported from Experiment 1) |
| `exp2_phrase_hmm/init.py` | k-means seeding, cyclic ordering, Theme 1 anchor; CSV-seeded variant |
| `exp2_phrase_hmm/decode.py` | Window path → per-second path → sessions and songs |
| `exp2_phrase_hmm/metrics.py` | Groups B–D, shared by all experiments and the CSV |
| `exp2_phrase_hmm/train.py` | `python -m analysis.song_hmm.exp2_phrase_hmm.train [--seed audio\|csv]` |
| `exp2_phrase_hmm/report.py` | `python -m analysis.song_hmm.exp2_phrase_hmm.report`: reruns Experiment 1's decode if its outputs are missing, computes everything, writes `report.md`, `figures/` and `work/report.json` |
| `analysis/test/unit/song_hmm/exp2/test_*.py` | Tests |
| `.gitignore` | Add `analysis/song_hmm/exp2_phrase_hmm/work/` |

## Verification

1. **Unit tests** (`pytest analysis/test/unit/song_hmm/exp2`):
   - Topology:
     - every row sums to 1;
     - there are no theme skips or backward edges;
     - only theme 6 or P6 enters theme 1;
     - the minimum theme and background lengths hold.
   - Features:
     - array shapes are as specified;
     - no window crosses the split;
     - the train statistics and PCA basis do not change when the test half is altered;
     - an up-sweep tone gives a higher up-sweep statistic than a down-sweep tone.
   - Metrics:
     - theme coherence is high for a planted block matrix;
     - novelty peaks fall at the planted block edges;
     - boundary matching matches a hand-computed example.
   - HMM:
     - Baum–Welch log-likelihood never decreases;
     - forward and Viterbi match brute force on a 3-step sequence.
2. **Real run:**
   - Run `train --seed audio`, then `train --seed csv`, then `report`.
   - Check that `report.md` contains the three required sections: previous experiment, current experiment, comparison.
   - Check that every table is filled for train and test, and that the figures render.
   - Check that the CSV's own label-free metrics appear, as the reference row.
3. **Experiment 1:** existing tests still pass, and its numbers in the new report match `analysis/song_hmm/exp1_slot_hmm/report.md`.
