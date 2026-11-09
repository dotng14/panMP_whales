# Experiment 3: input combinations × four model approaches — plan

## Context

**Experiment 1** (per-second slot HMM) and **Experiment 2** (phrase-window theme HMM) both scored poorly.

The pipeline check found that the processing is *implemented* correctly but throws away information that matters. Three steps do it:
- **Median noise floor:** it zeroes half of every band, quiet song included.
- **0.256 s FFT window:** it blurs pulses and short units.
- **Order-free 295 statistics, then PCA-20:** they lose the order of units within a phrase.

The boundary diagnostics found three more problems:
- **The models break Payne & McVay's rules.** Themes can be a single phrase (T1 ≈ 4 s, T4 ≈ 6 s per song), and pauses up to 60 s stay inside a session.
- **The CSV is unreliable.**
- **The audio repeats strongly** every ~13 min, once per song.

**Experiment 3 does three things:**
1. **Builds a trustworthy reference:**
   - you judge about 100 phrase/match pairs;
   - you or your professor hand-label 30–60 min of themes and song starts.
2. **Tries a grid of processing and input combinations**, scored first on cheap tests, then on the reference.
3. **Trains four model approaches** on the best inputs and compares them:
   - **M1:** a single theme-level HMM, as in Experiment 2;
   - **M2:** song-boundary-first;
   - **M3:** unit-based;
   - **M4 (your idea):** six theme HMMs, one per theme, chained into a song model that runs T1 → … → T6.

Fixed for everything:
- **Recording:** the 2 kHz file. No higher-rate version exists, so nothing above 1 kHz is available.
- **Split:** first 5 h train, last 5 h test.

**The report is a core deliverable.** It must contain:
- a section on the **previous experiment**: Experiment 2 plus the pipeline check;
- a section on the **current experiment**: Experiment 3;
- a section **comparing** the two.

Every input and metric is documented.

## Folder and documents

New package `analysis/song_exp3/`, committed except `work/`:

| File | Content |
|---|---|
| `plan.md` | This plan |
| `pipeline.md` | Every processing option and input type: what it is, how it is built, its shape and units, what it means, and which models use it |
| `metrics.md` | Every metric, including the new reference-based ones |
| `reference/README.md` | How to judge pairs and how to hand-label (instructions, file format) |
| `report.md` | Generated report with `figures/`. A script checks the three required sections before writing |
| `README.md` | How to run each phase |

Experiments 1 and 2 and the pipeline check are left untouched.

**Reused code:**
- `analysis/song_hmm_phrase/features.py` (spectrogram, bands, whitening, descriptors; already parametrised);
- `analysis/song_hmm/hmm.py` log-domain kernels (`_forward`, `_backward`, `_viterbi`);
- `analysis/song_hmm/labels.py`;
- `analysis/song_hmm_phrase/decode.py` and `metrics.py` (common decoded form, CSV and label-free metrics);
- `analysis/song_hmm_phrase/pipeline_check/check.py` (song-repeat test, patch similarity).

## Rules every model obeys

These follow from Payne & McVay and the diagnostics:
- **Fixed theme order:** themes run 1 → … → 6, with no skips and no going back.
- **Song boundary:** the T6 → T1 transition. The first song of a session may start at any theme.
- **Single-phrase themes are allowed:** no theme has a minimum longer than one phrase.
- **Sessions:** a session ends only after **more than 60 s** of background.
- **Timing:** frames are time-stamped at their centre, and the DC offset is removed when the file is read.

## Phase 0: the reference

### R-pairs: phrase-pair judging page

- **What it is:** a small private web page (an Artifact) that saves each judgement in its own shared database.
- **What it shows, per pair:**
  - two 20 s spectrogram images, with audio players: a query phrase and a candidate match;
  - buttons: same phrase type / different / unsure.
- **Which pairs:** about 120.
  - For each of 40 queries from the test half, show the top match under **several** input representations, plus one random window. Pooling the representations this way keeps the judging from favouring any one of them.
  - Each pair is shown once, in random order, with no hint of which representation produced it.
- **Score:** **pair precision** for a representation = the share of its top matches judged "same". Chance = the share of random pairs judged "same". This has a known ceiling of 1.

### R-labels: hand labels

- **What:** two stretches of 30 min each in the test half; one contains a song boundary and one is mid-session.
- **Tool:** Audacity (free). Load the audio clip I export, add a label track, and mark:
  1. each theme span, labelled `T1` … `T6`;
  2. each song start, labelled `S`;
  3. any non-song gap longer than 10 s, labelled `BG`.
- **Format:** export the labels as Audacity's tab-separated `start end label` file into `reference/labels/`.
- **Supporting files:** `reference/README.md` gives the steps, with spectrogram images of each CSV theme as examples (they may be wrong, so treat them as hints). `reference/labels.py` parses the export.
- **Used for:** the headline metrics (group H). The CSV only covers group B.

The report runs without the human reference. Sections H and P show "pending" until the judgements and labels exist; rerunning the report fills them in.

## Phase 1: processing and input combinations

### Processing options (stages I-0 … I-2)

| Code | Stage | Options |
|---|---|---|
| — | I-0 read | DC removed (all options) |
| S | I-1 spectrogram | **S1** 512 / 0.05 s (current); **S2** 256 / 0.025 s; **S3** multi-resolution: 512 below 150 Hz, 256 above |
| N | I-2 noise normalisation | **N1** median floor, clipped (current); **N2** none (one fixed offset per band); **N3** 10th-percentile floor; **N4** PCEN (per-band adaptive gain plus smoothing; standard in bioacoustics) |

### Input representations (what the models see)

Each is documented in `pipeline.md` with what it is, how it is built and what it means.

| Code | Input | How it's built | Meaning | Used by |
|---|---|---|---|---|
| **X1** | Window statistics | 5 statistics × bands on 20 s windows every 2 s (12 s also tried), then z-score and PCA to 20 / 40 | Pitch profile, choppiness, sweep direction of a phrase. No unit order | M1, M2 |
| **X2** | 2D-Fourier patch | Magnitude of the 2D FFT of each 20 s × bands window (time-shift invariant), log, PCA to 20 / 40 | Which pitches *and* which rhythms and sweep patterns repeat in the phrase. Keeps rhythm and order information that X1 loses | M1, M2 |
| **X3** | Frame vectors | 0.25 s frames: band levels from S·N averaged over each frame, plus the frame's flux and sweep, then PCA to 20 | The sound right now, at a time scale below the unit (units ≈ 1 s), so a sequence of these follows units in order | M4 |
| **X4** | Unit events | Cells above threshold in the normalised spectrogram, joined across time **and** frequency into blobs (one blob = one unit). Each unit gets about 10 numbers. Units are then clustered into **unit types** (GMM; K chosen by BIC, expected about 8–15) | Your energy / height / width idea. Each unit becomes a token; a phrase is a short token string; a theme is that string repeating | M3 (M4 variant) |
| **X5** | Raw patch | The S·N window itself, compared with a ±3 s shift-tolerant cosine | Everything in the window; the most information, but no compression | M2 (alignment), evaluation |

The X4 unit measurements:

| Measurement | Group |
|---|---|
| Total energy, peak level above floor | **energy** |
| Peak frequency, start frequency, end frequency, slope (semitones per second) | **height** |
| Duration, bandwidth | **width** |
| Pulse rate inside the unit, number of stacked harmonics, gap to the next unit | extras |

### Screening

**Window inputs (X1, X2):**
- **Grid:** 3 S × 4 N × 2 inputs × 2 window lengths (12 s, 20 s) × 2 sizes (PCA-20, PCA-40). That is 96 cheap runs.
- **First pass:**
  1. the song-repeat test from the pipeline check, as a proxy;
  2. the label-free metrics: within-theme coherence under a k-means labelling, and change-point alignment.
- **Second pass:** the top 6 combinations, plus X5 and the current setting, supply the pairs for R-pairs, and their **pair precision** decides.

**Unit inputs (X4)**, for each of the 3 S × 4 N detection settings:
- detection statistics: units per minute, the duration distribution, and the share of time covered;
- overlays of detected units on the spectrogram, for the three example stretches;
- **unit-type consistency across songs:** whether the token string of one song aligns with the next better than with shuffled strings.

The best detection setting goes forward.

**Frame vectors (X3):** use the best S·N from the window screening.

**Output:** `work/inputs/<code>.npz` with an input card for each (parameters, shapes, and its scores), plus a screening table in the report.

## Phase 2: four model approaches

All four produce the common decoded form (`theme[s]`, `session[s]`, song boundaries), so every metric applies to every model.

### M1: theme-level HMM (Experiment 2, updated)

- **Same as Experiment 2:** the 81-state theme graph.
- **What changes:**
  - inputs: the best X1 / X2;
  - the minimum theme length drops to one window, so single-phrase themes fit;
  - the background minimum rises to 60 s (30 steps);
  - emissions become a Gaussian mixture (2 components) per theme.
- **Seeding:** from the CSV (Experiment 2b showed this seeds better than the audio alone).
- **Role:** the baseline that shows how much the inputs alone help.

### M2: song-boundary-first

1. **Local song period:** the self-similarity of the best X1 / X2 at lags of 7–30 min, per 40-minute block.
2. **Align consecutive cycles:** run DTW between each song-length stretch and the next, using X5 patch similarity. This gives every window a **song phase** from 0 to 1, consistent across songs.
3. **Boundary:** the phase where T6 phrases switch to T1/T2 phrases. Chosen once:
   - from the hand labels if they exist;
   - otherwise as the phase with the strongest consistent change, named with the CSV.
4. **Themes inside each song:** an ordered 6-way split.
   - Each segment is at least one phrase.
   - Segment edges must match the same phase in the aligned neighbour songs (the theme templates are averaged across songs).
5. **Sessions:** background gaps longer than 60 s, from the loudness of X1 and N2.

### M3: unit-based

- **Input:** the sequence of unit tokens (X4), one step per unit, with gaps.
- **Model:** the six-theme slot graph from the drawing (28 slots), with **discrete emissions**: each slot holds a distribution over unit types.
  - Breath and background are handled by the gap tokens.
  - The same rules apply: fixed order, single-phrase themes allowed, 60 s session gaps.
- **Seeding:** from the drawing's unit names together with the CSV slot labels: map the CSV unit at each unit's time to its token to seed the slot distributions.
- **Training:** Baum–Welch on the train half.

### M4: six theme HMMs combined (your idea)

Built the way speech recognisers build sentences from word models:

1. **One HMM per theme.**
   - A left-to-right chain of phrase states, as many as the theme's slots in the drawing (3, 3, 6, 3, 5, 8).
   - Each state has a duration model and a 2-component Gaussian mixture emission on X3 frame vectors.
   - The last state can loop back to the first, so the phrase can repeat.
2. **Isolated training.** Each theme HMM is trained only on that theme's train-half segments from the CSV (a starting point only).
3. **Detection check.** Each theme HMM scores every test segment. The report shows how well each one picks out its own theme against the other five (ROC AUC), on the hand labels where they exist and on the CSV otherwise.
4. **Combine into one song model.** Chain T1 → T2 → … → T6 → (T1 of the next song = boundary), add breath states and a background state (minimum 60 s), and allow entry at any theme for the first song of a session.
5. **Embedded training.** Baum–Welch on the whole train half, where only the theme order is known, not the CSV positions. This lets the six models move away from the CSV.
6. **Decode** with Viterbi: songs run T1 … T6, and boundaries fall at T6 → T1.

**Variant M4-u:** the same, with X4 unit tokens instead of X3, if X4 screens well.

## Phase 3: evaluation and report

### Metrics (definitions in `metrics.md`)

| Group | What it scores | Status |
|---|---|---|
| **H** (headline) | Against the hand labels, test half: theme accuracy, balanced accuracy, Cohen's κ, song-start F1 (±15 / ±30 s), session-edge F1 (±30 s), per-theme recall (including T1 and T4) | new |
| **P** | Pair precision against chance, per input representation | new |
| **B** | Agreement with the CSV | as in Experiment 2 |
| **C** | Label-free: theme coherence, theme identity across songs, change-point precision, song regularity, longest single-theme run | as in Experiment 2 |
| **D** | Synthetic recovery for M1, M3 and M4 | as before |
| **S** | Screening: song-repeat proxy and the unit-detection statistics | new |
| **Rules** | Share of decoded sessions broken by gaps of 60 s or less (should be 0); share of complete songs with all six themes in order (structural) | new |

### Report structure (`report.md`)

1. **Summary:** headline H and P table, test half, with every model and the best inputs.
2. **Data, split, reference:** what the human reference covers, and the CSV caveats.
3. **`## Previous experiment: Experiment 2 and the pipeline check`** (required). The Experiment 2 recap, what the pipeline check and the boundary diagnostics found, and the numbers carried forward.
4. **`## Current experiment: Experiment 3`** (required):
   - 4.1 Processing and inputs: the S, N and X options, with a link to `pipeline.md`.
   - 4.2 Input screening: the grid table, figures, and pair precision.
   - 4.3 to 4.6, one subsection per model (M1–M4): architecture with a diagram, seeding, training, results (H, B, C, D, Rules), and figures. M4 also gets the per-theme detection AUCs.
5. **`## Comparing Experiment 2 and Experiment 3`** (required):
   - side-by-side tables (Experiment 2, Experiment 2b, M1–M4) on every metric group;
   - the same 30-minute timeline and self-similarity figures for all of them;
   - what improved, what did not, and why;
   - which input changes mattered, separated from which model changes did, using M1 as the bridge.
6. **Limitations and next steps.**

## Files

| Path | Purpose |
|---|---|
| `song_exp3/inputs/` (`spectrogram.py`, `pcen.py`, `window_features.py`, `fourier2d.py`, `frames.py`, `units.py`, `screen.py`) | Phase 1 |
| `song_exp3/reference/` (`pairs.py`, `labels.py`, `export_clips.py`, `page/` with the pair-judging page source, `README.md`) | Phase 0 |
| `song_exp3/models/` (`m1_theme.py`, `m2_songfirst.py`, `m3_units.py`, `m4_theme_hmms.py`, `graph.py` with shared graph and HMM helpers) | Phase 2 |
| `song_exp3/evaluate.py`, `report.py`, `report_text.py` | Phase 3 |
| `analysis/test/unit/song_exp3/` | Tests |
| `.gitignore` | Add `analysis/song_exp3/work/` |

## Order of work and checkpoints

1. Phase 1 window grid, then the unit-detection grid. **Checkpoint:** an interim screening section in the report, plus the unit overlays for you to look at.
2. Phase 0 pairs page, built from the screening winners. **You judge the pairs**, about 20 min. In parallel: export the two 30-minute clips and the labelling instructions, and **you label them**.
3. Phase 2, all four models, built while the judging and labelling happen.
4. Phase 3: the report. Sections H and P fill in once the reference files exist. Rerun the report after labelling.

## Verification

1. **Unit tests:**
   - PCEN matches a reference implementation on synthetic input;
   - the 2D-Fourier magnitude is unchanged by a time shift;
   - unit detection finds planted chirps and pulse trains with the right duration, frequency, slope and pulse rate;
   - the label parser reads an Audacity export;
   - the pair-precision computation is checked against a hand example;
   - **M4:** each theme HMM's rows sum to 1, the combined graph only allows T6 → T1 into T1, and Baum–Welch never decreases. Embedded training recovers planted theme segments on synthetic songs;
   - **M2:** DTW phase recovers a planted cycle with variable song lengths;
   - **M3:** discrete-emission forward and Viterbi match brute force.
2. **Rules check** on every model's output: no session break at gaps of 60 s or less; themes always in order.
3. **End to end:** `python -m analysis.song_exp3.report` writes `report.md` with the three required sections, every table filled for train and test, and H / P marked pending until the reference exists. After labelling, rerun and confirm H / P fill in.
4. Experiments 1 and 2 and the pipeline-check tests still pass.
