# Experiment 4 pipeline: wav → model input → training → decoding

Every step is listed with what it is, how it is built, and what it means. The plan is in [plan.md](plan.md); the checks and scores are in [metrics.md](metrics.md).

Steps 1–5 are shared by both options. Steps 6–7 are option 1, steps 8–10 option 2.

## Step 1: common audio format

- **What:** every recording at the same rate and frequency band.
- **How:**
  1. Anti-alias low-pass and resample to 2 kHz, so every file keeps only 0–1 kHz, like the main recording (`df100` = decimated by 100 from 200 kHz).
  2. Resample to 10 kHz, the rate GPL was run at.

  | Recording | Source rate | Steps needed |
  |---|---|---|
  | SOCAL_H_76_230426_193500 | 2 kHz | 10 kHz version exists (`Downloads/..._10kHz.wav`) |
  | SOCAL_H_76_230427_014845 | 200 kHz | 200 → 2 → 10 kHz |
  | CHNMS_NO01_231106_050140 | 10 kHz | 10 → 2 → 10 kHz (removes content above 1 kHz) |
- **Meaning:** atoms learned on one file can only be placed on another if both look the same to GPL. A different band or rate would change what the spectrogram shows.

## Step 2: GPL whitened spectrogram

- **What:** whitened spectrogram windows plus GPL call detections.
- **How:** `analysis/GPL_Python` with the settings of the existing run, `analysis/HMM/work/gpl_193500/`:

  | Setting | Value |
  |---|---|
  | FFT length / overlap | 1024 / 768 samples |
  | Frequency band | 10–1000 Hz, 102 rows |
  | Window | 75 s, 975 columns (≈ 0.077 s per column) |

  Each window has a JSON sidecar giving `start_sample` and `end_sample`. `CSV_*.csv` lists the GPL detections (start/end sample).
- **Meaning:** the spectrogram with steady noise removed per band. This is what dictionary atoms are learned from and matched against.

## Step 3: files

- **What:** the unit that the trained model is applied to.
- **How:**
  - The main recording's 480 windows are split into ten 1-hour files of 48 windows: hours 1–5 train, hours 6–10 test.
  - The 27 April and CHNMS recordings are one file each.
  - From here on, each file is processed **on its own**. Nothing computed from a test file is used for training.
- **Meaning:** this simulates "many wav files" with the data we have (plan §3).

## Step 4: motif detection

- **What:** placements of dictionary atoms (motifs) on each file.
- **How:**
  1. **Learn the dictionary (once, training files only):** `python -m analysis.dict_learning randomized-build --gpl-dir … --out DICT --seed S --loop-stats`.
     - The input windows are a sample from the training hours: the windows with the most GPL-detected call time, spread evenly over hours 1–5.
     - The first build is on R0's slice only (24 windows), to check the knobs. Then the build sample size is varied (metrics §3).
  2. **Place it on every file:** `python -m analysis.dict_learning matching-pursuit --gpl-dir … --dictionary DICT --out …`.
- **Output:**
  - atoms: `template_NNN.npy`, 102 rows × 10–20 columns, ≈ 0.8–1.5 s;
  - `matchings.json`, one row per placement: `template_id`, `window_id`, `t_start` (column), `amplitude`, `energy`.
- **Meaning:** each atom is a recurring sound, a candidate motif. Each placement says "this motif occurs here".
- **Option 2 outside the scope (CHNMS):** step 4.1 is run on that file itself. Its ids are only used within that file.

## Step 5: motif sequence

### 5a. Events

One row per placement:

| Field | How it is built | Meaning |
|---|---|---|
| `t0` | `(start_sample − 1) / 10000 + t_start × 75 / 975` s | absolute start time |
| `t1` | `t0 + width × 75 / 975` | absolute end time |
| `motif` | `template_id` | which motif |
| `energy`, `amplitude` | from MP | how strongly the atom fits |

Clean-up rules (values set on R0, then frozen):
1. **Weak placements:** drop events with `energy` below a threshold, set at the valley of the energy histogram (between real matches and noise fits).
2. **Negative amplitudes:** drop them or keep them as |amplitude| (plan, open question 4).
3. **Overlaps:** when two events overlap by more than half the shorter one, keep the higher-energy event. This handles one call explained by two atoms.

### 5b. Tokens

Sort the events by `t0`. Insert a gap token where the time from one event's `t1` to the next event's `t0` exceeds a threshold:

| Token | Gap | Meaning |
|---|---|---|
| `GAP_S` | 1.5–3 s | breath, or the pause between phrases |
| `GAP_L` | 3–60 s | long pause inside a session |
| `BREAK` | > 60 s | session break: sequences are cut here |

The thresholds are starting points, set from the histogram of inter-event gaps on R0.

**Output:** for each file, a list of sessions; each session is a token sequence with a time for every token.

**Meaning:** the song written as text. A theme should look like one short "word" (a phrase) repeated; a boundary is where the repeated word changes.

## Option 1

### Step 6 (option 1): input matrix

- **What:** each session's token sequence, encoded as integers 0 … K+2: K motif ids, then `GAP_S`, `GAP_L`, `BREAK`.
- **Training set:** all sessions of the five training files. The pair models use only clips of ±15 min around the A → B changes marked in R0 + R1.
- **Shape:** a list of 1-D integer arrays, one per session or clip, with their lengths. HMM libraries take multiple sequences this way.

### Step 7 (option 1): training and decoding

**1-pair-v0**
- **States:** `Other_in`, `A`, `B`, `Other_out`. Each loops on itself; the only moves are `Other_in → A → B → Other_out`.
- **Initialisation:**
  - `A` emissions = token counts in the 2 min before each marked change;
  - `B` emissions = token counts in the 2 min after;
  - `Other` emissions = token counts of all training data;
  - all with add-one smoothing.
  - Self-loop probabilities from the typical theme length in the marks.
- **Training:** Baum–Welch on all training clips together, up to 100 iterations or until the log-likelihood gain is < 1e-4 per token. Ten restarts with small random changes to the initial emissions; keep the best likelihood.
- **Decoding:** Viterbi. The boundary is the time of the first token decoded as `B`.

**1-pair-v1**
- As v0, but `A` and `B` are each a loop of slots: `A1 → … → Ak → A1`. A slot may repeat, which handles one sound detected as two events. Gap tokens may be emitted by any slot.
- k = the length of the most common phrase string in the A (or B) region of the marks. Slot j is initialised from the j-th token of those phrases.
- Training and decoding as v0.

**1-full**
- **States:** T theme loops (built like v1), connected in a cycle T1 → … → TT → T1, plus a silence state for `GAP_L` / `BREAK`. A session may start in any theme.
- **Initialisation:**
  1. Cut all training phrases at `GAP_S`.
  2. Cluster them by edit distance with k-medoids into T groups.
  3. Order the groups by when they typically occur relative to each other; this order is the theme order.
  4. Seed each theme's slots from its group's most common phrase.
- **Choosing T:** BIC over T = 4 … 8 on the training data.
- **Training:** Baum–Welch on all training sessions.
- **Decoding:** Viterbi on each session. A boundary is every move from one theme loop to the next. The theme labels come from the theme loops, so they are the same across files.

**Scope check (every file):** coverage = the fraction of GPL-detected call time covered by kept events. If it is below half of the training files' median coverage, the file is reported as **out of scope**.

## Option 2

### Step 8 (option 2): phrases

- **What:** the token sequence cut into phrases.
- **How:**
  - **Primary:** a phrase is the run of motif tokens between two gap tokens (`GAP_S`, `GAP_L` or `BREAK`).
  - **Fallback, if the phrase check fails on R0:** cut each run at the repeat period p, the lag where the motif string best matches itself shifted by p (normalised edit distance, p from 2 to 15 tokens).
- **Output:** for each session, a list of phrases. Each phrase has a start time, an end time and a motif string.
- **Meaning:** one step of the option 2 HMM. A theme is a run of similar phrases.

### Step 9 (option 2): input matrix

One row per phrase, five columns:

| Column | How it is built | Meaning |
|---|---|---|
| `d1` | normalised edit distance to the previous phrase's string: Levenshtein / longer length, between 0 and 1 | Small = repeat; large = new |
| `d2` | the same, to the phrase two steps back | Catches a single bad phrase in a run of repeats |
| `dmin3` | min(distance to each of the last 3 phrases) | Robust version of `d1` |
| `gap` | log(1 + pause before this phrase in seconds) | Longer pauses often come before a new theme |
| `len_ratio` | log(this phrase's token count / previous phrase's token count) | New theme, new phrase length |

- **Session starts:** for the first phrase of a session (no previous phrase), the distances are set to 1 and the row is flagged.
- **Standardisation:** each column becomes (value − mean) / std, with the mean and std computed on the training files and **frozen**.
- **Shape:** for each session, a matrix of (phrases × 5).

**Why the ids don't matter:** they are only compared with each other within one file, so any consistent labelling gives the same five numbers. That is what lets one trained model serve files whose motifs differ.

### Step 10 (option 2): training and decoding

- **States:** `C` (first phrase of a new theme); `R1 … Rm` (first m repeats); `R` (repeat loop); optionally `X` (transitional phrase).
- **Allowed moves:** C → R1 → … → Rm → R → R; R → C; R → X → C; C → C (single-phrase themes). m is 1, 2 or 3, chosen on R1.
- **Emissions:** a Gaussian with diagonal covariance over the 5 standardised columns, per state, with R1 … Rm sharing one Gaussian. If the fit is poor (metrics §2 on R1), a mixture of 2 per state.
- **Training:**
  1. **Supervised start:** label the phrases in the marked stretches (R0 + R1). The first phrase starting at or after a mark is `C`, the next m are `R1 … Rm`, the rest are `R`. Estimate means, variances and transition probabilities by counting.
  2. **Refinement:** Baum–Welch on the phrase matrices of all five training files, starting from step 1.
  3. Keep whichever of the two scores better on R1.
- **Decoding:** Viterbi on each session's phrase matrix. Every phrase decoded as `C` is a boundary, at that phrase's start time.

## Baselines

| Baseline | Input | How |
|---|---|---|
| B1 | Option 1 tokens (motif ids only) | Token histograms in two adjacent 60 s windows, sliding in 5 s steps. Jensen–Shannon distance; peaks above a threshold (set on R1), at least 60 s apart, are boundaries |
| B2 | Option 2 `d1` | Phrases with `d1` above a threshold (set on R1) are boundaries |
| B3 | GPL spectrogram, no motifs | 10 s blocks in 2 s steps. Cosine similarity between every pair of blocks, best over ±3 s of shift. Checkerboard novelty kernel (60 s) along the diagonal; peaks are boundaries |
| Exp 3 | Experiment 3's saved theme paths for M1 and M3 | Every change of theme in their path is a boundary |
