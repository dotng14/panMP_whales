# Experiment 4: theme boundaries from motif sequences — plan

Companion documents in this folder:
- [pipeline.md](pipeline.md): every processing step from wav to model input, every input (what it is, how it is built, what it means), and the exact training and decoding procedures.
- [metrics.md](metrics.md): every check and score, how it is computed, and the targets.

## 1. Context

**Experiments 1–3** built the HMM's input straight from the spectrogram with hand-made features: per-second band numbers, window statistics, 2D-Fourier patches, and energy blobs grouped by GMM. The HMM then had to learn two things at once:
- what the sounds are;
- how they are ordered.

What we learned:
- **The input step is the hard part.** When the HMM scored badly, we could not tell whether the features or the model were at fault, so Experiment 3 searched 120 input combinations.
- **Units help only if they are real units.** M3 (unit tokens + slot HMM) found units in song and few in quiet. But its blobs did not match real units one to one, so its token sequence was noisy (balanced accuracy 0.236 vs 0.313 for the best model, M1).
- **The CSV is not a usable reference.** It covers 100% of the time with whole-second units and looks generated. Experiment 4 does not use it.

**Experiment 4 changes the order:**
1. Detect motifs first, with the existing dictionary tools (GPL → `randomized-build` → `matching-pursuit`).
2. Train an HMM on the resulting motif sequences.

Each half is checked on its own.

## 2. Goal

**One trained model that finds theme boundaries in many wav files**, not only the one it was trained on. A boundary is the moment one repeated phrase stops and a different repeated phrase starts.
- Start with **one pair of themes** (A → B; which pair does not matter).
- Then extend to **every theme boundary** in a recording.

Two ways to get a model that works across files are tried and compared.

| | **Option 1: shared vocabulary** | **Option 2: repetition model** |
|---|---|---|
| Idea | One dictionary and one HMM for a **song scope** (same population, same season), where all whales sing the same song | The HMM never sees motif ids, only **how phrases repeat**: is this phrase like the last one? That looks the same in every humpback song |
| HMM observes | Motif ids + gap tokens, one step per motif | A few numbers per phrase (distance to the previous phrases, the pause before it), one step per phrase |
| What is trained once and reused | Dictionary (atoms) + HMM (transitions, emissions over motif ids) | HMM (transitions, emissions over distances) |
| What runs on a new wav | GPL → MP with the shared dictionary → HMM | GPL → motif detection (shared dictionary in scope; a dictionary built on the file outside it) → phrase distances → HMM |
| Works on | Files in the same scope | Any song, in principle |
| Gives theme identity? | Yes: "theme 3" means the same in every file in scope | No, boundaries only. Identity can be added later by clustering the segments between boundaries |
| Fails when | The song changes (new season, other site): motifs are no longer in the dictionary | Phrases are split badly, or a theme drifts gradually |

Both options share the same front end (wav → GPL → dictionary → motif sequence) and the same reference and metrics, so they can be compared directly.

## 3. Data

| Recording | Rate, length | Role |
|---|---|---|
| `SOCAL_H_76_230426_193500_df100.x.x.wav` (Downloads) | 2 kHz, 10.0 h | Main recording. Split into ten **1-hour files**: hours 1–5 train, hours 6–10 test |
| `SOCAL_H_76_230427_014845.x.wav` (`project/stumpy/Whales_Song/`) | 200 kHz, 0.62 h | **The same instrument and the same time** as hours 6.23–6.85 of the main recording (it starts 6 h 13 m 45 s after it), at full rate. So it is **not independent data**. It tests that processing a file from a different source rate gives the same motifs and boundaries |
| `CHNMS_NO01_231106_050140.x.x.wav` (`project/data/smallExport/`) | 10 kHz, 0.08 h (5 min) | **Different site and season** (November 2023). Out-of-scope check: option 1 should flag it; option 2 is run on it and checked by eye |

**Main limitation.** All the data we have from the training scope is one 10-hour recording. Splitting it into 1-hour files and processing each one on its own (sharing nothing but the trained model) is how "many files" is simulated for now. A real multi-file test needs:
- **In scope:** more SOCAL_H_76 recordings from other days in April 2023.
- **Out of scope:** longer recordings from other sites or seasons.

Getting these is open question 1.

## 4. Reference (hand marks)

No CSV is used. You mark theme changes by eye on the spectrogram. Each mark is one time stamp, "theme ends / next theme starts here", with a note if a transitional phrase sits at the change.

| Set | Where | What to mark | Used for |
|---|---|---|---|
| **R0: development slice** | ~30 min in hours 1–5, containing at least one A → B change | Every theme change, plus the phrase starts in 3–5 min of it | Choosing knobs and rules; checking phrase splitting |
| **R1: training marks** | ~1 h more in hours 1–5, including 3–5 more A → B changes | Every theme change | Initialising the HMMs; tuning |
| **R2: test marks** | ~1 h in hours 6–10, **including the 6.23–6.85 h stretch** covered by the 27 April file | Every theme change | Final scores only, never for tuning |

**Rule for transitional phrases** (until you decide otherwise): the boundary is at the start of the first phrase that is clearly the new theme.

## 5. How a wav file is processed (shared by both options)

Full detail in [pipeline.md](pipeline.md). In short:

1. **Common format:** resample every recording to 2 kHz (anti-alias filter, so all files keep only 0–1 kHz like the main recording), then to 10 kHz for GPL.
   - The main recording already has a 10 kHz version.
   - The 200 kHz file needs both steps.
2. **GPL** with fixed settings → whitened spectrogram windows of 75 s:
   - FFT 1024 / overlap 768, 10–1000 Hz, 102 rows × 975 columns, ≈ 0.077 s per column;
   - plus call detections.
   - The main recording's GPL output already exists in `analysis/HMM/work/gpl_193500/` (480 windows).
3. **Split into 1-hour files** (48 windows each) for the main recording. Each file goes through the remaining steps separately.
4. **Motif detection:**
   - **Training:** `randomized-build` learns a dictionary from a sample of windows (from the training hours only).
   - **Every file:** `matching-pursuit` places that dictionary on all its windows.
   - **Output:** a list of placements (motif id, time, energy).
5. **Motif sequence:** convert placements to absolute times, drop weak placements, merge overlaps, sort, and insert gap tokens (`GAP_S` 1.5–3 s, `GAP_L` 3–60 s, `BREAK` > 60 s). The result is the song written as a token string.

**Gate:** the motif sequence must pass the input-step checks in [metrics.md](metrics.md) §1 before any HMM is trained. If it fails, we change the dictionary knobs or sequence rules, not the HMM.

## 6. Option 1: shared-vocabulary HMM

**Input:** the motif sequence of each file. One step per token; a token is one of the K motif ids or one of the 3 gap tokens.

**Training data:** the token sequences of the five training files (hours 1–5), split into sessions at `BREAK`.

**Models, in order:**

| Model | States | Emissions | Training |
|---|---|---|---|
| **1-pair-v0** | Other ↺ → A ↺ → B ↺ → Other ↺ | One categorical distribution over tokens per state | **Initialise** from the hand marks: tokens in the ~2 min before each marked A → B change seed A, those after it seed B, and the token frequencies of the whole training data seed Other. **Refine** with Baum–Welch on clips around every A → B change in R0 + R1 (±15 min each), all clips fitted jointly as multiple sequences |
| **1-pair-v1** | As v0, but A and B are each a loop of slots in phrase order, `[a1 → … → ak] ↺` | One categorical per slot | Slot count k = the length of the most common phrase string in the A (or B) region of the marks. Initialise slot j from the j-th token of those phrases, then Baum–Welch as above |
| **1-full** | A cycle of T theme loops, T1 → … → TT → T1, each a slot loop as in v1; plus a silence state reached on `GAP_L` / `BREAK` | One categorical per slot | Initialise by clustering the training phrases (k-medoids on edit distance between phrase strings) into T groups, ordered by their typical order in time. T is chosen by BIC over 4–8. Then Baum–Welch on all training sequences |

**Decoding a new file:** GPL → `matching-pursuit` with the shared dictionary → motif sequence → Viterbi with the trained HMM. A boundary is the time of the first token in a new theme's states.

**Scope check:** if the new file's coverage (metrics.md §1) falls below half of the training files' coverage, its output is marked **"out of scope"**: the song is probably not the one the dictionary knows.

## 7. Option 2: repetition HMM

**Input:** one step per **phrase**, described by numbers that do not depend on which motif ids occur:

| Feature | What it is | Meaning |
|---|---|---|
| `d1` | edit distance between this phrase's motif string and the previous phrase's, divided by the longer length | Small = the phrase repeats; large = something new |
| `d2` | the same, to the phrase two steps back | Catches one badly sung or badly detected phrase in a run of repeats |
| `dmin3` | the smallest distance to any of the last 3 phrases | Robust version of `d1` |
| `gap` | pause before the phrase, log seconds | Theme changes often follow a longer pause |
| `len_ratio` | log(this phrase's length / previous phrase's length) | A new theme often has a different phrase length |

**How phrases are cut:** at `GAP_S` / `GAP_L` tokens. If that check fails on R0's hand-marked phrase starts, use a fallback: phrases are cut at the repeat period of the motif string (the lag where the string best matches a shifted copy of itself).

**Motif detection for option 2:**
- **Inside the training scope:** use the shared dictionary of option 1.
- **Outside it (CHNMS):** build a dictionary on that file only. The ids only need to be **consistent within the file**, because option 2 never compares ids across files.
- The trained HMM itself is shared everywhere.

**Model:**
- **States, one step per phrase:**
  - `C`: first phrase of a new theme;
  - `R1 … Rm`: the first m repeats, a chain that enforces a minimum theme length;
  - `R`: further repeats, loops on itself;
  - optionally `X`: a transitional phrase between `R` and `C`.
- **Transitions:** C → R1 → … → Rm → R ↺, R → C (or R → X → C), and C → C, because some themes are a single phrase.
- **Emissions:** a Gaussian with diagonal covariance over the five standardised features, per state. If one Gaussian fits badly, a mixture of 2.
- **Training:**
  1. **Supervised start:** label the training phrases from the hand marks (R0 + R1). The first phrase after each mark is `C`, the following ones are `R1 … Rm` then `R`. Fit the means, variances and transition counts directly.
  2. **Refine** with Baum–Welch on the phrase sequences of all five training files, which are mostly unmarked.
  3. Keep the refined model only if it scores better on R1 than the supervised start.
  - The minimum repeat count m (1, 2 or 3) is chosen on R1.
- **Standardisation:** feature means and spreads are computed on the training files and **frozen**. New files use the same constants.

**Decoding a new file:** motif sequence → phrases → features → Viterbi. Every phrase decoded as `C` is a boundary, at the phrase's start time.

## 8. Baselines

Both options are compared with simpler methods, which tell us whether the HMM adds anything:

| Baseline | How | Uses motifs? |
|---|---|---|
| **B1: token-histogram novelty** | Slide two adjacent 60 s windows over the motif sequence, compare their token histograms (Jensen–Shannon distance), and take peaks above a threshold as boundaries | Yes, ids within the file |
| **B2: phrase-distance threshold** | Option 2's `d1`, thresholded, with no HMM | Yes, ids within the file |
| **B3: spectrogram self-similarity** | Compare every 10 s spectrogram block with every other (shift-tolerant cosine, ±3 s), then run a checkerboard novelty filter along the diagonal; peaks are boundaries | **No**: tests whether motifs help at all |
| **Exp 3: M1 and M3** | Their theme paths from Experiment 3; every change of theme is a boundary | No / blob units |

## 9. Evaluation

Full definitions in [metrics.md](metrics.md).

**What is scored:**
1. **Input-step checks** on the motif sequence (the gate).
2. **Boundary scores** against the hand marks:
   - error in seconds;
   - hit rate at ±5 / ±10 / ±30 s;
   - false alarms per hour;
   - for option 1-full, also whether the theme labels are in the right order.
3. **Cost:** time per hour of audio for each step.

**Test conditions:**

| Condition | Files | What it shows |
|---|---|---|
| **T1: dev** | hours 1–5 (R0, R1) | Only for tuning. Reported, never used as the final result |
| **T2: held-out files** | hours 6–10, each processed as its own 1-hour file (R2) | **Main result:** does the trained model work on files it has not seen? |
| **T3: same audio, different source** | 27 April file vs the matching stretch of hours 6.23–6.85 | Processing consistency: the boundaries should agree within ±5 s |
| **T4: out of scope** | CHNMS | Option 1: is the scope alarm raised? Option 2: boundaries checked by eye (no marks) |

## 10. Stages

Each stage ends with a check. The next stage starts only when the check passes, or once we understand why it failed.

| Stage | Work | Check before moving on |
|---|---|---|
| **0. Data and marks** | Run GPL on the 27 April and CHNMS files with the same settings. Split the main recording into 1-hour files. You make R0 (then R1 and R2) | GPL windows of all files have the same shape; R0 exists |
| **1. Motif detection** | `randomized-build` on R0's slice with `--loop-stats`, twice (two seeds). Then on a sample of the training hours. `matching-pursuit` on every file | Runtime recorded |
| **2. Motif sequence** | Placements → tokens. Figure: coloured motif ticks under the spectrogram. Text: one line per phrase | **Gate:** the input-step checks pass, and the A → B change is visible by eye |
| **3. Pair stage** | B1–B3, option 1-pair-v0, -v1, option 2, all on the A → B changes | Scores on R1 (dev). At least one option beats B1 |
| **4. Full stage** | Option 1-full; option 2 on all boundaries; B1–B3 | Scores on R1 (dev) |
| **5. Test** | Freeze every setting. Run T2, T3 and T4 | Final scores |
| **6. Report** | `reports/exp4_motif_boundary.md` | Three required sections (§11) |

## 11. Deliverables and layout

- **`analysis/song_hmm/plan/`:** this plan, `pipeline.md`, `metrics.md`.
- **`analysis/song_hmm/exp4_motif_boundary/`:** code. Computed files go in `work/` (git-ignored).
- **Tests in `analysis/test/unit/song_hmm/exp4/`:**
  - placements → motif sequence (times, overlaps, gap tokens);
  - phrase cutting and option 2 features;
  - each HMM recovering a known boundary from a synthetic sequence.
- **Report `analysis/song_hmm/reports/exp4_motif_boundary.md`**, written with `report_paths.write_report` and listed in `reports/README.md`:
  1. **Previous experiment:** Experiment 3, its inputs, models and result.
  2. **Current experiment:** Experiment 4, both options, stage by stage.
  3. **Comparison:** Experiment 3's M1 and M3, the baselines, option 1 and option 2, all scored on the same hand marks with the same boundary scores. The CSV-based numbers from Experiment 3 are quoted for context only.

## 12. Open questions

1. **More recordings:** can you get other days of SOCAL_H_76 (in scope) and longer recordings from other sites or seasons (out of scope)? Without them, T2 tests new hours of the same day, not new files in the full sense.
2. **Which theme pair first?** Suggestion: two themes that look clearly different on the spectrogram.
3. **Transitional phrases:** keep the rule in §4, or give them their own label (option 2's `X` state)?
4. **Negative MP amplitudes:** drop them, or keep them as |amplitude|? Decided in stage 2 by looking at what they match.
5. **R0:** will you pick and mark the 30-minute development slice, or should I propose one for you to confirm?
