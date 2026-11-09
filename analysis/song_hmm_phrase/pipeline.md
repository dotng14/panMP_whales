# Pipeline: from the wav file to the matrix the model sees

This document covers both experiments:
- **Experiment 2** (current, `analysis/song_hmm_phrase/features.py`);
- **Experiment 1** (previous, `analysis/song_hmm/features.py`).

Shapes are those of the real run on the 10 h recording.

## Recording and split

| | |
|---|---|
| File | `C:\Users\hoang\Downloads\SOCAL_H_76_230426_193500_df100.x.x.wav` (override with `--wav`) |
| Format | RIFF WAV, PCM 16-bit, mono, 2,000 Hz, 72,000,000 samples = 36,000 s (10 h) |
| Start time | 2023-04-26 19:35:00 UTC, taken from the file name |
| Train | 0–18,000 s (first 5 h). Every statistic, noise floor, PCA basis, seed and Baum–Welch fit comes from here |
| Test | 18,000–36,000 s (last 5 h). Only decoded and scored |

Because the sample rate is 2 kHz, nothing above 1 kHz (the Nyquist frequency) exists in the data.

---

## Experiment 2 input

Run `python -m analysis.song_hmm_phrase.train`. The first call builds the features and caches them in `work/phrase_features.npz` (git-ignored). Later calls reuse that file while every parameter below is unchanged.

### I-0. Raw audio `a[n]`

- **What:** the hydrophone pressure signal.
- **How built:** `scipy.io.wavfile.read(path, mmap=True)`, then each chunk is cast int16 → float32. There is no rescaling, because every later step works on log-power differences.
- **Shape:** 72,000,000 samples.
- **Meaning:** everything the hydrophone heard, up to 1 kHz.

### I-1. Band spectrogram `S[n, b]`

- **What:** energy per frequency band per 0.05 s frame.
- **How built:**
  1. Frame `n` is `a[100n : 100n + 512]`: a 0.256 s window, hopped by 0.05 s. The audio is padded with zeros at the very end.
  2. Multiply by a Hann window and take the real FFT. That gives 257 bins, 3.906 Hz apart.
  3. Take the power `|X|²`.
  4. Sum the bins inside each band.
  5. `S = ln(band power + 1e-6)`.

  The work is done in 30-minute chunks to bound memory.
- **Bands:** 64 bands with log-spaced edges from 30 to 1000 Hz (`np.geomspace(30, 1000, 65)`). Neighbouring edges differ by a factor of 1.056, about one musical semitone. Below about 60 Hz some bands are narrower than one FFT bin and hold no bin; they are dropped, which leaves **59 bands**. Bin 256, at exactly 1000 Hz, is not used.
- **Shape:** **720,000 × 59** (20 frames per second), float32. Units are ln power; one unit ≈ 4.3 dB.
- **Meaning:** a fine spectrogram. A typical unit lasts about 1 s, so it spans about 20 frames and its pitch contour stays visible.

### I-2. Noise-whitened spectrogram `Ŝ[n, b]`

- **What:** energy above the background noise.
- **How built:** done separately for each half, so the halves never share statistics.
  1. Cut `S` into 1-minute blocks of 1,200 frames.
  2. For each band, take the median of each block.
  3. The noise floor of block `j` is the median of the block medians `j−2 … j+2`, i.e. ±2 minutes.
  4. `Ŝ = max(S − floor, 0)`.
- **Shape:** 720,000 × 59. 0 means at or below the noise; 1 means about 4.3 dB above it.
- **Meaning:** whale sound with steady noise removed. Steady noise means ship hum, constant tones, and the general sea level. A median is used so that song, which is present most of the time, does not become the floor. That works only while a band is quiet for more than half of the 5 minutes.

### I-3. Phrase window `P_i`

- **What:** 12 s of `Ŝ`. That holds one full phrase of any theme; the longest CSV phrase, in theme 6, is about 11 s.
- **How built:** window `i` = frames `40i … 40i + 239` of one half, i.e. it starts at `2i` s and lasts 12 s. A window must end inside its half. Consecutive windows overlap by 10 s.
- **Shape:** 240 × 59 per window. **8,995 windows in each half.** The window start times are stored as `t_train` and `t_test`, in seconds from the recording start.
- **Time stamp:** the window centre, start + 6 s, is the time a window stands for.
- **Meaning:** "the phrase being sung around this moment".

### I-4. Window descriptor `v_i` (295 numbers)

Each statistic is computed per band `b`. `n` runs over the 240 frames; the last three statistics use the 239 frame pairs (`n`, `n+1`). Window sums come from cumulative sums, so all windows are computed at once (`descriptors()`).

| # | Name | Formula | Range and units | What it tells |
|---|---|---|---|---|
| 1 | level `m_b` | `mean_n P[n,b]` | ≥ 0, ln power above floor | How much this pitch is used during the phrase. The 59 levels together form the phrase's **spectral profile** |
| 2 | variability `s_b` | `std_n P[n,b]` | ≥ 0 | On/off at this pitch (pulsed units, gaps) as opposed to a steady tone or noise |
| 3 | flux `f_b` | `mean_n |P[n+1,b] − P[n,b]|` | ≥ 0, per 0.05 s | How choppy the sound is at this pitch: many short units or one long moan |
| 4 | up-sweep `u_b` | `Σ P[n,b]·P[n+1,b+1] / Σ P[n,b]·(P[n+1,b−1] + P[n+1,b] + P[n+1,b+1])` | 0–1 | Share of the energy at `b` that is one semitone **higher** 0.05 s later. Rising units, about ≥ 20 semitones per second. 0 at the top band |
| 5 | down-sweep `d_b` | same, with `P[n+1,b−1]` on top | 0–1 | Share that is one semitone **lower**: falling units. 0 at the bottom band |

- **Layout:** `v = [m_1..m_59, s_1..s_59, f_1..f_59, u_1..u_59, d_1..d_59]`, i.e. 5 × 59 = **295 numbers**.
- **Shift invariance:** none of the five depends on where the phrase starts inside the window, so windows from the same theme look alike.
- **What carries what:** the sweep shares are ratios, so they describe contour shape and not loudness. Level, variability and flux do carry loudness, which is how song differs from background.

### I-5. Standardized descriptor `z_i`

- **How built:** `z = (v − μ) / σ` for each of the 295 dimensions. `μ` and `σ` are the mean and std over **train** windows, with σ floored at 1e-6.
- **Meaning:** every statistic on a common scale.
- **Stored:** as `z_train` and `z_test`. `z_test` is also the input to the evaluation self-similarity matrix, I-7.

### I-6. Model input `x_i`, the matrix fed to the HMM

- **How built:**
  1. Run an SVD on the centred train `z`.
  2. Keep the top **20** right singular vectors as the `basis` (295 × 20).
  3. `x = z @ basis`.
- **Explained variance:** the 20 components keep **64%** of the train variance. The first three keep 12.8%, 8.1% and 5.4%.
- **Shape:** `X_train` = 8,995 × 20, `X_test` = 8,995 × 20. Row `i` is window `i`, and **one HMM step = 2 s**.
- **Meaning:** the 20 most varied combinations of the phrase statistics. The report lists what the first three weigh most; PC1 is mostly level, variability and flux at 300–460 Hz.
- **Why 20:** each theme has one diagonal Gaussian, so the inputs should be decorrelated and few enough to estimate from 5 h.
- **Caveat:** neighbouring rows share 10 s of audio, but the HMM treats them as independent.

### I-7. Self-similarity `K` (evaluation only)

- **What:** `K[i, j] = cos(z_i − z̄, z_j − z̄)` on one half, where `z̄` is that half's mean descriptor.
- **Used for:** the label-free metrics in [metrics.md](metrics.md) and the comparison figures. The model never sees it.
- **How to read it:**
  - a theme shows up as a bright square on the diagonal;
  - the same theme in the next song shows up as a bright square about 13 minutes off the diagonal.

### I-8. Reference labels from the CSV

- **Source:** `backend/data/song_structure/SOCAL_H_76_230426_193500_df100.x.x.motif_segments.csv`
  - **Rows:** 30,923, one per unit.
  - **Columns:** `motif`, `song`, `theme`, `phrase_id`, `phrase_count`, `start_unix`, `end_unix`.
- **Read by:** `analysis/song_hmm/labels.py`.
  1. Times become seconds from the recording start.
  2. Units are grouped by (song, theme, phrase_count) into phrases.
  3. Each unit gets the slot at its position in the phrase.
- **Per-second theme:** the theme of the unit covering the second's midpoint. Every one of the 36,000 seconds is covered.
- **Per-window theme:** the per-second theme at the window centre.
- **Song starts:** the earliest unit of each song. Those starts, except the one at 0 s, are the reference song boundaries.
- **Uses:**
  - **Experiment 2b:** seeds the theme Gaussians;
  - **Experiment 2:** only chooses which of the six clusters is called Theme 1;
  - **scoring:** metric group B.

### Cache contents (`work/phrase_features.npz`)

| Key | Content |
|---|---|
| `key` | the parameter string the cache was built with |
| `X_train`, `X_test` | model inputs |
| `z_train`, `z_test` | standardized descriptors |
| `t_train`, `t_test` | window starts (s) |
| `level_train`, `level_test` | mean level over bands, used to seed background |
| `mu`, `sigma`, `basis`, `explained` | the train statistics and PCA basis |
| `band_edges` | band edges |
| `stats` | the statistic names |
| `spectrogram_shape`, `n_seconds`, `split_s`, `fps` | run dimensions |

---

## Experiment 1 input

Run `python -m analysis.song_hmm.train`. The cache is `analysis/song_hmm/work/features.npz`.

| Step | Experiment 1 setting |
|---|---|
| Read | Same as I-0 |
| Spectrogram | Hann window of 512 samples, **hop 200 (0.1 s)**, so 360,000 frames. Power summed into **16** log-spaced bands from 30 to 1000 Hz (a factor of 1.245 per band, about 4 semitones), plus a broadband 30–1000 Hz sum. `ln(power + 1e-12)` |
| Noise floor | For each band, the **10th percentile** of frame levels in a **10-minute** window, computed every 60 s and linearly interpolated. Computed separately in each half |
| Per-second vector, 18 numbers | (1–16) band level above floor, averaged over the second's 10 frames; (17) broadband level above its floor; (18) "modulation", the std of the broadband frame level within the second |
| Standardize | z-score each column with train statistics |
| Model matrix | 18,000 × 18 per half; **one HMM step = 1 s** |
| Meaning | The average pitch profile and loudness of one second. There is no pitch contour (frames are averaged), no sweep direction, and no phrase context. Pitch resolution is about 4 semitones |

## Differences that matter

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Time resolution inside a unit | 0.1 s, then averaged away | 0.05 s, kept through flux and sweeps |
| Pitch resolution | about 4 semitones | about 1 semitone |
| Context per observation | 1 s | 12 s (one phrase) |
| Contour (rising or falling) | no | yes (up-sweep, down-sweep) |
| Noise floor | 10th percentile over 10 min | median over 5 min, clipped at 0 |
| Observation size | 18 | 20 (PCA of 295) |
