# Experiment 3 pipeline: from the wav to every model input

Experiment 3 tries several ways to process the audio and to build model inputs, and screens them before training. This page defines each option and each input:
- what it is;
- how it is built;
- its shape and units;
- what it means;
- which models use it.

Code: `analysis/song_hmm/exp3/inputs/`. The Experiment 1 and 2 pipelines are in [../exp2_phrase_hmm/pipeline.md](../exp2_phrase_hmm/pipeline.md), and the check of the Experiment 2 pipeline is in [reports/exp2_pipeline_check.md](../reports/exp2_pipeline_check.md).

## Recording, split and common steps

| | |
|---|---|
| File | `C:\Users\hoang\Downloads\SOCAL_H_76_230426_193500_df100.x.x.wav`: 2 kHz, mono, int16, 72,000,000 samples = 36,000 s. Nothing above 1 kHz exists (×100 decimation at the source) |
| Train / test | 0–18,000 s / 18,000–36,000 s. Every statistic (floors, z-scores, PCA, unit-type clusters, model fits) is computed on the train half; noise floors are computed inside each half |
| DC offset | The recording mean (+631 counts) is subtracted before anything else (`dc_removed`) |
| Time stamps | Every frame and window is stamped at its **centre**, so the model's times line up with the audio exactly |

## Stage I-1: spectrogram (S)

Each option produces `S[n, b]` = ln(power in log band b during frame n). Bands run from 30 to 1000 Hz with log-spaced edges, about one semitone each (`np.geomspace(30, 1000, 65)`). Bands that contain no FFT bin are dropped; no power is lost, because those bins belong to neighbouring bands.

| Code | Window | Hop | Frames per s | Bands | Meaning / why |
|---|---|---|---|---|---|
| **S1** | 512 samples (0.256 s) | 0.05 s | 20 | 59 | Experiment 2's setting. Fine pitch (3.9 Hz bins), but it blurs anything faster than about 4 changes per second: it keeps only 3% of a 10 Hz pulse train |
| **S2** | 256 samples (0.128 s) | 0.025 s | 40 | 50 | Twice the time detail: keeps pulses and fast sweeps (27% of a 10 Hz train, 72% at 5 Hz). The bins are 7.8 Hz apart, so fewer bands survive below ~120 Hz |
| **S3** | 512 below 150 Hz, 256 above | 0.025 s | 40 | 6 + 46 | Fine pitch where it matters (low notes), fine time elsewhere. The 256-sample frames are shifted by 64 ms, so both sets of frames share centres |

## Stage I-2: noise normalisation (N)

| Code | What it does | Meaning / why |
|---|---|---|
| **N1** | `S − F`, clipped at 0. `F` is the median of 1-minute block medians over ±2 min, per band, inside each half | Experiment 2's setting. Removes steady noise, but the whale sings most of the time, so the median *is* song. Half of every band is zeroed and loud and quiet periods look alike |
| **N2** | `S − median_band(S)`: one constant per band over the half | Keeps everything, including the loud/quiet difference. Slow noise (ships, the 50 Hz-multiple instrument lines) stays in |
| **N3** | Like N1 but with the 10th percentile instead of the median | A floor set by the quiet moments: removes much less song than N1 |
| **N4** | PCEN (per-channel energy normalisation): `E / (ε + M)^0.98`, offset 2, power 0.5. `M` is `E` smoothed per band with a 2 s time constant; `E` is the band energy scaled so each band's median is 1 | An automatic gain control per band. It suppresses whatever has been steady for the last few seconds and lifts onsets. Matches librosa's `pcen` after the first ~20 s (warm-up) |

## Model inputs (X)

### X1: window statistics (as in Experiment 2)

- **What:** a phrase-length window (12 s or 20 s, one every 2 s), summarised per band by 5 statistics:
  - **level:** mean value;
  - **variability:** standard deviation;
  - **flux:** mean |change| between frames;
  - **up-sweep:** share of energy that appears one band higher in the next frame;
  - **down-sweep:** the same, one band lower.
- **Shape:** 5 × bands values per window → z-score (train) → PCA to 20 or 40 components (train).
- **Meaning:** which pitches the phrase uses, how choppy it is, and which way its units sweep. The **order** of units inside the phrase is lost.
- **Used by:** M1 and M2, if it screens best.

### X2: 2D-Fourier patch

- **What:** the window (12 s or 20 s, every 2 s) averaged into 0.1 s columns, giving a (time × bands) image.
- **How it's built:**
  1. Subtract the image's mean.
  2. Take the magnitude of its 2D Fourier transform. Keep:
     - time modulations from 0 to 3 Hz (the phrase rhythm and unit rate);
     - frequency modulations −15 … +15 cycles across the band axis. Both signs are kept, so rising and falling patterns differ.
  3. Apply `log1p`, and append the mean level per band (so absolute pitch is kept).
  4. z-score, then PCA to 20 or 40 (all statistics from the train half).
- **Shape:** 3 Hz × window length + 1 time modulations × 31 frequency modulations + bands, before PCA. With a 20 s window that is 61 × 31 + bands ≈ 1,941.
- **Meaning:** what a phrase sounds like, *including* its rhythm and the shape of its units. The magnitude does not change when the phrase starts earlier or later inside the window, so two windows of the same theme match even when they are out of step.
- **Used by:** M1 and M2, if it screens best.

### X3: frame vectors

- **What:** the normalised spectrogram averaged into 0.25 s frames. For each frame:
  - band levels;
  - flux (mean |change| between short frames);
  - up-sweep share and down-sweep share;

  then z-score and PCA to 20 (train).
- **Shape:** (frames × 20), 4 frames per second, 72,000 per half.
- **Meaning:** the sound right now, at a time scale below a unit (~1 s). A sequence of frames follows the units of a phrase in order.
- **Used by:** M4, taken from the S × N of the best window input.

### X4: unit events (your energy / height / width idea)

- **What:** each sound in the spectrogram found as a blob in time and frequency, then measured.
- **How it's built (`inputs/units.py`):**
  1. **Score.** Robust per-band score `z = (value − band median) / spread`, where spread = max(MAD, ½ standard deviation). The ½ standard deviation is needed after a clipped floor (N1, N3), where half of a band is exactly 0 and its MAD is 0.
  2. **Mark and join.** A cell is "on" when `z > 3`. Gaps of up to 0.1 s in time are closed, so the pulses inside one unit (the paper's subunits) join into one unit, and a one-band gap in frequency is bridged.
  3. **Blobs.** Joined cells form a blob. Blobs shorter than 0.05 s or smaller than 4 cells are dropped. Blobs overlapping in time by at least 70% are one sound (for example stacked harmonics) and are merged.
  4. **Measure each unit:**

     | Measurement | Group |
     |---|---|
     | Total energy (sum of z above 0), peak level `peak_z` | **energy** |
     | Peak frequency, start and end frequency, slope (semitones per second) | **height** |
     | Duration, bandwidth (semitones) | **width** |
     | Pulse rate inside the unit (dominant fast modulation ≥ 3 Hz), number of spectral peaks (harmonics), gap to the next unit | extras |

     Start and end frequency come from a straight-line fit of the unit's pitch over time, because the first and last frames are smeared by the window.
  5. **Unit types.** A Gaussian mixture on 8 standardised shape measurements of the train-half units (log duration, peak, start and end pitch, slope, bandwidth, pulsed or not, harmonics). The number of types is chosen by BIC between 6 and 20.
- **Shape:** one row per unit, 13 values plus a type label.
- **Meaning:** the song written as a sequence of unit types, the way Payne & McVay describe it: unit → phrase → theme.
- **Used by:** M3.
- **For screening:** each 20 s window becomes a bag of unit types plus type bigrams (square-rooted counts).

### X5: raw patch

- **What:** the normalised spectrogram window itself (20 s), compared with another window by the best cosine over ±3 s of time shift. It is averaged to 0.4 s columns for speed.
- **Meaning:** every bit of information in the window, with no compression. It is the reference for how much X1 and X2 lose.
- **Used by:** screening, and the pair-judging page.

## Screening

- **Grid:**
  - **window inputs:** 3 S × 4 N × {X1, X2} × {12 s, 20 s} × {PCA-20, PCA-40}, plus X5 for each S × N;
  - **unit inputs:** X4 for each S × N.
- **Score, on the test half:** the same 500 loud query windows as the pipeline check. For each, the nearest window at least 5 minutes away is a **hit** when it lies 1 or 2 song periods away (13 ± 2 or 26 ± 2 min). Chance ≈ 0.05.
- **Also recorded:**
  - **far share:** the share of nearest windows at least 7 minutes away (another song);
  - **CSV same-theme:** the share of nearest windows with the same CSV theme (a weak reference).
- **Final choice:** pair precision from the human pair judging, once you have judged them. Until then the song-repeat hit decides which input the models use.

The choice made by `run_models.py` is written to `work/models/inputs_used.json` and reported in [reports/exp3_inputs_and_models.md](../reports/exp3_inputs_and_models.md).
