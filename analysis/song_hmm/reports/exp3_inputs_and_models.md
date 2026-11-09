# Song-session HMM: Experiment 3 report

Branch `dat-hmm`. Plan: [plan.md](../analysis/song_exp3/plan.md). Inputs: [pipeline.md](../analysis/song_exp3/pipeline.md). Metrics: [metrics.md](../analysis/song_exp3/metrics.md). Numbers: `work/eval.json`, `work/models/results.json`, `work/inputs/*.json`. Regenerate with `python -m analysis.song_exp3.evaluate` then `python -m analysis.song_exp3.report`.

## Summary

Test half (last 5 h, never used for fitting). Group H is **pending**: no hand labels yet (`reference/labels/`); group B is agreement with the CSV, which looks generated and is a reference, not truth; C is measured on the audio itself; R checks Payne & McVay's rules.

| test half | H bal. acc. | H song-start F1 ±30 s | B bal. acc. (chance 0.167) | B κ | B boundary F1 ±30 s | C coherence | C identity | C change-point prec. | sessions / gaps ≤ 60 s | complete songs / in 7–30 min | median song (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Exp 2 | pending | pending | 0.156 | 0.041 | 0.18 | 0.104 | 0.109 | 0.53 | 18 / 10 | 15 / 9 | 8.3 |
| Exp 2b | pending | pending | 0.244 | 0.171 | 0.15 | 0.200 | 0.144 | 0.50 | 28 / 14 | 6 / 6 | 10.4 |
| M1 theme HMM | pending | pending | 0.313 | 0.233 | 0.23 | 0.134 | 0.217 | 0.28 | 2 / 0 | 33 / 13 | 6.3 |
| M2 song-first | pending | pending | 0.231 | 0.150 | 0.04 | 0.053 | 0.129 | 0.24 | 2 / 0 | 21 / 21 | 12.8 |
| M3 units | pending | pending | 0.236 | 0.187 | 0.05 | 0.127 | 0.149 | 0.46 | 1 / 0 | 15 / 9 | 10.6 |
| M4 six theme HMMs | pending | pending | 0.285 | 0.232 | 0.16 | 0.126 | 0.055 | 0.38 | 4 / 0 | 48 / 17 | 4.6 |
| CSV (reference) | – | – | – | – | – | 0.089 | 0.120 | 0.36 | 1 / 0 | 24 / 24 | 12.3 |

In short:
- **Inputs mattered most.** The 2D-Fourier phrase patch (X2) roughly doubles the song-repeat score of Experiment 2's input (0.304 vs 0.148; chance ≈ 0.05), and wins with every spectrogram and noise setting. The chosen input is `S2_N1_X2_20s`.
- **Best model on the CSV: M1 theme HMM** (balanced accuracy 0.313, κ 0.233), against 0.244 / 0.171 for Experiment 2b. The gain is real but the level is still low.
- **The session rule now holds.** No Experiment 3 model ends a session on a pause of 60 s or less (Experiment 2 / 2b did 10 / 14 times).
- **Songs are too short in M1 theme HMM, M4 six theme HMMs** (median 6.3 min, 4.6 min; Payne & McVay: 7–30 min): the HMMs pass through all six themes too quickly. M2, which finds the song cycle first, gives songs in range by construction.
- **The CSV remains a weak reference** (lowest coherence of all labellings, song starts drifting ~110 s from the audio's own song-to-song alignment). The hand labels (group H) and the pair judgements (group P) will decide; both are pending until you fill them in.

## Data, split and reference

- Recording `SOCAL_H_76_230426_193500_df100.x.x.wav`: 2 kHz, mono, 10 h. Nothing above 1 kHz exists. **Train** 0–18,000 s, **test** 18,000–36,000 s; every statistic and fit uses train only.
- CSV reference: 53 songs, 30,923 units; covers every second with whole-second units, 8 songs skip themes, song starts drift a median of 110 s from the audio's alignment.
- Hand labels: two 30-minute test-half clips exported to `work/reference/`; instructions in [reference/README.md](../analysis/song_exp3/reference/README.md); **not labelled yet**.
- Pair judgements: the blind pair page is built (`pairs/page/`); **not judged yet**.

## Previous experiment: Experiment 2 and the pipeline check

Full reports: [Experiment 2](exp2_phrase_hmm.md), [pipeline check](exp2_pipeline_check.md); diagnostics in [`../song_hmm_phrase/diagnostics/`](../analysis/song_hmm_phrase/diagnostics/).

**Experiment 2** described each 2 s step by a 12 s phrase window from a 0.256 s / 0.05 s spectrogram with a 5-minute median noise floor, summarised as 295 order-free statistics (level, variability, flux, up/down sweep per band) and reduced to 20 PCA components. An 81-state theme HMM (theme chains of ≥ 20 s, pauses, background ≥ 30 s, T6 → T1 boundary) was seeded from k-means clusters (Exp 2) or from the CSV (Exp 2b).

| Experiment 2, test half | Exp 2 (audio seed) | Exp 2b (CSV seed) |
|---|---|---|
| theme balanced accuracy vs CSV | 0.156 | 0.244 |
| Cohen's κ vs CSV | 0.041 | 0.171 |
| sessions / gaps ≤ 60 s | 18 / 10 | 28 / 14 |
| theme coherence (audio) | 0.104 | 0.200 |

**What went wrong, from the pipeline check and the boundary diagnostics:**

- *Processing was correct but lossy.* Reading, STFT framing, chunking and band pooling all passed (inverse-STFT SNR 315 dB). But the median floor zeroed half of every band (song included), the 0.256 s window kept only 0.03 of a 10 Hz pulse train, and the 295 statistics + PCA-20 kept less 'which phrase is this' than the raw spectrogram.
- *Better settings existed.* On the song-repeat test the current setting scored 0.148; the best combination (6 short FFT + no whitening, 20 s) 0.252.
- *The model broke Payne & McVay's rules.* Themes T1 and T4 last ~4–6 s per song (single phrases) but the model forced ≥ 20 s; most session breaks were pauses ≤ 60 s, which by definition stay inside a session.
- *The CSV is unreliable.* Its song starts sit at inconsistent points of the audio's cycle, and even given its own song boundaries the audio's best 6-way split matches its themes no better than equal slices.
- *The audio repeats strongly once per song (≈ 13 min)* — the structure the professor suggested using.

<details><summary>Boundary diagnostics output</summary>

```
1. CSV seconds per theme per song (median [IQR]); models' minimum theme length: Exp 1 none (≥ 3 s phrase), Exp 2 20 s
   T1:      4 s  [3-5]   songs with < 20 s: 100%
   T2:     77 s  [28-174]   songs with < 20 s: 21%
   T3:     69 s  [19-191]   songs with < 20 s: 26%
   T4:      6 s  [3-7]   songs with < 20 s: 94%
   T5:    124 s  [32-239]   songs with < 20 s: 17%
   T6:    240 s  [130-348]   songs with < 20 s: 15%

2. decoded gaps (background runs) inside the recording, test half; a session ends only after a pause > 60 s
   Exp 1   gaps:  20   ≤ 60 s:  15   61-120 s:   1   > 120 s:   4
   Exp 2   gaps:  17   ≤ 60 s:  10   61-120 s:   1   > 120 s:   6
   Exp 2b  gaps:  27   ≤ 60 s:  14   61-120 s:   7   > 120 s:   6

3. song-scale repetition: mean cosine of phrase windows L minutes apart (2 s steps)
   train: 1:0.172 3:0.059 5:0.069 7:0.051 9:0.075 10:0.078 11:0.073 12:0.055 13:0.069 14:0.075 15:0.075 17:0.060 20:0.031 25:0.045 30:0.032
          best lag ≥ 5 min: 10 min;  CSV median song length 11.8 min
   test: 1:0.174 3:0.090 5:0.059 7:0.076 9:0.083 10:0.092 11:0.102 12:0.114 13:0.120 14:0.103 15:0.090 17:0.075 20:0.044 25:0.076 30:0.052
          best lag ≥ 5 min: 13 min;  CSV median song length 11.8 min
   local song period in 40-min blocks (min): median 13.2, IQR 12.4-14.5

4. oracle: CSV song boundaries given, each song split into 6 ordered segments from the phrase descriptors alone
   train: theme accuracy with known boundaries 0.234   (equal-length ordered split 0.247)
   test: theme accuracy with known boundaries 0.244   (equal-length ordered split 0.247)

5. phase check: best offset of the next CSV song start (s), searched over +-300 s
   pairs 51; |offset| <= 15 s: 6%;  <= 60 s: 31%;  median |offset| 110 s
   CSV song length median 11.8 min;  audio-aligned song length median 11.2 min;  corr(CSV length, audio length) 0.80;  mean best similarity 0.252
```
</details>

## Current experiment: Experiment 3

### Processing options and inputs

Every option and input is defined in [pipeline.md](../analysis/song_exp3/pipeline.md) (what it is, how it is built, shape, meaning, which model uses it).

| code | what |
|---|---|
| S1 / S2 / S3 | spectrogram: 512-sample window every 0.05 s / 256-sample window every 0.025 s / 512 below 150 Hz + 256 above, every 0.025 s |
| N1 / N2 / N3 / N4 | noise: median floor, clipped (Exp 2) / none (one offset per band) / 10th-percentile floor / PCEN adaptive gain |
| X1 | window statistics (Exp 2's descriptor), 12 or 20 s windows every 2 s, PCA-20/40 |
| X2 | 2D-Fourier magnitude of the window (rhythm and unit pattern, shift-invariant) + mean band level, PCA-20/40 |
| X3 | 0.25 s frame vectors (band levels, flux, sweeps), PCA-20: for M4 |
| X4 | detected units with energy / height / width measurements, grouped into unit types: for M3 |
| X5 | the raw spectrogram window, shift-tolerant cosine: screening reference |

### Input screening

Song-repeat hit rate on the test half (500 loud query windows; a hit = nearest window ≥ 5 min away lies 1 or 2 song periods away; chance ≈ 0.05). It ranks inputs; it is not an accuracy (correct matches 3+ songs away count as misses).

![](../analysis/song_exp3/figures/screening.png)

| input | PCA-20 hit | PCA-40 hit | CSV same-theme of the match |
|---|---|---|---|
| S2_N1_X2_20s | 0.304 | 0.298 | 0.60 |
| S1_N1_X2_20s | 0.290 | 0.278 | 0.56 |
| S1_N1_X2_12s | 0.280 | 0.270 | 0.57 |
| S3_N2_X2_20s | 0.274 | 0.272 | 0.54 |
| S3_N1_X2_20s | 0.274 | 0.274 | 0.54 |
| S2_N3_X2_20s | 0.272 | 0.252 | 0.59 |
| S2_N2_X2_20s | 0.268 | 0.248 | 0.56 |
| S2_N1_X2_12s | 0.266 | 0.268 | 0.56 |

**Unit detection (X4)** for each setting: units per minute in loud vs quiet minutes (a good detector finds units in song and few in quiet), median unit length, unit types (BIC), and the song-repeat hit of a bag-of-unit-types window:

| setting | units | per min loud / quiet | median length (s) | types | song-repeat hit |
|---|---|---|---|---|---|
| S1_N1 | 44,214 | 51 / 92 | 0.55 | 20 | 0.110 |
| S1_N2 | 16,520 | 45 / 4 | 0.35 | 18 | 0.136 |
| S1_N3 | 19,944 | 40 / 23 | 0.40 | 20 | 0.136 |
| S1_N4 | 64,747 | 105 / 110 | 0.25 | 20 | 0.122 |
| S2_N1 | 55,823 | 58 / 123 | 0.35 | 18 | 0.096 |
| S2_N2 | 19,458 | 58 / 5 | 0.23 | 20 | 0.174 |
| S2_N3 | 22,794 | 48 / 25 | 0.25 | 20 | 0.190 |
| S2_N4 | 100,084 | 145 / 185 | 0.17 | 18 | 0.128 |
| S3_N1 | 53,622 | 57 / 117 | 0.42 | 20 | 0.092 |
| S3_N2 | 20,209 | 60 / 5 | 0.23 | 20 | 0.130 |
| S3_N3 | 23,465 | 49 / 27 | 0.25 | 20 | 0.136 |
| S3_N4 | 107,478 | 158 / 199 | 0.17 | 20 | 0.108 |

Detected units on the three example stretches (S2 spectrogram, each noise option; cyan boxes = units):

![](../analysis/song_exp3/figures/units_overlay_S2_N1.png)
![](../analysis/song_exp3/figures/units_overlay_S2_N2.png)
![](../analysis/song_exp3/figures/units_overlay_S2_N3.png)
![](../analysis/song_exp3/figures/units_overlay_S2_N4.png)

**Chosen inputs** (`work/models/inputs_used.json`): M1 and M2 use `S2_N1_X2_20s`; M4 uses X3 frames from `S2_N1`; M3 uses units from `S2_N3`.

### Pair precision (group P)

**Pending.** The [blind pair page](https://claude.ai/artifact/2PHkxyNwseVqyTdkcd7dJM) shows each query with the match proposed by each input (and a random window), without saying which input proposed it. Once you judge the pairs, Claude reads the judgements into `work/pairs/judgments.json` and this table fills in.

### M1: one theme-level HMM

One state per theme (a theme can be a single 2 s window), a pause state per theme, a background chain of 60 s; from each theme: stay / next / pause / end; T6 → T1 is the song boundary. 2-component Gaussian mixture per theme on the chosen window input (PCA-40). Seeded from the CSV theme at each window (train), background from the quietest 10% of windows.

| half | H bal. acc. | B accuracy | B bal. acc. | B κ | B boundary F1 ±30 s | C coherence | C identity | C change-point prec. | sessions / gaps ≤ 60 s | songs / complete / in 7–30 min | median song (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | – | 0.420 | 0.385 | 0.293 | 0.13 | 0.121 | 0.164 | 0.35 | 8 / 0 | 42 / 29 / 13 | 5.9 |
| test | pending | 0.389 | 0.313 | 0.233 | 0.23 | 0.134 | 0.217 | 0.28 | 2 / 0 | 37 / 33 / 13 | 6.3 |

Per-theme recall against the CSV (test): T1 0.09, T2 0.28, T3 0.82, T4 0.11, T5 0.11, T6 0.47.

Log-likelihood per step: train -86.33, test -81.12 (comparable only within this model). Baum–Welch: 27 iterations.

Synthetic recovery (9,000 steps sampled from the fitted model, refit, decoded): theme accuracy 0.999, song vs background 1.000, song boundaries found 31 of 31.

### M2: song-boundary-first

Find the song period from the audio's own repetition (median over 40-minute blocks), learn a template of one song cycle, align the recording to it with a cyclic Viterbi (each window stays, moves one or two template phases; tempo prior and score sharpness chosen on the train half so the alignment completes as many cycles as the period predicts). Split the template's phases into T1…T6 in order (start and cut points chosen to match the CSV on the train half; T1 / T4 may be a single phase). Song boundary = passing the template's start. Background = > 60 s of audio that does not match the template.

| half | H bal. acc. | B accuracy | B bal. acc. | B κ | B boundary F1 ±30 s | C coherence | C identity | C change-point prec. | sessions / gaps ≤ 60 s | songs / complete / in 7–30 min | median song (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | – | 0.440 | 0.248 | 0.184 | 0.20 | 0.089 | 0.170 | 0.24 | 1 / 0 | 23 / 21 / 21 | 12.9 |
| test | pending | 0.386 | 0.231 | 0.150 | 0.04 | 0.053 | 0.129 | 0.24 | 2 / 0 | 25 / 21 / 21 | 12.8 |

Per-theme recall against the CSV (test): T1 0.00, T2 0.00, T3 0.29, T4 0.00, T5 0.18, T6 0.91.

Song period 12.8 min; tempo prior [0.02, 0.96, 0.02], sharpness 3.0 (chosen on train: expected 23.4 cycles); template phases per theme [1, 1, 79, 1, 45, 258]; CSV themes matched by the phase split on train 0.44.

### M3: unit-based

One step per detected unit, plus GAP (3–60 s pause) and BGT (> 60 s) tokens. The 28 phrase slots of the drawing, each emitting a unit type (categorical); a slot may repeat (one sung unit detected as several pieces); after a theme's last slot: repeat / next / pause / end. Seeded from the CSV slot at each unit's time.

| half | H bal. acc. | B accuracy | B bal. acc. | B κ | B boundary F1 ±30 s | C coherence | C identity | C change-point prec. | sessions / gaps ≤ 60 s | songs / complete / in 7–30 min | median song (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | – | 0.360 | 0.272 | 0.212 | 0.10 | 0.158 | 0.192 | 0.47 | 1 / 0 | 15 / 13 / 5 | 10.4 |
| test | pending | 0.324 | 0.236 | 0.187 | 0.05 | 0.127 | 0.149 | 0.46 | 1 / 0 | 17 / 15 / 9 | 10.6 |

Per-theme recall against the CSV (test): T1 0.28, T2 0.15, T3 0.15, T4 0.00, T5 0.14, T6 0.69.

Log-likelihood per step: train -2.32, test -2.24 (comparable only within this model). Baum–Welch: 39 iterations.

Synthetic recovery (20,000 steps sampled from the fitted model, refit, decoded): theme accuracy 0.975, song vs background 1.000, song boundaries found 29 of 28.

Unit types: 20; tokens: train 13122, test 11396.

### M4: six theme HMMs chained into a song model

Each theme is a left-to-right chain of its phrase slots (3, 3, 6, 3, 5, 8), each slot two states (≥ 0.5 s) with a 2-component Gaussian mixture on 0.25 s frame vectors (X3); the phrase may repeat. (1) Each theme HMM is trained alone on its CSV segments (train); (2) each is tested on picking out its own theme (one-vs-rest AUC on test CSV segments); (3) the six are chained T1 → … → T6 → T1 with pauses and ≥ 60 s background; (4) embedded Baum–Welch on the whole train half with only the theme order imposed; (5) Viterbi.

| half | H bal. acc. | B accuracy | B bal. acc. | B κ | B boundary F1 ±30 s | C coherence | C identity | C change-point prec. | sessions / gaps ≤ 60 s | songs / complete / in 7–30 min | median song (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | – | 0.410 | 0.282 | 0.228 | 0.19 | 0.187 | 0.097 | 0.39 | 1 / 0 | 49 / 47 / 18 | 5.0 |
| test | pending | 0.397 | 0.285 | 0.232 | 0.16 | 0.126 | 0.055 | 0.38 | 4 / 0 | 56 / 48 / 17 | 4.6 |

Per-theme recall against the CSV (test): T1 0.05, T2 0.16, T3 0.49, T4 0.15, T5 0.12, T6 0.76.

Log-likelihood per step: train -26.82, test -26.71 (comparable only within this model). Baum–Welch: 24 iterations.

Synthetic recovery (20,000 steps sampled from the fitted model, refit, decoded): theme accuracy 1.000, song vs background 1.000, song boundaries found 8 of 8.

![](../analysis/song_exp3/figures/m4_detection.png)

Theme detection on 144 test CSV segments: top-1 accuracy 0.32 after isolated training, 0.24 after embedded training (chance 0.17). AUC above 0.5 means the theme's HMM scores its own segments higher than the others'.

## Comparing Experiment 2 and Experiment 3

**Test half**

| model | B bal. acc. | B κ | B boundary F1 ±30 s | C coherence | C identity | C change-point prec. | sessions / gaps ≤ 60 s | complete songs / in 7–30 min | median song (min) | longest theme run (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| Exp 2 | 0.156 | 0.041 | 0.18 | 0.104 | 0.109 | 0.53 | 18 / 10 | 15 / 9 | 8.3 | 19.0 |
| Exp 2b | 0.244 | 0.171 | 0.15 | 0.200 | 0.144 | 0.50 | 28 / 14 | 6 / 6 | 10.4 | 32.7 |
| M1 theme HMM | 0.313 | 0.233 | 0.23 | 0.134 | 0.217 | 0.28 | 2 / 0 | 33 / 13 | 6.3 | 34.7 |
| M2 song-first | 0.231 | 0.150 | 0.04 | 0.053 | 0.129 | 0.24 | 2 / 0 | 21 / 21 | 12.8 | 14.0 |
| M3 units | 0.236 | 0.187 | 0.05 | 0.127 | 0.149 | 0.46 | 1 / 0 | 15 / 9 | 10.6 | 39.4 |
| M4 six theme HMMs | 0.285 | 0.232 | 0.16 | 0.126 | 0.055 | 0.38 | 4 / 0 | 48 / 17 | 4.6 | 8.0 |

**Train half**

| model | B bal. acc. | B κ | B boundary F1 ±30 s | C coherence | C identity | C change-point prec. | sessions / gaps ≤ 60 s | complete songs / in 7–30 min | median song (min) | longest theme run (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| Exp 2 | 0.140 | 0.017 | 0.22 | 0.043 | 0.129 | 0.56 | 17 / 8 | 19 / 6 | 6.2 | 36.3 |
| Exp 2b | 0.342 | 0.213 | 0.12 | 0.053 | 0.187 | 0.64 | 14 / 7 | 14 / 8 | 9.0 | 37.1 |
| M1 theme HMM | 0.385 | 0.293 | 0.13 | 0.121 | 0.164 | 0.35 | 8 / 0 | 29 / 13 | 5.9 | 10.1 |
| M2 song-first | 0.248 | 0.184 | 0.20 | 0.089 | 0.170 | 0.24 | 1 / 0 | 21 / 21 | 12.9 | 13.0 |
| M3 units | 0.272 | 0.212 | 0.10 | 0.158 | 0.192 | 0.47 | 1 / 0 | 13 / 5 | 10.4 | 22.9 |
| M4 six theme HMMs | 0.282 | 0.228 | 0.19 | 0.187 | 0.097 | 0.39 | 1 / 0 | 47 / 18 | 5.0 | 23.0 |

Per-theme recall against the CSV, test half:

| model | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| Exp 2 | 0.14 | 0.51 | 0.15 | 0.00 | 0.06 | 0.09 |
| Exp 2b | 0.23 | 0.34 | 0.22 | 0.01 | 0.27 | 0.40 |
| M1 theme HMM | 0.09 | 0.28 | 0.82 | 0.11 | 0.11 | 0.47 |
| M2 song-first | 0.00 | 0.00 | 0.29 | 0.00 | 0.18 | 0.91 |
| M3 units | 0.28 | 0.15 | 0.15 | 0.00 | 0.14 | 0.69 |
| M4 six theme HMMs | 0.05 | 0.16 | 0.49 | 0.15 | 0.12 | 0.76 |

![](../analysis/song_exp3/figures/timeline.png)

![](../analysis/song_exp3/figures/song_lengths.png)

![](../analysis/song_exp3/figures/confusions.png)

### What changed, what improved, what did not, and why

1. **The input was the biggest lever.** Keeping the phrase's rhythm and unit pattern (X2) instead of order-free statistics (X1) roughly doubled the song-repeat score (0.304 vs 0.148). Shortening the FFT and dropping the median floor helped X1 (the pipeline check's finding), but for X2 the median floor is slightly *better* than no floor (S2 X2 20 s: N1 0.304, N2 0.268): its rhythm and pattern terms do not depend on absolute level, so the floor's loss of level hurts it less.
2. **Same model family, better input and rules (Exp 2b → M1):** balanced accuracy 0.244 → 0.313, κ 0.171 → 0.233, sessions 28 → 2, theme identity across songs 0.144 → 0.217. M1 is the bridge: it differs from Exp 2b mainly by its input and the Payne & McVay rules.
3. **The rules fixed sessions.** Every Experiment 3 model has 0 session breaks at pauses ≤ 60 s (Exp 2: 10, Exp 2b: 14).
4. **Song length separates the approaches.** Median complete song: M1 theme HMM 6.3 min, M2 song-first 12.8 min, M3 units 10.6 min, M4 six theme HMMs 4.6 min (CSV 12.3 min; Payne & McVay 7–30 min). The HMMs that build songs theme by theme (M1, M4) pass through the cycle too fast: nothing in them knows how long a song is. M2 knows by construction (it aligns whole song cycles), but its theme labels inside the song are weaker.
5. **Six separate theme HMMs (M4) barely tell the themes apart from 0.25 s frames** (top-1 0.32 on CSV segments, chance 0.17; best AUC 0.77). Chaining them gives κ comparable to M1, but the theme models themselves are weak — the frame input, not the chaining, is the limit.
6. **Units (M3):** balanced accuracy 0.236, κ 0.187. Unit detection works (units appear in song and almost vanish in quiet with N2/N3), but detected blobs do not line up one-to-one with the drawing's units, so slot-by-slot emissions are noisy.
7. **Song boundaries against the CSV stay low for everyone** (F1 ±30 s in the table). Because the CSV's own song starts drift from the audio's cycle, this metric cannot separate the models; the hand labels must.
8. **Model capacity is not the bottleneck.** Every HMM recovers its own synthetic data almost perfectly.

## Limitations and next steps

- **Fill in the reference.** Label the two clips (`reference/README.md`) and judge the pairs. Every CSV-based number above is limited by the CSV; groups H and P are the real test.
- **Give the theme-by-theme models a song length.** Combine M2's song cycle with M1/M4's theme models: use M2's boundaries as hard song limits and let M1/M4 place the themes inside each song.
- **Better theme models for M4.** Use the X2 phrase input (or units) instead of 0.25 s frames for each theme HMM, so each detector sees whole phrases.
- **Unit detection.** Tune the detector against hand-marked units on a few minutes, so blobs match units one-to-one before M3's slot model.
- **Higher sample rate.** Not available for this recording; anything above 1 kHz is lost.
