# Pipeline check: does wav → matrix keep the information in the recording?

## Context

Experiments 1 and 2 both fed the HMM a matrix built from the wav file, and both underperformed. Before Experiment 3, we need to know whether the processing itself is correct and keeps what matters in the recording, so that a weak model result can't be blamed on a broken or lossy input without us knowing.

The pipeline under test is Experiment 2's (`analysis/song_hmm/exp2_phrase_hmm/features.py`), which Experiment 3 would reuse:

- **I-0** wav, 2 kHz
- **I-1** STFT: 512-sample (0.256 s) Hann window, 0.05 s hop, 59 log bands from 30 to 1000 Hz
- **I-2** remove a rolling 5-minute per-band median, then clip at 0
- **I-3** 12 s windows every 2 s
- **I-4** 5 statistics × 59 bands = 295
- **I-5** z-score
- **I-6** PCA down to 20

Experiment 1's pipeline is checked only where it differs. That means its resolution: 0.1 s frames, 16 bands, 1 s averages.

Every step except reading the wav loses something by design. The check answers three questions per step:
1. **Is it implemented correctly?** Pass/fail tests.
2. **What does it throw away?** Measured as a number.
3. **Does what it throws away matter for whale song?** Judged against what the song contains: unit lengths, pulse rates, sweeps, pitch range, and the song-to-song repeat.

Where a step loses something that matters, the check tries a few alternative settings and recommends the best one. Each alternative is judged on whether that step keeps the information; no model is retrained here.

One limit can't be fixed at all. The file was decimated ×100 at the source (`df100`), so nothing above 1 kHz exists in it, and anything the whales sang above 1 kHz is gone before this pipeline starts. The report states this up front.

## Where it lives

New folder `analysis/song_hmm/exp2_phrase_hmm/pipeline_check/`, all committed except `work/`:

| File | Content |
|---|---|
| `plan.md` | This plan |
| `check.py` | `python -m analysis.song_hmm.exp2_phrase_hmm.pipeline_check.check`: runs every check below and writes the report |
| `report.md` | Result: a pass/fail table, then one section per step (what it keeps, what it loses, whether that matters), then recommendations |
| `figures/` | One figure per stage for the same example stretches |
| `work/` | Git-ignored. Audio reconstructions for listening and cached arrays |

`check.py` reuses `features.py` (`band_matrix`, `band_spectrogram`, `whiten`, `descriptors`, `load_or_build`) so that the code being checked is the real code, not a copy. It also uses `analysis/song_hmm/exp1_slot_hmm/labels.py` for the CSV, only to pick example stretches. The alternative settings are passed as parameters. Any small refactor that `features.py` needs for this, such as letting `NFFT` and `HOP` be arguments, keeps the default behaviour byte-identical, and the existing tests prove it.

**Example stretches** used in the figures and audio:
- 2 minutes around a song boundary, where the theme changes;
- 2 minutes inside a long T6 stretch;
- 2 minutes of the quietest background, from the start of the recording.

## Checks

### A. Reading the file (I-0): correctness

| Check | Pass if |
|---|---|
| Header: rate, channels, sample count | 2,000 Hz, mono, 72,000,000 samples = 36,000 s. The "chunk not understood" warning is explained (an extra metadata chunk) and the data chunk length equals file size − header |
| Values | No NaN. Fraction of clipped samples (\|x\| ≥ 32,767) reported; pass if < 0.01% |
| DC offset | Mean per 10 minutes reported; pass if \|mean\| < 1% of RMS |
| Drop-outs | Runs of exactly-zero or constant samples longer than 0.1 s, listed with their times |
| Time origin | The start time in the file name is consistent with the CSV (CSV first unit at 0 s) |

### B. What frequencies are in the data (I-0 → band limits)

- **Long-term spectrum** from 0 to 1000 Hz, shown as the 10th, 50th and 90th percentile across time. It is drawn separately for loud stretches (top 20% level) and quiet stretches (bottom 20% level).
- **Energy outside 30–1000 Hz:** the share of energy below 30 Hz, and how much of that is song rather than noise. Measured as the loud-minus-quiet difference below 30 Hz.
- **Top-end roll-off:** where the decimation filter's roll-off starts below 1 kHz, i.e. how much of 30–1000 Hz is actually usable.
- **Pass if:** the song energy outside 30–1000 Hz is under 5% of the song energy inside it. Otherwise the band limits are changed.

### C. STFT framing (I-1): correctness and invertibility

| Check | Pass if |
|---|---|
| Inverse STFT of the complex STFT (overlap-add, `scipy.signal.istft`) on 10 minutes | Reconstruction SNR > 40 dB. This shows the framing loses nothing before the magnitude is taken |
| Chunked vs one-pass spectrogram on a 2-minute span that crosses a chunk edge | Max abs difference < 1e-4 |
| Time stamps | A 0.1 s, 300 Hz tone burst inserted at a known time lands in frame `t / 0.05 ± 1` |
| Band pooling | Every FFT bin from 30 to 1000 Hz belongs to exactly one band. Summed band power equals summed bin power to float precision |

### D. Time and frequency resolution vs. the song (I-1)

These measure what the song itself contains, from the raw audio, and compare it with what 0.256 s frames every 0.05 s can follow.

- **Pulse and modulation rates.**
  - Take the 30–1000 Hz envelope of the raw audio at 200 Hz.
  - Compute its modulation spectrum over loud stretches.
  - Report the share of modulation energy above 2, 5 and 10 Hz.
  - Payne & McVay's subunits are pulses heard only when slowed down, so they may sit above what the frames follow. A 0.256 s Hann window smooths out modulation above about 4 Hz, and the hop's Nyquist limit is 10 Hz.
- **Unit lengths:** the distribution of above-floor energy runs in the I-2 spectrogram, i.e. how long the continuous sounds actually are.
- **Sweep rates:** the distribution of ridge slopes, in semitones per second, in loud frames. Compare with what the up/down-sweep statistics can detect: a one-band shift within 0.05 s is ≥ 20 semitones per second, so slower glides are invisible to them.
- **Alternatives tried:**

  | Setting | Window | Hop | Frequency step |
  |---|---|---|---|
  | (a) current | 512 (0.256 s) | 0.05 s | 3.9 Hz |
  | (b) shorter window | 256 (0.128 s) | 0.025 s | 7.8 Hz |
  | (c) longer window | 1024 (0.512 s) | 0.05 s | 2 Hz |

  For each: how much of the modulation it keeps, and how sharp a test chirp and a test pulse train come out. A test pulse train is 10 pulses per second at 400 Hz, inserted into a quiet stretch.

### E. Noise whitening (I-2)

- **Floor level:** for each band, the share of time the signal is above the floor, overall and in loud stretches.
- **Song removed by the floor:** if a band carries song more than half of the time, a median floor sits *on* the song and clipping deletes part of it. Measure the song energy removed, i.e. the energy below the floor in loud stretches, as a share of loud-stretch energy, per band.
- **Alternatives tried:**
  - (a) current: median over 5 min, clip at 0;
  - (b) 10th percentile over 5 min, clip at 0;
  - (c) 10th percentile over 10 min, no clip;
  - (d) no whitening.

  Each is judged by:
  1. the song energy it keeps;
  2. how well loud and quiet stretches separate;
  3. how much slow noise (ship passages) leaks through, judged by the background figure.
- **Listening:** for each example stretch, Griffin–Lim audio is rebuilt from:
  - the full STFT magnitude;
  - the 59-band spectrogram;
  - the whitened spectrogram;

  and saved to `work/` (32 iterations; phase is guessed, so the clips sound rough). You can hear what each step keeps.

### F. Phrase windows, statistics and PCA (I-3 to I-6)

- **Window length vs phrase length:**
  - measure the phrase period with the in-window autocorrelation of I-2, in the same 12 s windows: the share of windows whose dominant repeat is under 12 s;
  - also try 8, 12 and 20 s windows.
- **Compression:** the 240 × 59 patch (14,160 values) becomes 295 values, then 20.
  - Report the PCA variance curve: how many components give 80%, 90% and 95%.
- **Do the summaries keep the "which phrase is this" information?** This is a label-free retrieval test that uses the song's own repetition, which the boundary diagnostics showed is real (period about 13 min).
  - For a sample of windows, take the nearest neighbour under each representation:
    1. the raw I-2 patch, with a shift-tolerant distance: best correlation over ±3 s of time shift;
    2. the 295 descriptors;
    3. PCA-20;
    4. PCA-40.
  - Pairs closer than 12 s are excluded.
  - Score = the share of nearest neighbours that sit 7–30 minutes away. Chance is computed from random pairs.
  - A good representation finds the same phrase in another song.
  - Also report how often PCA-20's 10 nearest neighbours are among the patch's 10 nearest neighbours (neighbourhood overlap).
- **Loudness vs shape:** the share of each PC's weight on level, variability and flux, which all scale with loudness, vs the sweep statistics. Then try a level-normalized variant: divide each window's I-2 patch by its mean before computing the statistics. Run the same retrieval test on it.

## Report (`pipeline_check/report.md`)

1. **Summary table:** one row per check, with result, pass/fail or keep/lose, the number, and the recommendation.
2. **The source limit:** 2 kHz / `df100`, with the long-term spectrum.
3. **One section per stage, I-0 to I-6:**
   - what it is (with a link to `../pipeline.md`);
   - the correctness tests;
   - what it loses;
   - whether that matters for whale song;
   - the figure.
4. **Alternatives:** a table of settings × measures from D, E and F.
5. **Recommendation for Experiment 3:** the pipeline settings to keep or change, each with its reason. If anything changes, `pipeline.md` gets an "Experiment 3 input" section when Experiment 3 is planned. This check does not change Experiment 2's features or results.

## Verification

- **Tests:** add `analysis/test/unit/song_hmm/exp2/test_pipeline_check.py`:
  - the inverse STFT reaches its SNR target on synthetic noise;
  - the band-pooling energy identity holds;
  - a planted tone burst lands in the right frame;
  - a planted 10 Hz pulse train shows a modulation peak at 10 Hz;
  - shift-tolerant patch retrieval finds a planted repeat.
- **Unchanged behaviour:** the existing Experiment 1 and Experiment 2 tests still pass, and with default parameters `features.build` produces the same cache key and arrays as before.
- **Run:** `python -m analysis.song_hmm.exp2_phrase_hmm.pipeline_check.check` runs end to end on the 10 h file. `report.md` has every row filled; the figures render; the Griffin–Lim clips exist in `work/`.
