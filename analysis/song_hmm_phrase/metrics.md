# Metrics

Every metric is computed the same way for:
- **Experiment 1**, the per-second slot HMM;
- **Experiment 2**, the phrase-window HMM seeded from the audio;
- **Experiment 2b**, the same model seeded from the CSV;
- **the CSV itself**, wherever a metric does not need the CSV as its reference.

Each is reported for the train half and the test half. **The test half is the headline**, because it was never used for fitting. Code: `analysis/song_hmm_phrase/metrics.py`; Experiment 1's group A and D numbers come from `analysis/song_hmm/report.py`.

## Common decoded form

Every labelling is first turned into three per-second arrays for one half (`decode.py`):

- `theme[s]`: 0–5, or −1 for background. A breath counts as the theme it interrupts.
  - Experiment 2 decodes one state per 2 s window; each second takes the state of the window whose centre is nearest.
- `session[s]`: True inside a session, breaths included.
- `boundaries`: the seconds where a new song starts, i.e. a theme 6 → theme 1 transition.
  - For Experiment 2, the time is the centre of the first Theme 1 window.

From these:
- **sessions** are maximal runs of `session`;
- **songs** are pieces of a session cut at `boundaries`.

A song is **complete** when all of these hold:
- it starts at a boundary, or at a session start that is in theme 1;
- it ends at a boundary, or at a session end that is in theme 6;
- it does not touch the edge of the half.

## A. Fit (within one experiment only)

| Metric | Definition | Caveat |
|---|---|---|
| Log-likelihood per step | Forward-algorithm log p(X) of a half, divided by its number of steps (s for Exp 1, windows for Exp 2) | Two experiments see different X, so their numbers **cannot be compared**. Only model vs baselines on the same X, and train vs test, mean anything |
| One-Gaussian baseline | Diagonal Gaussian fit on train, mean log-density per step | A floor: a model with time structure should beat it |
| 2-state HMM baseline | Ergodic 2-state Gaussian HMM (background + one song state) fit on train | Shows what is gained beyond "song vs not song" |

## B. Agreement with the CSV

The CSV is a reference, not ground truth: it looks generated. These metrics measure agreement with that reference.

| Metric | Definition |
|---|---|
| Theme accuracy | Fraction of seconds where the decoded theme equals the CSV theme. Background counts as wrong |
| Majority baseline | Accuracy of always answering the CSV's most common train theme |
| Per-theme recall | For each CSV theme, the fraction of its seconds decoded as that theme |
| Balanced accuracy | Mean of the six per-theme recalls. Chance = 1/6 = 0.167, whatever the theme sizes |
| Cohen's κ | `(p_o − p_e)/(1 − p_e)` on the 7 × 7 confusion matrix (6 themes + background, which the CSV never uses). 0 = chance agreement, 1 = perfect |
| Song-boundary P / R / F1 at ±15, ±30, ±60 s | Greedy one-to-one matching of decoded boundaries to CSV song starts within the tolerance. CSV starts within 15 s of a half edge are left out. Precision = matched / decoded; recall = matched / CSV; F1 = harmonic mean |
| Songs / complete songs | Counts under the definitions above |
| Sessions, fraction in session | Number of sessions; fraction of seconds with `session` True. The CSV marks every second as song |

## C. Label-free audio structure

These metrics ask whether a labelling matches what the audio itself does. They are computed on the phrase-window descriptors `z` (295-D, input I-5 in [pipeline.md](pipeline.md)), using cosine similarity between windows (I-7). The same yardstick is applied to all four labellings, and it uses none of their parameters.

Each second's theme is read at every window centre, giving one label per window. Window pairs closer than 12 s (6 steps) share audio, so they are **excluded** from the similarity averages.

| Metric | Definition | Reads as |
|---|---|---|
| Within-theme similarity | Mean cosine over all pairs of windows in the **same** decoded theme segment (a maximal run of one theme, background excluded) | How uniform a decoded theme is |
| Adjacent-theme similarity | Mean cosine over pairs where one window is in a segment and the other is in the **next** theme segment | How different neighbouring themes are |
| **Theme coherence** | within − adjacent | Higher = themes are uniform inside and distinct from their neighbours, which is what "a run of similar phrases" means |
| **Theme identity** | For each song, the mean descriptor of each theme (≥ 3 windows), unit-normalized. Mean cosine of theme k in song n with theme k in song n+1, **minus** mean cosine with the *other* themes of song n+1 | Higher = a label means the same phrase type from song to song |
| Novelty curve | A Gaussian-tapered checkerboard kernel of ±15 windows (±30 s) slid along the diagonal of the self-similarity matrix | Peaks where the sound changes character |
| Novelty peaks | `scipy.signal.find_peaks` with spacing ≥ 10 windows and prominence ≥ 0.5 × the curve's std | Audio change points, found without any labels |
| **Change-point precision** | Fraction of decoded theme changes (between two non-background themes) within ±5 windows (±10 s) of a novelty peak, one-to-one matching | Higher = the labelling changes theme where the sound changes |
| Change-point recall | Fraction of novelty peaks matched by a theme change | Low for everyone: novelty also peaks inside themes (breaths, noise), so this is secondary |
| Longest run of one theme | Longest stretch of consecutive seconds with the same (non-background) theme | Payne & McVay themes last minutes, not hours; a run of an hour or more means the model is stuck in one theme |
| Song length median, CV | Median and coefficient of variation (std / mean) of complete-song lengths | Payne & McVay songs last 7–30 min. Very low CV can also mean an artificially regular labelling |
| All themes in train IQR | Fraction of complete songs whose six theme durations each fall inside the train interquartile range for that labelling | Regularity of the theme pattern. Small samples make it noisy |

## D. Synthetic recovery

1. Sample a sequence from the fitted model: 18,000 s for Experiment 1, 8,995 windows for Experiment 2.
2. Re-seed from the true state path, the same way the real fit is seeded.
3. Run Baum–Welch, then Viterbi.
4. Score the result against the **true** path:
   - theme accuracy on sung steps;
   - song vs background accuracy;
   - session-edge F1 and song-boundary F1 (Experiment 1 at ±5 s and ±15 s, Experiment 2 at ±10 s, i.e. ±5 windows).

This is the only evaluation with real ground truth. It measures what the model and the decoding can do when the data follow the model's own assumptions. A high score here and a low score on real data means the input or the assumptions are the problem, not the inference.
