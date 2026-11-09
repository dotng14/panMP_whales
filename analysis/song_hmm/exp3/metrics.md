# Experiment 3 metrics

Every metric is computed the same way for:
- Experiment 2 and 2b (the previous experiment);
- the four Experiment 3 models M1–M4;
- the CSV itself, where the metric doesn't use the CSV as its reference.

Each is reported for train and test; **test is the headline**.

Code:
- `analysis/song_exp3/evaluate.py`;
- `analysis/song_hmm_phrase/metrics.py`, shared with Experiment 2 (common decoded form, groups B and C).

## Common decoded form

Every model writes, per second of a half:
- `theme[s]`: 0–5, or −1 for background;
- `session[s]`;
- `boundaries`: seconds where a new song starts at a T6 → T1 transition.

Sessions and songs come from these three. A song is **complete** when it starts at a boundary (or at a session start in T1) and ends at a boundary (or at a session end in T6), without touching the edge of the half.

## H: against the hand labels (headline, once labelled)

The source is the two 30-minute test-half clips labelled in Audacity: `reference/labels/*.txt`, parsed by `reference/labels.py`. Only labelled seconds count.

| Metric | Definition |
|---|---|
| Theme accuracy | Share of labelled seconds where the decoded theme equals the hand label. Background counts as a 7th class |
| Balanced accuracy | Mean over the six themes of each theme's recall. Chance = 1/6. T1 and T4 (single-phrase themes) count as much as the long ones |
| Cohen's κ | Agreement beyond chance on the 7 × 7 confusion matrix |
| Per-theme recall | For each hand-labelled theme, the share of its seconds decoded as that theme |
| Song-start P / R / F1 at ±15 s and ±30 s | Decoded boundaries matched one-to-one to hand-marked song starts (`S`) within the tolerance, inside the labelled clips |

Until the label files exist, the report shows these as **pending**.

## P: pair precision (inputs, once judged)

- **Pairs:** for each of 20 test-half query windows, the nearest window (at least 5 minutes away) under each screened input, plus one random window. They are judged blind on the pair page: same phrase type / different / unsure.
- **Pair precision** of an input = the share of its pairs judged "same".
- **Chance** = the share of random pairs judged "same".
- "Unsure" pairs are left out.

## S: screening (inputs)

| Metric | Definition | Caveat |
|---|---|---|
| Song-repeat hit | For 500 loud test windows, the nearest window at least 5 min away lies 1 or 2 song periods away (13 ± 2 or 26 ± 2 min). Chance ≈ 0.05 | A proxy with an unknown ceiling: a correct match 3 or more songs away, or in a song shorter than 11 min, counts as a miss. Useful to *rank* inputs, not as an accuracy |
| Far share | Share of those nearest windows at least 7 minutes away | Close to 1 for every input, so it hardly separates them |
| CSV same-theme | Share of nearest windows with the query's CSV theme | Limited by the CSV's reliability |
| Unit detection | Units per minute in loud and quiet minutes, duration percentiles, share of time covered, number of unit types (BIC) | A good detector finds units in loud minutes and almost none in quiet ones |

## B: agreement with the CSV

Same as Experiment 2: theme accuracy, balanced accuracy and κ (background as a 7th class), per-theme recall, song-boundary P / R / F1 at ±15 / 30 / 60 s against CSV song starts, song counts, sessions.

**Caveat.** The CSV looks generated:
- it covers 100% of the recording with whole-second units;
- 8 songs skip themes;
- its song starts drift a median of 110 s from the audio's own song-to-song alignment.

## C: label-free structure

All measured on one common yardstick: Experiment 2's 295-number phrase descriptors (12 s windows every 2 s, test half), compared with the cosine.

| Metric | Definition |
|---|---|
| Theme coherence | Mean similarity of windows within one decoded theme segment, minus between adjacent segments. Pairs closer than 12 s are left out |
| Theme identity | Similarity of theme k in song n to theme k in song n+1, minus its similarity to the other themes of song n+1 |
| Change-point precision | Share of decoded theme changes within ±10 s of a novelty peak, i.e. a point where the audio changes character |
| Song length median, CV; longest single-theme run | Regularity checks. Payne & McVay: songs last 7–30 min, themes last minutes |

## R: rules (Payne & McVay)

| Metric | Should be |
|---|---|
| Background gaps of 60 s or less inside the half (a session ends only after more than 60 s) | 0 |
| Complete songs with all six themes present | equal to the number of complete songs |

## A and D: fit and synthetic recovery

- **A, fit:** log-likelihood per step, train vs test. It can only be compared within one model, because each model sees different inputs.
- **D, synthetic recovery:** sample from the fitted model, refit from the true state groups, decode, and score against the true path. The score is theme accuracy plus the number of song boundaries found against the number planted. This is the only measure with real ground truth, and it checks the model and inference, not the audio.
- **M4 only, theme-detection AUC:** each theme HMM scores every test-half CSV theme segment, as log-likelihood per frame minus the mean over the six models. One-vs-rest ROC AUC per theme, computed after isolated training and again after embedded training.
