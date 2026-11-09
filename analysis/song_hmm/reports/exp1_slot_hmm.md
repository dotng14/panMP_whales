# Song-session HMM: model report

Recording `SOCAL_H_76_230426_193500_df100.x.x.wav`: 10 h at 2 kHz. **Train** = first 5 h (0–18000 s), **test** = last 5 h. Features, initialization and Baum–Welch use the train half only. Architecture and choices: [plan.md](../analysis/song_hmm/plan.md).

The CSV reference covers every second of the recording with whole-second units, and some of its songs skip themes. It looks generated, so agreement with it is **agreement with the reference, not accuracy**. The synthetic section is the only score against a true path.

## 1. Fit and generalization

| log-likelihood / s | train | test |
|---|---|---|
| song-session HMM | -19.024 | -19.381 |
| one Gaussian for every second | -25.541 | -26.202 |
| 2-state HMM (background + one song Gaussian) | -21.640 | -20.806 |

Baum–Welch ran 30 iterations. ![](../analysis/song_hmm/figures/loglik.png)

## 2. Session detection

| | train | test |
|---|---|---|
| sessions | 17 | 21 |
| fraction of seconds in a session | 0.944 | 0.921 |
| fraction in breath states (inside sessions) | 0.109 | 0.154 |
| fraction in background | 0.056 | 0.079 |
| CSV-song seconds decoded as song | 0.944 | 0.921 |
| mean P(song) | 0.944 | 0.921 |
| median / max session length (s) | 500 / 8170 | 500 / 2650 |

![](../analysis/song_hmm/figures/timeline.png)

30 minutes of the test half at 1 s resolution:

![](../analysis/song_hmm/figures/timeline_zoom_test.png)

## 3. Song boundaries (T6 → T1)

| | train | test |
|---|---|---|
| decoded songs (incl. partial) | 50 | 47 |
| complete songs (T1 … T6) | 29 | 18 |
| CSV songs fully inside the half | 27 | 25 |
| boundary P / R / F1 at ±5 s vs CSV song starts | 0.06 / 0.07 / 0.07 (2/33 pred, 27 ref) | 0.00 / 0.00 / 0.00 (0/26 pred, 25 ref) |
| boundary P / R / F1 at ±15 s vs CSV song starts | 0.09 / 0.11 / 0.10 (3/33 pred, 27 ref) | 0.08 / 0.08 / 0.08 (2/26 pred, 25 ref) |
| median offset of nearest decoded boundary to each CSV start (s) | 93 | 87 |

![](../analysis/song_hmm/figures/song_lengths.png)

## 4. Theme labeling

| | train | test |
|---|---|---|
| per-second theme agreement with CSV (background counts as wrong) | 0.332 | 0.269 |
| same, inside decoded sessions only | 0.351 | 0.292 |
| mean posterior on the CSV theme | 0.332 | 0.269 |
| always-majority-theme baseline | 0.353 | 0.326 |

![](../analysis/song_hmm/figures/confusion.png)

![](../analysis/song_hmm/figures/theme_seconds.png)

Phrases per complete song:

| theme | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| decoded, train | 6.7 | 12.2 | 5.4 | 7.5 | 4.0 | 9.4 |
| CSV, train | 1.0 | 28.7 | 22.7 | 2.2 | 29.6 | 21.8 |
| decoded, test | 5.1 | 2.2 | 10.1 | 11.2 | 4.8 | 14.3 |
| CSV, test | 1.0 | 55.7 | 18.7 | 2.3 | 31.0 | 21.8 |

## 5. Learned structure

End-of-theme branch (from the last slot of each theme) and breath states. `end` is fixed at 1e-03 and the background self-loop at 0.99972 (a prior, not learned).

| theme | repeat | next | breath | end | breath: stay | resume | next |
|---|---|---|---|---|---|---|---|
| T1 | 0.281 | 0.099 | 0.619 | 1e-03 | 0.711 | 0.276 | 0.013 |
| T2 | 0.769 | 0.087 | 0.143 | 1e-03 | 0.613 | 0.362 | 0.025 |
| T3 | 0.501 | 0.108 | 0.390 | 1e-03 | 0.856 | 0.139 | 0.005 |
| T4 | 0.507 | 0.091 | 0.402 | 1e-03 | 0.648 | 0.325 | 0.027 |
| T5 | 0.638 | 0.245 | 0.115 | 1e-03 | 0.645 | 0.243 | 0.112 |
| T6 | 0.889 | 0.071 | 0.039 | 1e-03 | 0.211 | 0.571 | 0.218 |

Session entry (background → theme): T1 0.42, T2 0.12, T3 0.18, T4 0.24, T5 0.04, T6 0.00.

![](../analysis/song_hmm/figures/slot_lengths.png)

## 6. Synthetic recovery

18000 s sampled from the fitted model (2 sessions, 18 song boundaries), re-initialized from the true slot path the same way the CSV seeds the real fit, refit, and decoded.

| metric | value |
|---|---|
| song vs background, per second | 1.000 |
| theme accuracy (sung seconds) | 1.000 |
| slot accuracy (sung seconds) | 0.983 |
| session start/end F1 at ±5 s | 1.00 (3/3) |
| song boundary F1 at ±5 s | 1.00 (18/18) |
| session start/end F1 at ±15 s | 1.00 (3/3) |
| song boundary F1 at ±15 s | 1.00 (18/18) |
