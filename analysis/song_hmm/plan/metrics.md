# Experiment 4 metrics

Three groups of measures:
1. **Input-step checks:** is the motif sequence good enough for an HMM? This is the gate in stage 2 of the plan.
2. **Boundary scores:** do the models find the hand-marked theme changes, on files they were not trained on?
3. **Cost:** how long each step takes.

The plan is in [plan.md](plan.md); the inputs, training and decoding are in [pipeline.md](pipeline.md).

## 1. Input-step checks (motif sequence)

Computed on R0 first, then on every file.

| Check | How it is computed | Meaning | Target (first guess) |
|---|---|---|---|
| **Repeats look alike** | Inside each hand-marked theme, the median `d1` (normalised edit distance between consecutive phrase strings). Compared with `d1` at the marked changes | The same phrase sung again should give nearly the same motif string, and a new theme a different one | median ≤ 0.3 inside themes; ≥ 0.6 at changes |
| **Phrase cutting** | On R0's hand-marked phrase starts: the fraction of marked phrase starts with a cut within ±1 s, and the fraction of cuts that match a mark | Option 2 depends on clean phrases | both ≥ 0.7 |
| **Coverage (time)** | Fraction of GPL-detected call time covered by at least one kept event | How much of the singing gets a motif label. Also the scope check of option 1 | ≥ 0.6 on training files |
| **Coverage (energy)** | 1 − residual energy / original energy over GPL-detected time (from MP) | How much of the sound the atoms explain | reported; compared across knob settings and files |
| **Quiet check** | Kept events per minute in quiet minutes / in singing minutes | Atoms should fit song, not noise | ≤ 0.2 |
| **By eye** | Coloured motif ticks under the spectrogram of R0 | Do the motifs line up with the units you would mark? | your judgement: yes / mostly / no |
| **Seed stability** | Build twice with different seeds. Match atoms between the builds (correlation, Hungarian assignment), relabel, then compute the fraction of events that get the same motif | Do the motifs depend on the random seed? | ≥ 0.7 |
| **Change visible** | B1's novelty curve on R0: is its highest peak within ±15 s of the marked A → B change? | Is the change in the sequence at all? | yes |

## 2. Boundary scores

### Matching predictions to marks

A predicted boundary **hits** a marked change if it falls within the tolerance τ. Each mark can be hit once; when several predictions are close to one mark, the nearest one counts and the others are false alarms. Scores are computed only inside the marked stretches (R0, R1, R2).

| Score | Definition | Meaning |
|---|---|---|
| **Hit rate (recall)** | hits / marks, at τ = 5, 10 and 30 s | How many changes are found |
| **Precision** | hits / predictions, at the same τ | How many predicted changes are real |
| **F1** | 2 · precision · recall / (precision + recall), at τ = 10 s | **Main single number** |
| **Boundary error** | \|predicted − marked\| in seconds for each hit at τ = 30 s; median and 90th percentile | How precise the found boundaries are |
| **False alarms per hour** | predictions that hit nothing / hours of marked audio | How often a change is invented |
| **Phrase-edge offset** | distance from each predicted boundary to the nearest phrase start | Whether boundaries land on phrase edges (they should for 1-pair-v1, 1-full and option 2) |

**Pair stage:** only the A → B marks count as changes. For option 1-pair, the clip contains exactly one change, so only boundary error and hit rate are reported. Option 2 and B1–B3 find every change, so they are scored on all marks in the clips, and separately on the A → B marks alone.

**Full stage:** all marks count.

### Theme labels (option 1-full only)

The decoded theme between two consecutive marked changes is compared with the theme order: does the label move one step forward in the cycle at each change?

**Order accuracy** = the fraction of marked changes where the decoded label moves exactly one step forward.

### Consistency (test T3)

For the 27 April file against the matching 6.23–6.85 h stretch of the main recording:
- **Boundary agreement:** the fraction of boundaries in one that have a boundary within ±5 s in the other (computed both ways and averaged);
- **Motif agreement:** for option 1, which shares the dictionary, the fraction of events with the same motif id within ±0.2 s.

**Target:** ≥ 0.9 on both.

### Out of scope (test T4)

- **Option 1:** is the scope alarm raised (yes / no)? Report the file's coverage next to the training median.
- **Option 2:** the number of boundaries and their times, shown on the spectrogram for you to judge. There are no marks for this file.

### How the results are reported

| Model | Condition | Recall @5/10/30 s | Precision @10 s | F1 @10 s | Median / p90 error (s) | False alarms / h |
|---|---|---|---|---|---|---|
| B1, B2, B3 | T1, T2 | | | | | |
| Exp 3 M1, M3 | T2 | | | | | |
| Option 1 (pair v0, v1; full) | T1, T2 | | | | | |
| Option 2 | T1, T2 | | | | | |

T2 (held-out files) is the result that counts. T1 is shown only to reveal overfitting.

## 3. Cost

| Measure | How | Meaning |
|---|---|---|
| **Build time** | Wall time of `randomized-build` (`--loop-stats`), per window in the build sample | One-time cost of the dictionary |
| **Placement time** | Wall time of `matching-pursuit`, per hour of audio | Cost of motif detection on every new file |
| **HMM time** | Training (all restarts) and decoding time | Expected to be seconds |
| **Sample size vs quality** | Build on 24, 48, 96 and 192 training windows, then repeat the §1 checks and the T2 F1 | How small the build sample can be. For option 2 outside the scope, this is also the cost of a per-file build |
