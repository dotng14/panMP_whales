# Experiment 5 metrics

Plan: [plan.md](plan.md). Pipeline: [pipeline.md](pipeline.md).

## 1. Motif-sequence gate (Approach A only)

Computed on **train** before trusting the HMM.

| Check | How | Target (first guess) |
|---|---|---|
| Phrase repeat | Median normalised edit distance `d1` between consecutive phrases inside long runs vs at JS novelty peaks | Lower inside runs than at peaks |
| Coverage | Fraction of GPL-detected call time covered by kept events | Report; aim ≥ 0.5 on train |
| By eye | Motif ticks under spectrogram on a train slice | Judgement: mostly / no |

If the gate fails, change dictionary / MP / cleanup — not the sticky HMM.

## 2. Jensen–Shannon at boundaries

For a candidate time `t` and window half-width `W` (default **60 s**):

1. Build distribution `P` on `[t−W, t)` and `Q` on `[t, t+W)`.
   - **Approach A:** normalised histogram over motif ids (tokens with `t0` in the interval).
   - **Approach B:** normalised histogram over quantized features, or JS between empirical Gaussians / histogram of PCA scores.
2. `JS(P,Q) = ½ KL(P‖M) + ½ KL(Q‖M)`, `M = ½(P+Q)`.

**Uses:**
- Score each **model-predicted** boundary (higher JS → sharper change).
- **Baseline:** peak-pick the JS novelty curve over time (no HMM).
- Sanity: mean JS at predictions vs mean JS at random times on test.

## 3. Boundary scores vs hand marks

Same matching rule as Exp4: greedy nearest prediction within tolerance τ; each mark hit at most once.

| Score | Definition |
|---|---|
| Recall @ τ | hits / marks at τ ∈ {5, 10, 30} s |
| Precision @ τ | hits / predictions |
| F1 @ 10 s | Harmonic mean at τ = 10 s (**main number** when marks exist) |
| Boundary error | \|pred − mark\| for hits at τ = 30 s; median and p90 |
| False alarms / h | unmatched predictions / hours of marked (or test) audio |

Report **separately** for Approach A, Approach B, and JS-novelty baseline — **test 3 h only**.

## 4. Theme display checks (notebooks)

Not a numeric score, but required:

- Every predicted boundary listed with time.
- For each boundary, **both** adjacent themes shown (`theme_id`, `t0`, `t1`) on the spectrogram.
- No collapsed single-colour strip for the whole test half without explanation in the report.

## 5. Cost

Wall time for dictionary build, MP, feature build, HMM train/decode — per hour of audio where useful.
