# Phrase-type sequences + random-position JS null

Follow-up to Exp6: (1) evaluate JS against **random times**, (2) build an **A→B→C…** phrase sequence that keeps motif times.

## 1. JS at random positions

Approach A predicted boundaries: mean JS **0.546**.

| Null | Value |
|---|---|
| One matched random draw (n=20) | mean JS **0.505** |
| 50 random trials (mean of means) | **0.487** ± 0.029 |
| 90th percentile of trial means | **0.528** |

`looks_like_random: true` (pred / random ≈ **1.08**). Pred is only slightly above the random p90. So the earlier “JS beats random” story is **weak** — random positions already look similar.

Code: `evaluate.random_position_js_null` + `summarize_js(..., looks_like_random)`. Wired into Exp5 stage 3.

## 2. Motif → phrase-type sequence (with times)

Idea: cut motif stream into units, cluster units → labels **A, B, C, …**, keep `(t0, t1, motifs)` per unit so the HMM can sit on `A B C A B C` instead of raw template ids.

```bash
python -m analysis.song_hmm.exp5.run_phrase_sequences
```

Artifacts: `exp5/work/phrase_sequences/`

| File | Role |
|---|---|
| `phrase_units_period.csv` | `file_id,label,t0,t1,motifs` |
| `phrase_sequences_period.json` | full timed sequences |
| `report.json` | stats + JS null |

### What we got (period cut, edit≤0.34)

| Metric | Value |
|---|---|
| Units | 4839 |
| Phrase types | **487** (almost unique) |
| Frac consecutive same label | ~0.02 |
| Local ABCABC (period-3) | **0** |

So the machinery works (labels + times are tracked), but with the current 164-id alphabet the stream is **not** clean `A→B→C A→B→C` yet — nearly every chunk gets a new letter at strict clustering.

Looser clustering helps a little (still weak ABCABC):

| cut | max_edit | #types | frac same consecutive | frac local period-3 |
|---|---:|---:|---:|---:|
| period | 0.34 | 487 | 0.02 | 0 |
| period | 0.50–0.80 | 70 | 0.11 | ~0.006 |
| gap | 0.80 | 90 | 0.11 | ~0.004 |

Example (period, edit 0.66): `A B C A A D A A E F G H H F E D …` — some repeats, not stable themes.

## Next (recommended)

1. Coarsen motifs (merge similar templates) **then** rebuild phrase types until period-3 repeats appear by eye and in `frac_local_period3_repeat`.
2. Train theme–slot HMM on **phrase labels** (A,B,C,…), not raw motif ids; decode still uses `t0/t1` from the CSV.
3. Always report JS vs the multi-trial random null (`looks_like_random`).
