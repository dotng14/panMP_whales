# Experiment 4: theme boundaries from motif sequences (`014845` only)

**Verdict:** On this short file, **neither HMM is good enough yet**. Option 1 either collapses or chatters; option 2 finds the coarse changes but **over-segments** badly. The simple token-novelty baseline (B1) is the only method that looks stable — but its “scores” against provisional marks are circular and must not be read as truth.

Plan: [../plan/plan.md](../plan/plan.md). Code: [../exp4/](../exp4/). Figures: [../exp4/work/figures/014845/](../exp4/work/figures/014845/). Notebooks: [option1](../exp4/notebooks/option1_themes.ipynb), [option2](../exp4/notebooks/option2_themes.ipynb). Gallery: [spectrogram_gallery.html](../exp4/notebooks/spectrogram_gallery.html).

---

## 1. Setup (this run)

| Item | Value |
|---|---|
| Wav | `SOCAL_H_76_230427_014845` → 2 kHz, GPL 30 × 75 s windows |
| Dictionary | `randomized-build` on **train half only** (15 windows) → **14 atoms**, time coverage ≈ 0.26 |
| Split | Train ≈ 0–18.75 min; eval ≈ 18.75–37.5 min |
| Motif sequence (eval) | ~458 tokens, ~135 phrases |
| Reference | **Provisional** B1 novelty peaks on the eval half (6 marks). Not hand marks. |

Numbers below are from `exp4/work/results/014845/summary.json` after `python -m analysis.song_hmm.exp4.run_014845`.

### Eval-half scores (provisional marks — interpret with care)

| Model | F1 @10 s | Recall @10 s | Precision @10 s | Predictions | False alarms / h |
|---|---:|---:|---:|---:|---:|
| **option1_pair** | 0.00 | 0.00 | 0.00 | 0 | 0 |
| **option1_full** | 0.11 | 0.33 | 0.06 | 32 | ~93 |
| **option2** | 0.28 | 1.00 | 0.16 | 37 | ~99 |
| **B1** | 1.00* | 1.00* | 1.00* | 6 | 0* |
| **B2** | 0.10 | 1.00 | 0.05 | 112 | ~340 |

\*B1 matches provisional marks by construction (marks were taken from B1). That row only shows the pipeline is wired; it is **not** an independent win.

---

## 2. Overall: good or bad?

| Model | Judgment | One-line reason |
|---|---|---|
| **option1_pair** | **Bad** | Never fires an A→B boundary on eval; sticks in `Other` after a 1-token blip of `B`. |
| **option1_full** | **Bad / unusable** | 32 boundaries and 40 theme snippets; median theme ≈ 12 s; labels cycle T1→T4 too fast to be real themes. |
| **option2** | **Poor (over-segmenting)** | Hits near every provisional change (±3 s) but invents ~30 extra boundaries; 76% of “themes” last &lt; 30 s (median ≈ 12 s). |
| **B1** | **Useful as a coarse novelty detector** | Sparse, spaced peaks; good for proposing marks / debugging — not an HMM. |
| **B2** | **Bad** | Same as raw `d1` threshold: almost every phrase looks “new” (median `d1` = 1.0). |

**Bottom line:** Motif strings on this file do **not** repeat cleanly enough for either HMM. Option 1 cannot lock onto stable theme emissions; option 2 treats almost every phrase as a change.

---

## 3. Where things go well

1. **Front end runs end-to-end on 014845 alone**  
   2 kHz GPL (127×779), dictionary on the train half (14 atoms), MP → 654 kept events, train/eval sequences non-empty. Notebooks and `work/figures/014845/` show the eval spectrogram with overlays.

2. **Option 2 is *sensitive* to real coarse changes**  
   For each provisional B1 peak, there is an option-2 boundary within ~1.5–3.5 s. So when the motif histogram *does* shift for a while, phrase distance spikes and option 2 notices. That is the right *kind* of signal — it is just not selective enough.

3. **B1 gives a readable coarse timeline**  
   Six well-spaced peaks across the ~19 min eval half (~ every 2–4 min). On the spectrogram slices, those times are the places to look first when hand-marking.

4. **Option 1-full at least *attempts* multi-theme structure**  
   Unlike the earlier 193500-dictionary run (which collapsed to one theme), the 014845-trained full model uses T1–T4. The cycle idea is visible in the label sequence — the failure is duration and chatter, not total silence.

---

## 4. Where things go badly

### 4.1 Motif / phrase quality (root cause)

On the eval half, consecutive phrase edit distance has **median `d1` = 1.0** and **~84% of phrases with `d1` ≥ 0.5**. In plain terms: the “same” sung phrase is rarely written as the same motif string twice. That breaks both options:

- Option 1’s categorical emissions never see a stable token distribution per theme.
- Option 2’s `d1` is high almost everywhere, so `C` (new theme) fires too often.

Likely contributors: only 14 atoms and ~0.26 train coverage; MP still places ~4k raw matches; gap cutting may split phrases inconsistently between halves.

### 4.2 Option 1 pair — under-segmentation

Decoded eval path: a **1-token** `B` at the start of eval, then **`Other` for the entire rest of the half** (~18 min). **Zero** boundaries. The pair topology (Other→A→B→Other) never finds a second theme change. With almost no clean A/B seed on this short train half, Baum–Welch cannot specialise A vs B.

**Where it fails:** everywhere after ~18.8 min — one flat “Other” band on the spectrogram overlay.

### 4.3 Option 1 full — chatter / over-segmentation

- 32 boundaries, 40 theme pieces; **median duration ≈ 12 s**; **75% shorter than 30 s**.
- Labels often go T1→T2→T3→T4→T1… in quick succession, including illegal-looking back-steps (e.g. T3→T2) relative to a strict song cycle.
- F1 @10 s ≈ 0.11 vs provisional marks: a few accidental hits, mostly false alarms (~93 / h).

**Where it looks less bad:** brief stretches where a single theme label holds for ~30–60 s (visible as a longer colour band on some slices). Those are rare.

**Where it fails:** most of the eval half — boundaries every few phrases, no stable theme blocks.

### 4.4 Option 2 — over-segmentation

- 37 boundaries vs ~6 plausible coarse changes → precision @10 s ≈ 0.16, ~99 false alarms / h.
- Median “theme” ≈ 12.5 s; **76% under 30 s**.
- It **does** place a boundary near each B1 peak, then adds many more wherever `d1` spikes (which is most phrases).

**Where it does well:** neighbourhoods of the six B1 times (~20.0, 22.9, 25.3, 28.4, 32.2, 36.3 min wall time) — sensitivity is fine.

**Where it fails:** between those times — dense cyan lines on the spectrogram that do not correspond to sustained theme changes.

### 4.5 B2 — unusable threshold

112 predictions; precision ≈ 0.05. Same root cause as option 2 without the HMM’s mild smoothing.

---

## 5. Comparison to Experiments 1–3 (brief)

Exp 1–3 learned structure from spectrogram features and were scored against a CSV that Exp 4 rejects. Exp 4’s claim was that **better units** would make the HMM easier. On 014845, the units are still too noisy for theme-scale structure: we have swapped “feature ambiguity” for “token instability.” That matches Exp 3’s lesson that units help only if they are real units.

---

## 6. What to do next

1. **Hand-mark** the eval half on the spectrogram (replace provisional B1 marks) so F1 means something.
2. **Improve the motif sequence before touching the HMM:** raise dictionary coverage / atom count on the train half; tighten energy / overlap cleaning; check phrase cuts by eye on 3–5 min of train audio.
3. **Option 2:** raise the effective bar for `C` (e.g. require high `d1` *and* `dmin3`, or a minimum theme length `m`≥3 with stronger self-loops on `R`); or smooth `d1` over a short window before the HMM.
4. **Option 1:** drop pair-v0 on this short file until A/B regions are clearly markable; for `1-full`, constrain transitions to a forward cycle only and put a much higher self-loop prior so themes last minutes, not seconds.
5. Only after the sequence looks repetitive by eye, re-train and re-score.

---

## 7. How to reproduce

```bash
cd /path/to/panMP_whales
python -m analysis.song_hmm.exp4.run_014845 --workers 8
```

Then open the notebooks with kernel **Python (panMP / ensemble)**, or `exp4/notebooks/spectrogram_gallery.html`.
