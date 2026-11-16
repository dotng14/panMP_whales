# Experiment 5 pipeline

Plan: [plan.md](plan.md). Metrics: [metrics.md](metrics.md).

**Split:** train = windows `[0, 336)` (0–7 h), test = windows `[336, 480)` (7–10 h).  
`WINDOWS_PER_HOUR = 48`. Absolute time for window `i`: ≈ `i * 75` s (from GPL `start_sample`).

---

## Shared: audio and GPL

- Wav: `SOCAL_H_76_230426_193500_df100.x.x.wav` at 2 kHz.
- GPL output already on ensemble under `…/SOCAL_H_76_230426_193500_df100.x.x.gpl/output/` (`spectrogram_npy/` + `CSV_*.csv`).
- Ensemble GPL shape: **127 × 779** per 75 s window (FFT 256 / overlap 192). Exp5 uses this common format for both approaches when reading GPL spectrograms.

Nothing from the test windows is used to fit dictionary atoms, PCA, standardisation, or HMM parameters.

---

## Approach A: motif sequence

### A1. Dictionary (train only)

Preferred Exp5 recipe: **strict no overlap**, high coverage target:

```bash
python -m analysis.song_hmm.exp5.run_pipeline dict --force \
  --min-coverage 0.95 --max-outer 40 --workers 8
# → work/dict/train_w336_s0_noov/  (--disallow-overlap)
```

Or directly:

```bash
python -m analysis.dict_learning randomized-build \
  --gpl-dir GPL_OUT --out DICT_DIR \
  --window-offset 0 --n-windows 336 \
  --seed 0 --loop-stats --verbose \
  --disallow-overlap --min-coverage 0.95 --max-outer 40
```

### A2. Matching pursuit (all windows)

Place with **`--disallow-overlap`** on train and test (hour slices of 48 windows).  
`python -m analysis.song_hmm.exp5.run_pipeline mp --force`

Write `matchings.json` under `work/mp/`. Motif CSVs: `python -m analysis.song_hmm.exp5.run_pipeline export-motifs` → `work/exports/`.

### A3. Motif sequence

For each placement: absolute `t0`/`t1` from window sidecar + column index; drop weak / overlapping events; sort; insert `GAP_S` / `GAP_L` / `BREAK` as in Exp4 sequence rules. Split sessions at `BREAK`.

Optional: cut phrases at gap tokens for the motif-sequence gate (`d1` between consecutive phrases).

### A4. Theme–slot categorical HMM

Topology (same **family** as the fixed soft-HMM / `song_soft_hmm_t6`, but **not** that fixed graph):

- Each **theme** is a left-to-right chain of **slots** (phrase positions).
- Within a theme: only `j → j+1` (**no skip**). If the data needs a different motif order or length, that is modelled as a **different theme**, not a skip edge.
- From the **last** slot of a theme: **repeat** (back to slot 1 of the same theme) or **next** (slot 1 of the following theme). Branch probs learned.
- Observations: motif-id tokens (vocabulary from **train**); each slot has a categorical emission (learned; not pre-labelled `U*`).
- `T` and per-theme slot counts are **unknown a priori** — chosen on train (BIC / held-out). The soft-HMM six-theme unit list is reference only (`figures/architecture/song_soft_hmm_t6_reference.dot`).
- Decode: Viterbi → theme path; **boundary** = move via a **next** edge (theme change).

Diagram: `figures/architecture/approachA_theme_slot_hmm.{dot,png}`.

---

## Approach B: raw signal / Fourier features

### B1. Spectrogram

Either:
- reuse GPL whitened windows stitched in time, or
- STFT on the 2 kHz wav (document FFT, hop, band 10–1000 Hz).

### B2. Features (train stats frozen)

Per frame or per short hop (e.g. 0.5–2 s):
- log energy in frequency bands;
- spectral centroid / bandwidth;
- optional Δ features;
- optional PCA on stacked patches (fit on train only).

Standardise with train mean/std; apply the same to test.

### B3. Theme–slot Gaussian HMM

- **Same topology as A** (theme rows of slots; repeat / next; **no skip** inside a theme).
- Observations: feature vectors `x_t` on a time grid; each slot emits a diagonal Gaussian `N(μ, Σ)`.
- `T` and slot counts unknown a priori; chosen on train as in A.
- Decode: Viterbi; boundary = **next**-theme transition.

Diagram: `figures/architecture/approachB_theme_slot_hmm.{dot,png}`.

---

## Baselines (both approaches)

- **JS novelty:** slide adjacent windows of length W, compute JS between left/right distributions (motif hist for A, feature hist or mean-pooled soft hist for B); peak-pick with minimum separation.
- Optional: spectrogram self-similarity novelty (no motifs).

---

## Outputs layout

```
exp5/work/
  dict/                 # Approach A dictionary
  mp/                   # matching-pursuit outputs
  sequences/            # motif sessions
  features/             # Approach B matrices
  models/A/  models/B/
  results/A/ results/B/
  figures/
```
