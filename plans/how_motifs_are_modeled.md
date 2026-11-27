# How motifs are modeled

Terminology (see [`agent_plan1.md`](agent_plan1.md)): an **atom** is the model; a **placement** (detection) is one occurrence in time; **motif** covers either when the distinction is not needed.

## Where they live in the song hierarchy

Payne–McVay: **motif → phrase → theme → song**. Motifs are the short repeated units (~0.5–4 s). Phrases and themes are built above them; the theme HMM consumes placement scores (or later phrase scores), not raw audio.

## Representation

Motifs are modeled on the **GPL whitened spectrogram**, not on the waveform.

- Time is discrete spectrogram columns (hop ≈ 0.12288 s on SOCAL GPL sidecars).
- An **atom** is a short spectrogram template of width \(W\) columns and \(F\) frequency rows.
- Shape:
  - **Rank‑1:** `(F, W)` — a single unit‑norm template (or the top principal component).
  - **Subspace:** `(k, F, W)` plus optional `mean_(F, W)` — mean patch plus top \(k\) principal components of the patches that placed it (typical reference: \(k = 5\)).

So an atom is a low‑dimensional linear model of a local whitened spectrogram patch: either one matched filter, or mean + PCA subspace.

A **placement** records that atom at a start column in a window: `template_id`, `window_id`, `t_start`, amplitude / unexplained energy, and subspace coefficients when \(k > 1\).

## How atoms are learned

Dictionary learning (`analysis/dict_learning`) builds atoms with an outer **Create → Refine** loop (randomized‑build or pan‑build):

1. **Seeds**  
   - Randomized‑build: crops from GPL contour‑island detections (width between `w_min` and `w_max`).  
   - Pan‑build: seeds from pan matrix‑profile correlated pairs (often longer / once‑per‑song structure).

2. **Matching pursuit (Mallat)** places current atoms on the spectrogram windows, choosing starts that maximize **explained fraction** of patch energy (subject to coverage / overlap rules).

3. **Rebuild** each surviving atom from the patches at its placements: average (mean) and top‑`n_components` PCA of the centered patches; orient components to agree with the mean.

4. **Prune** weak placements (unexplained fraction above a threshold `tau`) and atoms with too few placements (`n_min`). Merge near‑duplicates when configured.

5. Repeat outer passes until coverage is high enough or no seeds remain.

Reference Site H dictionary (wav 3 / `014845`): limited overlap, \(k_{\mathrm{cover}}=1\), widths 10–40, 5 PCs, ~35 atoms after refine.

## How placements are scored

For a candidate start, compare the spectrogram patch to the atom:

- **No mean:** explained energy = sum of squared dots with each unit‑norm component.  
- **With mean:** residual after subtracting the mean and projecting onto the components; explained = patch energy − residual.

**Explained fraction** = explained energy / patch energy. Pursuit picks the best (atom, start) under the current overlap limits (`free`, `limited` with `--max-partial-overlap` / `--max-nested-ratio`, or hard non‑overlap).

Absolute time of a placement is recovered from the window’s `start_sample`, hop, and `t_start`.

## Alternate / related motif sources

| Source | Role |
|--------|------|
| Dictionary atoms + MP | Primary short‑motif model used by the theme HMM |
| GPL islands | Seed crops / alternate short detections (0.5–4 s) |
| panMP pairs | Seeds for long or once‑per‑song atoms |
| Phrase model (planned) | Longer (2–20 s) repeating units above motifs |

## Downstream use

Theme models treat a song stretch as a sequence of **placement** labels (which atom hit when). Each theme is a multinomial (Dirichlet‑smoothed) over atom ids; a cyclic HMM over themes decodes song start/end and theme order. Motif quality for agents is scored by **coverage**, **specificity**, and **gap** (best atom vs second‑best on the same stretch)—see [`agent_plan1.md`](agent_plan1.md).

## Code and specs

- Learning / MP: `analysis/dict_learning/` (`randomized-build`, `pan-build`, `matching-pursuit`)
- Step‑level port of the reference build: `analysis/dict_HMM/01_build_dictionary.md`
- Placement → themes/songs: `analysis/dict_HMM/02_place_dictionary.md`, `03_themes_and_songs.md`
