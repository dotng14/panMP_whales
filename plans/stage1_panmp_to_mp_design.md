# panMP → Matching Pursuit: design for discussion

How to seed a Matching Pursuit (MP) dictionary from pan matrix profile (panMP) pairs on a 10 h recording, efficiently and without duplicate motifs. Implementation steps, file names and tests are in [stage1_motif_dictionary.md](stage1_motif_dictionary.md). This document only covers the design.

**Summary.** Run the matrix profile **once** to find candidate pairs. Then, for each candidate, find its other copies with a cheap one-candidate search (an FFT distance profile) and **mask** those copies so the same call can't be picked again. Finally, let all the atoms compete in one joint MP. This keeps the duplicate protection of the current one-pair-per-round loop without rerunning the matrix profile for every atom.

---

## 1. Background: two kinds of search

The rest of the document depends on the difference between these two.

| | Matrix profile | Distance profile |
|---|---|---|
| Question it answers | For **every** stretch of the recording, how close is its nearest match? | For **one** given stretch, how far is it from every position? |
| What it is for | **Discovering** motifs without knowing what to look for | **Finding the copies** of a candidate you already have |
| Cost (n = number of time steps) | About n² (all pairs). The existing `mpx` code is already the fast way. | About n log n, using FFTs (the MASS method) |
| In the code | `matrix_profile.mpx_fast` (1 row), `mpx_channels` (many rows) | Not yet. About 50 lines for a multi-row version. |

A distance profile is one row of the all-pairs comparison. It **can't replace the first matrix profile**. It can replace any later matrix profile whose only job is to find more copies of something already found.

**Scale:** the 10 h whitened spectrogram is about 102 frequency rows × 470,000 time steps. Each row is about 9.8 Hz and each step 76.8 ms, for the 10 kHz files with GPL at 10–1000 Hz.

---

## 2. The duplicate problem

A song's most repeated, most consistent unit has the lowest matrix-profile distances everywhere it occurs. If you take the top-k pairs straight from one matrix profile, many of them can be copies of that same unit.

The matrix profile's exclusion zone doesn't prevent this. It only stops picks from **overlapping in time**; it never compares the **shapes** of two different pairs.

**Example.** Unit U is sung 50 times, at t1…t50. Pick 1 is (t3, t17). Only those two places are blocked. t8's nearest neighbour is t41, another copy of U, so pick 2 is U again, and so on.

**The HMM's current `pan_prototypes` (`HMM/observations.py`) works exactly this way.** Its 12 prototypes can include several copies of the same unit. Duplicate prototypes give nearly identical score columns, which the diagonal-Gaussian HMM counts as independent evidence, and less-repeated units get no prototype. *Not measured yet:* checking the correlation between the 12 saved prototypes in `work/motif_scores.npz` would confirm or rule this out.

**`learn_from_motifs` avoids duplicates by picking one pair per round.** After each round, the new atom's placements are subtracted from the residual and their frames are masked. The next matrix profile can't find that call again.

---

## 3. Why `learn_from_motifs` is slow

Each round of `learn_dictionary_from_motifs` (`dict_learning/code/learn/learn_from_motifs.py`) does the following:

1. **A full matrix profile on the whole residual, using all ~102 frequency rows.** It calls `band_motif_profile` → `mpx_channels` on the stitched whitened spectrogram (the window is named `"band_profile"`, but it is the full spectrogram), once each for 5 s and 10 s.
2. **One pair becomes one atom.**
3. **A full MP refinement** with the whole dictionary over the whole recording (`refine_dictionary_by_averaging`). Earlier atoms' placements are frozen ("pinned").
4. Rebuild the residual and the covered-frames mask, then go back to step 1.

The matrix profile is the dominant cost. It's the most expensive kind (all pairs, all 102 rows), and it runs **2 × (number of atoms)** times:

| Per run | Rough work on 10 h |
|---|---|
| One full-detail matrix profile, one length | ~1 × 10¹³ (about 170× the HMM's current 1-D matrix profile) |
| `learn_from_motifs`, 20 atoms (2 lengths × 20 rounds = 40 matrix profiles) | ~4 × 10¹⁴ |

Other limits visible in the code, which may explain why it also "works not that great":

- **Only two motif lengths (5 s and 10 s),** so 1–4 s units can only be found inside longer windows.
- **Placements are frozen once made,** so an early atom that grabbed the wrong patches is never corrected by later rounds.
- **The acceptance gate (`MAX_RELATIVE_DISTANCE = 0.859`) was tuned by hand on one clip** (014845).
- **It stops at 50% frame coverage, or as soon as one round fails to raise coverage.**

---

## 4. Proposed redesign

### Step 0: choose the search signal

The matrix profile runs on a **search signal** built from the whitened spectrogram. This decides **where** motifs are and **how long** they are. It's a parameter so the options can be compared.

**Atoms, MP and HMM scores always use the full whitened spectrogram**, whatever search signal is chosen.

| Option | How it's built | Keeps | Matrix profile cost (vs today's HMM) | In the code today? |
|---|---|---|---|---|
| **Energy** (1 row) | Sum all ~102 rows per time step, average into 0.1 s bins | Loudness over time only. An up-sweep and a down-sweep of equal loudness look the same. | 1× | Yes: `energy_series_10hz`, used by the HMM |
| **Bands** (5–10 rows) | Sum the rows within each frequency range (10–100, 100–200, … Hz), 0.1 s bins | A rough pitch outline | ~5–10× | Partly: `bands.py`, but from raw audio, 0–500 Hz, and only used by a test |
| **PCA** (~8 rows) | Project the ~102 rows onto the recording's top ~8 frequency patterns | Most of the call shape, with far fewer rows | ~8× | No (new) |
| **Full** (~102 rows) | As is | Everything | ~170× | Yes: what `learn_from_motifs` uses |

### Steps

1. **Run the pan matrix profile once** on the search signal, at each motif length (default 1–10 s). This is the only all-pairs step in a batch.
2. **Pick atoms greedily, without recomputing the matrix profile.** Repeat:
   - Take the best pair that isn't masked, ranked by z-score within its length (as `top_pairs_by_zscore` does). If its nearest neighbour has been masked, skip it.
   - **Seed atom:** the mean of the two copies, cut from the full spectrogram.
   - **Find all its copies** (see "Finding a candidate's copies" below). If it has fewer than `min_repeats` copies, discard it.
   - **Shape check:** compare it to the accepted atoms, sliding to the best time offset, to catch the same call at a different length.
   - **Accept it and mask every copy.** This removes every other pair of that call from the candidate list.
3. **Joint refinement:** MP plus averaging (`refine_dictionary_by_averaging`) with **all** accepted atoms competing on the full spectrogram.
   - The copies found in step 2 feed only the first average. They are **not** frozen, so later passes can correct them.
   - Each atom is rebuilt from its group of copies. Near-duplicates lose their placements to the stronger atom and are pruned or merged.
   - Atoms left with no placements are dropped instead of raising an error.
4. **Another batch only if coverage is short:** subtract the placements, mask the covered frames, recompute the matrix profile on that residual, and repeat steps 2–3.
5. **Output:** every atom that survives (no fixed count), with its placements and the pair(s) it came from. Per-second HMM scores come from sliding each atom over the full spectrogram, which is FFT-type work, not a matrix profile.

### Finding a candidate's copies

**Neither option below is another panMP.** Both slide the one candidate across the recording; they differ in how they measure the match.

| | **A. Distance profile (recommended)** | **B. Single-atom MP** |
|---|---|---|
| Computation | z-normalized distance by FFT (MASS), summed over rows | Raw sliding dot product (`mp.score_shifts`, numba/GPU) |
| Match measure | **The same one the matrix profile used** to rank the pair. Loudness is ignored. | Least-squares fit. Loudness matters. |
| Rule for a copy | Non-overlapping positions closer than a radius set by the pair itself, e.g. r = 2 × the pair's distance | Keep placing while each placement removes enough residual (`eps_step_fraction`) and is close enough to the atom (`max_relative_distance`); each placement is subtracted |
| Cost, full rows, 10 h | ~1 × 10⁹ | ~2 × 10⁹ |
| Code | New, about 50 lines | Exists (`tools.matching_pursuit`, one-atom dictionary) |

Option A is recommended for step 2 because its match measure agrees with how candidates were ranked. Real MP (Option B's machinery) is still used in the joint refinement, step 3.

---

## 5. How the redesign avoids a matrix profile every round

`learn_from_motifs` reruns the matrix profile only to stop the same call from being picked again. The redesign gets that protection differently:

| Job | `learn_from_motifs` | Redesign |
|---|---|---|
| Find candidate pairs | Full matrix profile, every round | Matrix profile **once per batch**, on a cheaper search signal |
| Find the rest of a call's copies | Implicitly, through the next round's matrix profile and MP | **One distance profile** per accepted atom (~10,000× cheaper than a full-detail matrix profile) |
| Stop the same call being picked again | Subtract it, then recompute the matrix profile | **Mask its copies** in the matrix profile already computed |
| Fix atoms that grabbed the wrong patches | Never (placements are frozen) | Joint refinement re-places every atom together |

**Cost for the same 20 atoms on 10 h:**

| | Rough work |
|---|---|
| `learn_from_motifs` (40 full-detail matrix profiles) | ~4 × 10¹⁴ |
| Redesign, PCA search signal, 1–3 batches × 10 lengths | ~0.5–1.5 × 10¹³ for the matrix profiles, plus ~2 × 10¹⁰ for the distance profiles |

That's about **30–80× less** work in the matrix-profile part. The joint MP refinement then adds roughly the same cost per pass in both designs.

**The trade-off.** Masking uses the nearest neighbours that were computed **before** anything was masked. A stretch whose nearest neighbour gets masked loses its pair, even if it has a second-nearest match that isn't masked. Some motifs can therefore be missed within a batch. Step 4's batch recompute is what recovers them, so a few batches replace one matrix profile per atom.

---

## 6. Open questions

1. **Search signal default:** PCA, with energy, bands and full as options? For bands, equal 100 Hz ranges or log-spaced, and over GPL's frequency range or a fixed one?
2. **Copy-finding:** Option A (distance profile) or Option B (single-atom MP)?
3. **Stopping:** what coverage should trigger another batch?
4. **Data and speed:** where is the 193500 GPL output, and what run time is acceptable for 10 h?
