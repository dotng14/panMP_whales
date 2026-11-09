# Overlap and speed of the dictionary builds

Experiments of 2026-11-09 on `SOCAL_H_76_230427_014845.x` (200 kHz, LeiData).
Branch `richard/experiments`.

**Results**

1. **Speed: 90–500× faster, same results.** Matching pursuit now keeps each
   atom's scores and rescores only the positions a placement changed, and the
   GPU scorer does one batched FFT instead of one kernel per frequency row.
   With `--overlap free` the builds reproduce the old placements exactly. A
   3-pass pan-build went from 54 min to 35 s, and a randomized-build from
   41 min to 5 s.
2. **Overlap: gone with `--overlap limited`.** Before, about 95% of
   placements overlapped another and each covered column was covered 2.6–3.6
   times. Most of that was the same atom placed again one column over, on the
   leftover of a call. With the rule, no pair breaks it, and each covered
   column is covered 1.0–1.3 times. Units inside phrases are still allowed,
   and pan-build uses that (81 nested pairs).
3. **Coverage: holds for pan-build once `--n-min` is lowered to 3.** Stacking
   had counted each call about 3–4 times, so `--n-min 10` effectively meant
   3 calls. With `limited` and `--n-min 3`, pan-build covers 0.469 of the time
   bins, against 0.488 free (each column covered 4.5 times there) and 0.348
   for the old 3-pass run. randomized-build covers 0.352 ± 0.033.

Recommended: two stages (section 6).

1. `pan-build --overlap limited --max-partial-overlap 0 --max-nested-ratio 0 --n-min 3`:
   no overlap, so every atom is learned on whole calls.
2. `pan-build --overlap limited --max-partial-overlap 0.05 --m-per-outer 10 --n-min 3 --init-dictionary STAGE1`:
   the stage-1 atoms stay fixed, and smaller units nest inside them.

This covers 0.506 of the time bins with 0.701 of the energy explained: more
than free overlap's 0.488, which covered each column 4.5 times. Of its 132
overlapping pairs, 131 are units inside phrases; the other shares one
column. With `--max-partial-overlap 0.1` and 5 seeds a pass the coverage is
about the same (0.512), with 23 partial pairs.


## Setup

- GPL output `result/LeiData/SOCAL_H_76_230427_014845.x.gpl/output`, its first
  10 windows (12.7 min of audio). Each window is 1637 frequency rows × 609
  columns.
- Every run uses `--m-per-outer 5 --refine-iters 1 --workers 2`.
  pan-build adds `--pan-threads 2` and its default `--lengths-s 1,2,3,4`
  (`pca` signal); randomized-build uses `--seed`.
- 32-core server, 2 RTX 3080 Ti, at most 2 cores per run. Wall time comes
  from `/usr/bin/time -v`.
- Measures (`compare.py`):
  - **Coverage:** the fraction of time bins under at least one placement, as
    the builds report it.
  - **Energy explained:** 1 − residual energy ÷ input energy.
  - **Cover per column:** placed columns ÷ distinct covered columns. 1.0 means
    no overlap.
  - **Forbidden pairs:** pairs of placements in one window that the overlap
    rule below would not allow.
  - **Nested pairs:** pairs where one placement lies inside the other.
  - **Similar atom pairs:** pairs with a best shifted cosine above 0.8.

## 1. Speed

### Where the time went

Refinement's matching pursuit took nearly all the run time (6,082 of 6,095 s
in one old pan-build). Two causes:

- **Full rescans.** Each step scored every atom at every position of the
  window again, although a placement at `t` only changes positions within one
  atom width of `t`.
- **A slow GPU path.** `--compute auto` sends scoring calls above 2 million
  operations to the GPU. The CuPy scorer looped over the 1,637 frequency rows
  and launched one small correlation per row: 192 ms per call, against 5.3 ms
  for the CPU (numba) scorer.

### Changes (`code/mp.py`)

- **Cached scores.** The per-window loop keeps every atom's shift scores and
  every width's patch energies. After a placement it rescores only the
  positions whose patch overlaps it. It chooses each step with the same
  function as the full rescan (`_select_rank1`), so placements are identical.
  `test_per_window_inplace_matches_manual_copy_loop` checks that bit for bit.
- **One batched FFT on the GPU.** All rows are transformed at once and summed
  across frequency in the frequency domain: 2.5 ms per full window, matching
  the CPU result to 2e-15 relative error. Slices stay on the CPU (0.3 ms),
  under the `auto` threshold.
- **The rejected step is no longer applied.** The old loop subtracted the
  step that the 1% stopping rule rejected before stopping, so the residual
  held one more placement than the matchings. Placements are unchanged; the
  matching-pursuit residual is now consistent.

One window, the 11 atoms of an earlier pan-build, 30 placements:

| | Full rescan | Cached |
|---|---|---|
| `--compute auto` (GPU for full windows) | 63.8 s (old GPU scorer) | 0.4 s |
| CPU only | 2.1 s | 0.3 s |

### Same results, much faster (3 passes, `--n-min 10`)

| Run | Old code | New code, `--overlap free` | Same output |
|---|---|---|---|
| pan-build, lengths 1–4 s | 3,248 s | 34.6 s (94×) | yes: coverage 0.348, 279 placements, the same overlap counts |
| randomized-build, seed 0 | 2,463 s | 4.8 s (515×) | yes: coverage 0.258, 397 placements, the same overlap counts |

To find the exact match, I replayed pass 1 with the old code (commit
`9863e86`, in a separate worktree) and with the new code. The placements and
amplitudes were identical bit for bit, apart from one order bug of my own
during the work, found by this comparison. `refit_and_deflate` refits
placements one at a time on the shrinking residual. When placements overlap,
the order of the list changes amplitudes by up to 50×. The merge of refined
twins now keeps the list in its original order.

## 2. Overlap

### What it was

On the old 3-pass runs, 93–95% of placements overlapped another placement and
70–86% overlapped one of the same atom. For neighbouring placements of the
same atom, the offset was typically 6–8% of the atom's width (about one
column), and the weaker one had 20–30% of the stronger one's amplitude.
Matching pursuit subtracts the best fit at one position, and the leftover
correlates with the same atom one column over, so it was placed again. The
consequences:

- Placement counts were 3–4 times the number of calls, so `--n-min` pruned
  far less than it seemed.
- Rebuilt atoms averaged the main placement with the shifted ones, which
  blurred them.
- Matching pursuit ran 2–3 times the steps.
- Placements couldn't be read as call occurrences.

### The rule (`--overlap limited`, `mp.OverlapRule`)

Two placements in one window may both be kept when either:

- they overlap by at most a quarter of the narrower width (calls that touch), or
- one lies wholly inside the other and is at most half its width (a unit
  inside a phrase).

Two placements of the same atom have the same width, so they never nest; they
may overlap by a quarter of the width at most. The rule applies inside each
matching-pursuit pass, to the spans of earlier outer passes (passed in as
`fixed_spans`), and to the placements the merge moves. `--overlap free` keeps
the old behaviour and is still the default.

### Results (3 passes, `--n-min 10`)

| | Old pan-build | pan-build, `limited` | Old randomized-build | randomized-build, `limited` (seeds 0 / 1 / 2) |
|---|---|---|---|---|
| Placements overlapping another | 93% | 39% | 95% | 59% / 14% / 19% |
| Cover per column | 2.63 | 1.09 | 3.60 | 1.05 / 1.01 / 1.02 |
| Forbidden pairs | 486 | 0 | 2,054 | 0 |
| Nested pairs (unit in phrase) | 156 | 13 | 377 | 0 |
| Similar atom pairs | 1 | 2 | 11 | 0 |
| Coverage | 0.348 | 0.226 | 0.258 | 0.186 / 0.269 / 0.210 |

The overlap is gone, but so is a third of the coverage. Most seeds were
pruned (3–4 of 5 per pass), and the runs stopped early with `no_survivors`.

## 3. Coverage

Under `limited`, a placement is one call, so `--n-min 10` asks for 10
separate calls of one atom in 12.7 minutes. Free placements counted each call
3–4 times, so `--n-min 10` had meant about 3 calls. Each build was run to its
normal stopping rule (`--max-outer 10 --min-coverage 0.5`), now that runs take
seconds:

| Run | Wall | Coverage | Energy explained | Atoms | Placements | Cover per column | Forbidden pairs | Nested pairs | Similar atom pairs | Stop |
|---|---|---|---|---|---|---|---|---|---|---|
| pan-build free, n_min 10 | 78 s | 0.488 | 0.657 | 27 | 801 | 4.51 | 2,960 | 1,431 | 2 | 10 passes |
| pan-build limited, n_min 10 | 25 s | 0.226 | 0.304 | 5 | 84 | 1.09 | 0 | 13 | 2 | no survivors, pass 3 |
| pan-build limited, n_min 5 | 119 s | 0.384 | 0.593 | 18 | 237 | 1.56 | 0 | 152 | 3 | no survivors, pass 8 |
| **pan-build limited, n_min 3** | **48 s** | **0.469** | **0.577** | **18** | **206** | **1.33** | **0** | **81** | **1** | no survivors, pass 8 |
| randomized-build free, n_min 10 (3 seeds) | 30 s | 0.395 ± 0.001 | 0.594 | 34 | 1,214 | 6.83 | 11,687 | 2,376 | 11 | 10 passes |
| randomized-build limited, n_min 10 | 5 s | 0.221 ± 0.035 | 0.335 | 4 | 89 | 1.02 | 0 | 0 | 0 | no survivors |
| randomized-build limited, n_min 5 | 7 s | 0.310 ± 0.036 | 0.423 | 9 | 137 | 1.04 | 0 | 0 | 0.7 | no survivors |
| randomized-build limited, n_min 3 | 6 s | 0.352 ± 0.033 | 0.463 | 12 | 152 | 1.04 | 0 | 0 | 0.7 | no survivors |

- **Free coverage is padded.** Free pan-build reaches 0.488 by covering each
  column 4.5 times. Limited pan-build with `n_min 3` reaches 0.469 covering
  each column 1.3 times: nearly the same time covered, by 206 placements
  instead of 801.
- **pan-build beats randomized-build under the rule,** by 0.469 against
  0.352 ± 0.033. Its seeds are calls that really repeat; random detection
  crops more often turn out to be calls without 3 separate copies.
- **randomized-build never nests,** because its atoms come from detection
  crops of similar width. pan-build seeds at 1–4 s, so units sit inside
  phrases.
- **No limited run reached 0.5 coverage.** Each stopped when a pass kept no
  atom. A higher `--m-per-outer` or all 30 windows (more copies of each
  call) are the next things to try.

## 4. Less overlap

The two limits are options: `--max-partial-overlap` (default 0.25) and
`--max-nested-ratio` (default 0.5; 0 forbids nesting). All runs here use
`--n-min 3` and stop as in section 3. Two extra measures:

- **Columns > 1:** the share of covered columns that more than one placement
  covers.
- **Overlapping pairs:** pairs of placements that share at least one column.

| Setting | Build | Wall | Coverage | Energy explained | Atoms | Placements | Cover per column | Columns > 1 | Overlapping pairs | Nested pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| free (`n_min` 10) | pan | 78 s | 0.488 | 0.657 | 27 | 801 | 4.51 | 79% | 4,797 | 1,431 |
| | randomized | 30 s | 0.395 ± 0.001 | 0.594 | 34 | 1,214 | 6.83 | 81% | 13,314 | 2,376 |
| partial 0.25, nested 0.5 | pan | 48 s | 0.469 | 0.577 | 18 | 206 | 1.33 | 32% | 132 | 81 |
| | randomized | 6 s | 0.352 ± 0.033 | 0.463 | 12 | 152 | 1.04 | 3.9% | 37 | 0 |
| partial 0.10, nested 0.5 | pan | 105 s | 0.338 | 0.578 | 16 | 143 | 1.16 | 15% | 46 | 40 |
| | randomized | 9 s | 0.372 ± 0.020 | 0.453 | 17 | 153 | 1.01 | 1.0% | 23 | 0 |
| **partial 0, nested 0.5** | **pan** | 139 s | **0.429** | 0.640 | 23 | 247 | 1.32 | 32% | 101 | **101** |
| | randomized | 7 s | 0.300 ± 0.060 | 0.426 | 12 | 118 | 1.00 | 0% | 0 | 0 |
| partial 0, nested 0 (none) | pan | 90 s | 0.309 | 0.485 | 10 | 87 | 1.00 | 0% | 0 | 0 |
| | randomized | 6 s | 0.300 ± 0.060 | 0.426 | 12 | 118 | 1.00 | 0% | 0 | 0 |

- **What overlap is left at partial 0.25:** in pan-build, mostly nesting
  (81 of 132 pairs). The rest are calls that touch by up to a quarter of a
  width. At partial 0, every one of pan-build's 101 overlapping pairs is a
  unit inside a phrase, the case you said is fine. randomized-build has
  nothing left to remove, because it never nests.
- **Coverage:** banning all overlap costs pan-build a third of its coverage
  (0.429 to 0.309). Without nesting it can't keep a 1 s unit inside a 4 s
  phrase, so one of the two atoms loses its placements and gets pruned.
- **Partial 0.10 is not on the way from 0.25 to 0:** pan-build drops to 0.338.
  One less-good seed in pass 2 changes every later pass; on one recording
  that is within the noise. randomized-build's spread across seeds
  (± 0.02–0.06) shows how large the noise is.

Registered in pmp as dictionaries 6 (pan-build, partial 0, nested 0.5),
7 (pan-build, no overlap at all) and 8 (randomized-build seed 0, no overlap
at all).

## 5. Re-placing the no-overlap dictionary with partial overlap

Dictionary #7 (pan-build, no overlap at all, `--n-min 3`) looks the cleanest
in the web viewer: each atom covers a whole call. Its build coverage is 0.309.
I ran `matching-pursuit` with its 10 atoms unchanged, on the same 10 windows,
with no nesting and three partial-overlap limits. Each was run twice: as
`matching-pursuit` runs by default (no placement gate), and with
`--max-relative-distance 0.9`, the gate the builds apply after refinement. The
gate drops placements that leave more than 90% of their patch unexplained.
Each run takes about 2 s.

| Re-placement | Gate | Coverage | Placements | Energy explained | Columns > 1 | Overlapping pairs |
|---|---|---|---|---|---|---|
| (the build itself) | 0.9 | 0.309 | 87 | 0.485 | 0% | 0 |
| no overlap | none | 0.521 | 180 | 0.478 | 0% | 0 |
| no overlap | 0.9 | 0.300 | 89 | 0.482 | 0% | 0 |
| partial 0.10 | none | 0.531 | 187 | 0.490 | 0.5% | 14 |
| partial 0.10 | 0.9 | 0.307 | 91 | 0.489 | 0.3% | 5 |
| partial 0.25 | none | 0.567 | 204 | 0.486 | 3.8% | 53 |
| partial 0.25 | 0.9 | 0.306 | 92 | 0.483 | 1.8% | 13 |

- **The ungated coverage is padding.** Without the gate, coverage rises to
  0.52–0.57 and the placements double. Energy explained doesn't move
  (0.48–0.49), so the extra placements explain almost nothing: each leaves
  more than 90% of its patch unexplained.
- **Partial overlap adds almost nothing on top of no overlap.** With the gate,
  0.25 gives 0.306 against 0.300; the energy explained is the same.
- **The missing coverage is missing atoms, not overlap.** About 0.3 of the
  time is all that these 10 atoms cover well. To cover more, the build needs
  more atoms. Next to try: more passes or seeds per pass, a lower `--n-min`,
  and all 30 windows, so rarer calls reach 3 copies.

Registered in pmp as matching-pursuit dictionaries of #7: 9 (no overlap,
ungated), 10 (partial 0.25, ungated) and 11 (partial 0.25, gated). Backup
first: `catalog.sqlite.bak.20261109T124419_before_mp_on_7`.

## 6. Two stages: no overlap, then nesting and a little partial overlap

Section 5 kept #7's atoms and only placed them again. Here the build
continues from #7 with `--init-dictionary`:

- #7's 10 atoms and 87 placements stay fixed, and are subtracted from the
  spectrogram.
- New passes seed and refine atoms in what is left. Their placements may nest
  in, or touch, the fixed ones under a looser rule.
- The pan seeder's duplicate check compares seeds against #7's atoms too.

Same 10 windows, `--n-min 3 --m-per-outer 5 --max-outer 10 --min-coverage 0.5`;
the wall time includes #7's own 90 s build.

| Run | Wall | Coverage | Energy explained | Atoms | Placements | Columns > 1 | Overlapping pairs | Nested | Partial | Stop |
|---|---|---|---|---|---|---|---|---|---|---|
| #7 (stage 1, no overlap) | 90 s | 0.309 | 0.485 | 10 | 87 | 0% | 0 | 0 | 0 | no survivors |
| #7 → no overlap | +5 s | 0.309 | 0.485 | 10 | 87 | 0% | 0 | 0 | 0 | no survivors, pass 1 |
| #7 → partial 0, nested 0.5 | +12 s | 0.309 | 0.485 | 10 | 87 | 0% | 0 | 0 | 0 | no survivors, pass 1 |
| **#7 → partial 0.10, nested 0.5** | **+111 s** | **0.512** | **0.689** | **36** | 288 | 35% | 171 | 148 | 23 | **coverage**, pass 10 |
| #7 → partial 0.25, nested 0.5 | +174 s | 0.481 | 0.654 | 25 | 224 | 30% | 144 | 78 | 66 | no survivors, pass 8 |
| one stage, partial 0, nested 0.5 | 139 s | 0.429 | 0.640 | 23 | 247 | 32% | 101 | 101 | 0 | no survivors |
| one stage, partial 0.10, nested 0.5 | 105 s | 0.338 | 0.578 | 16 | 143 | 15% | 46 | 40 | 6 | no survivors |
| one stage, partial 0.25, nested 0.5 | 48 s | 0.469 | 0.577 | 18 | 206 | 32% | 132 | 81 | 51 | no survivors |
| free overlap, `n_min` 10 | 78 s | 0.488 | 0.657 | 27 | 801 | 79% | 4,797 | 1,431 | 3,366 | 10 passes |

- **The two stages beat any single stage.** At the same limits (partial
  0.10, nested 0.5), one stage reaches 0.338 and two reach 0.512. Learning the
  long calls first with no overlap gives clean atoms that cover whole calls;
  the later passes only fill in.
- **The new atoms are mostly short:**
  - 13 at 1 s (8 columns), 9 at 2 s, 1 at 3 s and 3 at 4 s;
  - 148 of the 171 overlapping pairs are a unit inside a longer placement;
  - the other 23 share at most 10% of the narrower width.
- **Nesting alone adds nothing.** With partial 0 the first new pass kept no
  atom, so the build stopped there. A unit that sits a column or two past the
  edge of a fixed call needs a little partial overlap.
- **Partial 0.25 finds fewer atoms than 0.10** (25 against 36) and stops
  early. The looser rule lets the first new atoms take more placements each,
  which leaves fewer copies for later seeds to reach `--n-min`.
- **Caveat:** a build stops at its first pass that keeps no atom. The two runs
  that added nothing stopped after one pass of 5 seeds, so more seeds per pass
  might still find something there.

### Just enough partial overlap for nesting

Partial overlap is a fraction of the narrower width, rounded down to whole
columns. With atoms of 8, 16, 24 and 33 columns:

| `--max-partial-overlap` | 8-col pair | 16 | 24 | 33 |
|---|---|---|---|---|
| 0.03 | 0 | 0 | 0 | 0 |
| 0.05 | 0 | 0 | 1 | 1 |
| 0.07 | 0 | 1 | 1 | 2 |
| 0.10 | 0 | 1 | 2 | 3 |

Stage 2 from #7, nested 0.5 (0.03 is the same as 0):

| Partial | Seeds a pass | Wall | Coverage | Energy explained | Atoms | Placements | Nested pairs | Partial pairs (most shared columns) | Stop |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 5 | 76 s | 0.438 | 0.613 | 21 | 171 | 62 | 8 (1) | no survivors, pass 8 |
| 0.07 | 5 | 80 s | 0.434 | 0.658 | 27 | 205 | 89 | 14 (2) | no survivors, pass 8 |
| 0.10 | 5 | 111 s | 0.512 | 0.689 | 36 | 288 | 148 | 23 (2) | coverage, pass 10 |
| **0.05** | **10** | **106 s** | **0.506** | **0.701** | **38** | 282 | **131** | **1 (1)** | **coverage**, pass 8 |

- **The stopping rule held back the tight settings, not the limit.** With 5
  seeds a pass, 0.05 and 0.07 each stopped at pass 8, when one pass kept no
  atom. With 10 seeds, 0.05 keeps finding atoms and reaches the 0.5 target.
- **The result has almost no partial overlap.** One pair, sharing a single
  column; every other overlap is a unit inside a phrase. Its energy explained
  is the highest of any run.
- **New atoms:** 38 − 10 = 28. Registered in pmp as dictionary 14, a
  `refinement` of #7. Backup first:
  `catalog.sqlite.bak.20261109T130926_before_stage2_p05`.

Registered in pmp as dictionaries 12 (partial 0.10) and 13 (partial 0.25),
both `refinement`s of #7. Backup first:
`catalog.sqlite.bak.20261109T125952_before_stage2`.

## 7. The whole recording (30 windows)

The two-stage recipe on all 30 windows of 014845 (37.5 min), with
`--max-outer 20 --min-coverage 0.5`:

- **Stage 1:** no overlap, 5 seeds a pass, `--n-min` 3 or 5.
- **Stage 2:** from each stage 1, nesting 0.5 with partial 0.05 or 0.10, and
  10 seeds a pass.

Stage-2 wall times don't include their stage 1.

| Run | Wall | Coverage | Energy explained | Atoms | Placements | Columns > 1 | Nested | Partial | Windows matched | Stop | Atom widths (8/16/24/33) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stage 1, n_min 3 | 6:31 | 0.391 | 0.403 | 20 | 496 | 0% | 0 | 0 | 29 | no survivors, pass 8 | 7/5/5/3 |
| → partial 0.05 | 4:48 | 0.467 | 0.547 | 48 | 789 | 29% | 323 | 16 | 29 | no survivors, pass 7 | 20/11/9/8 |
| **→ partial 0.10** | 9:43 | **0.484** | **0.579** | **68** | 924 | 38% | 470 | 41 | 30 | no survivors, pass 17 | 33/16/10/9 |
| stage 1, n_min 5 | 1:36 | 0.298 | 0.371 | 11 | 338 | 0% | 0 | 0 | 26 | no survivors, pass 4 | 2/3/3/3 |
| → partial 0.05 | 3:24 | 0.431 | 0.514 | 35 | 746 | 30% | 285 | 11 | 28 | no survivors, pass 8 | 14/7/4/10 |
| → partial 0.10 | 2:42 | 0.414 | 0.520 | 34 | 690 | 33% | 312 | 49 | 27 | no survivors, pass 6 | 12/6/7/9 |

- **Keep `--n-min 3`.** Raising it to 5 for 3× the data stops stage 1 after
  3 passes, with 11 atoms. Calls rarely have 5 separate copies, even in
  37.5 minutes.
- **Two stages still help:** coverage goes from 0.391 to 0.484, and energy
  explained from 0.403 to 0.579.
- **The whole file is covered less than the first 10 windows.** 0.484
  against 0.506–0.512, so the first 10 windows hold more song. All but one
  window now has placements, against 8–10 of the first 10.
- **The best run** (stage 1 then partial 0.10) took 16 min for 37.5 min of
  audio on 2 cores. Each build still ends when one pass keeps no atom.

### New atoms

`new_atoms.py` compares each atom with the best 10-window dictionary (#14): an
atom is new when its best shifted cosine with every #14 atom is below 0.8.
As a baseline, two 10-window builds from the same stage 1 (#12 against #14)
share all 10 stage-1 atoms but only 10 of 26 stage-2 atoms (median best
similarity 0.77). Stage 2 finds different, overlapping sets of shapes for the
same calls.

| Full-file run | Atoms | New (below 0.8) | Median best similarity of new | New placed only in windows 11–30 | #14 atoms matched |
|---|---|---|---|---|---|
| → partial 0.10 | 68 | 57 | 0.57 | 14 | 9 of 38 |
| → partial 0.05 | 48 | 38 | 0.62 | 10 | 8 of 38 |

- **Some atoms are new, and some come from run-to-run variation.** The new
  atoms are less like #14 (median 0.57–0.62) than two builds on the same data
  are like each other (0.77), so some are new shapes. 10–14 of them appear
  only in windows the 10-window build never saw.
- **This matters for merging dictionaries across recordings.** With shapes
  this variable, a duplicate check at 0.9 would keep almost everything, and
  a merged dictionary would grow with every recording. Merging needs a looser
  threshold, or clustering of atoms by the calls they cover.

Registered in pmp (backup first:
`catalog.sqlite.bak.20261109T134523_before_full_file`):

- 15: stage 1, `n_min` 3;
- 16: → partial 0.05;
- 17: → partial 0.10;
- 18: stage 1, `n_min` 5, registered without a label as the parent of 19;
- 19: → partial 0.05, `n_min` 5.

## 8. Five-component atoms

Yoav's `m8x32 refine2` dictionary (pmp #20) stores 5 principal components
per atom. Every run above used 1 (`--n-components` defaults to 1): an atom
is the top uncentered principal component of its placed patches (close to
their average shape), and each placement fits one amplitude. Here the
two-stage recipe runs with `--n-components 5`. Each atom becomes the top 5
uncentered components of its patches, and each placement fits 5
coefficients.

This is our subspace atom, not a port of Yoav's. His subtracts the mean of
the patches first, stores it (`mean_NNN.npy`), and fits mean + 5
variations without subtracting anything from the spectrogram. A fixed mean
can't scale with a louder or quieter call, so it doesn't fit a build that
subtracts each placement.

Same 10 windows, `--n-min 3 --refine-iters 1`, stage 1 with 5 seeds a pass
and stage 2 with 10. Round 1 used `--max-outer 10 --min-coverage 0.5`;
round 2 used `--max-outer 20 --min-coverage 0.7`, so that stage 2 can run.
Stage-2 wall times don't include their stage 1.

| Run | Components | Wall | Coverage | Energy explained | Atoms | Placements | Columns > 1 | Nested pairs | Stop |
|---|---|---|---|---|---|---|---|---|---|
| stage 1, no overlap, target 0.7 | 1 | 16 s | 0.309 | 0.485 | 10 | 87 | 0% | 0 | no survivors, pass 5 |
| → partial 0.05, target 0.7 | 1 | +70 s | **0.560** | 0.716 | 49 | 329 | 77% | 172 | no survivors, pass 15 |
| → partial 0.10, target 0.7 | 1 | +39 s | 0.558 | 0.700 | 44 | 327 | 72% | 152 | no survivors, pass 11 |
| stage 1, no overlap, target 0.5 | 5 | 74 s | 0.490 | 0.850 | 18 | 197 | 0% | 0 | pass limit (10) |
| stage 1, no overlap, 10 seeds a pass | 5 | 110 s | 0.231 | 0.548 | 11 | 74 | 0% | 0 | no survivors, pass 4 |
| **stage 1, no overlap, target 0.7** | 5 | 155 s | **0.710** | **0.895** | 31 | 341 | **0%** | 0 | **coverage**, pass 17 |
| → partial 0.05 or 0.10, target 0.7 | 5 | +6 s | 0.710 | 0.895 | 32 | 346 | 3% | 5 | coverage, pass 1 |

The rank-1 stage 1 at target 0.7 is #7 again: the same 87 placements. It
took 16 s here against 90 s for #7; the cause was not investigated.

- **Five components make stage 1 enough on its own.** With no overlap at
  all, it reaches the 0.7 target: more coverage than any rank-1 run, two
  stages included (0.560), with no overlapping placements. Stage 2 then
  has nothing to do: the target is met after its first pass.
- **It doesn't hit the early stop.** Every pass kept 1–3 atoms, up to
  pass 17. Likely because a 5-dimensional atom also fits a call's varied
  copies, so more of them reach the placement minimum.
- **Energy explained isn't comparable.** Each placement fits 5 numbers
  instead of 1. Component 1 alone explains 0.593 of the energy (0.565 at
  target 0.5); components 2–5 add the rest.
- **The placement minimum is 5, not 3.** A 5-component atom needs at least
  5 patches (`min_keep = max(n_min, n_components)`), so no atom has fewer
  than 5 placements (median 7). Of the rank-1 two-stage run's 49 atoms, 27
  have only 3–4.
- **More near-duplicates.** 7 atom pairs have a shifted cosine above 0.8
  (by component 1), against 2 in the rank-1 runs:
  - 3 pairs are a shorter atom that matches part of a longer one: two 1 s
    atoms in a 4 s atom (0.95, 0.89) and a 2 s atom in a 4 s atom (0.86).
    With no overlap, a unit can't nest in its phrase, so a separate atom
    learns it.
  - 4 are among 1 s atoms: one pair at 0.87, and three atoms that are all
    0.88–0.94 alike.

  The duplicate checks (at seeding and when merging) compare only
  component 1.
- **10 seeds a pass hurts stage 1.** Half the seeds of pass 1 were pruned:
  near-copies of the same calls split their placements, and too few were
  left for each to reach the minimum. The build stopped at pass 4 (0.231).
  Stage 2 from it kept nothing.
- **Slower.** A subspace atom runs MP in `global` mode, without the cached
  per-window loop: 155 s for 17 passes, about 9 s a pass, against 3–5 s
  for rank 1.

Registered in pmp (backup first: `catalog.sqlite.bak.20261109T173238_before_pc5`):

- 21: 5 components, stage 1, target 0.7;
- 22: rank 1, stage 1, target 0.7 (the same placements as #7), the parent of 23;
- 23: rank 1, → partial 0.05, target 0.7.

Folders: `~rivo/whale_runs/014845_overlap/pc5/`
(`s1_m5`, `s1_m10`, `s1_m5_p05`, `s1_m5_p10`, `s1_m10_p05`, `s1_m10_p10`,
`t70_k5_s1`, `t70_k5_p05`, `t70_k5_p10`, `t70_k1_s1`, `t70_k1_p05`,
`t70_k1_p10`). Script: `run_pc5.sh` (round 2 with `ROUND2=1`).

## How this differs from Yoav's limited-overlap learner

`analysis/dictionary_learning_limited_overlap/` (commit `dd8e21f`) also
limits overlap, but by a different rule and on a different signal:

| | `--overlap limited` here | Yoav's learner |
|---|---|---|
| What a placement does | Subtracts its fit from the residual (matching pursuit) | Leaves the spectrogram unchanged; adds 1 to a cover count on each of its columns |
| The limit | Geometric, between each pair of placements: shared columns at most `max_partial` of the narrower width, or nested inside one at least 1 ÷ `max_nested` times as wide | At most `k` placements on any one column (default `k = 1`) |
| Same atom, same call | Never stacked (equal widths never nest) | With `k ≥ 2`, the same peak can be placed up to `k` times, because nothing is subtracted (its plan says so) |
| Unit inside a phrase | Allowed (`max_nested` 0.5) | Only with `k ≥ 2`, and then any two placements may stack too |
| `k = 1` / partial 0 + nested 0 | — | The same constraint: no column is shared |
| Seeds | Random detection crops or the pan matrix profile | Random detection crops of width `w_min`–`w_max` (10–40) |
| Lives in | The shared MP, so every build and `matching-pursuit` can use it | Its own learner and CLI |

So they agree only at the extreme of no shared columns. At that setting,
mine still subtracts each placement before choosing the next; Yoav's scores
every placement on the untouched spectrogram. His `k` can't express "a unit
inside a phrase, but no stacking"; the pairwise rule can.

## Not done, and caveats

- **One recording and 10 windows.** randomized-build has 3 seeds; pan-build is
  deterministic, so it ran once per setting.
- **Coverage and energy only.** The motif-score coverage and specificity of
  `analysis/README.md` and `analysis/evaluation/` were not computed.
- **`--overlap` defaults to `free`, and `--n-min` to 10.** Changing the
  defaults is a separate decision.
- **The HMM.** Building on all 30 windows of 014845, or on the 10-hour
  193500, is the next speed test. Its GPL output wasn't found on this
  server.

## Viewing the runs

Registered in the production pmp catalog
(`production-pmp-webserver/backend/data/catalog.sqlite`). Backups were taken
before each batch: `catalog.sqlite.bak.20261109T113751_before_overlap_exp`
(before 2–5), `catalog.sqlite.bak.20261109T120729_before_less_overlap`
(before 6–8), `catalog.sqlite.bak.20261109T124419_before_mp_on_7` (before 9–11) and
`catalog.sqlite.bak.20261109T125952_before_stage2` (before 12–13) and
`catalog.sqlite.bak.20261109T130926_before_stage2_p05` (before 14) and
`catalog.sqlite.bak.20261109T134523_before_full_file` (before 15–19) and
`catalog.sqlite.bak.20261109T173238_before_pc5` (before 21–23). #20 is Yoav's
`m8x32 refine2`, not part of these experiments.

| Dictionary id | Label | Folder |
|---|---|---|
| 2 | overlap experiment: pan-build, free overlap | `~rivo/whale_runs/014845_overlap/grid/pan_free_n10` |
| 3 | overlap experiment: pan-build, limited overlap, n_min 3 | `.../grid/pan_limited_n3` |
| 4 | overlap experiment: randomized-build seed 0, free overlap | `.../grid/rand_free_n10_s0` |
| 5 | overlap experiment: randomized-build seed 0, limited overlap, n_min 3 | `.../grid/rand_limited_n3_s0` |
| 6 | overlap experiment: pan-build, no partial overlap, units inside phrases allowed, n_min 3 | `~rivo/whale_runs/014845_overlap/less/pan_p00_n50` |
| 7 | overlap experiment: pan-build, no overlap at all, n_min 3 | `.../less/pan_p00_n00` |
| 8 | overlap experiment: randomized-build seed 0, no overlap at all, n_min 3 | `.../less/rand_p00_n00_s0` |
| 9 | overlap experiment: MP of #7, no overlap, ungated | `~rivo/whale_runs/014845_overlap/mp_on_7/none` |
| 10 | overlap experiment: MP of #7, partial overlap 0.25, ungated | `.../mp_on_7/p25` |
| 11 | overlap experiment: MP of #7, partial overlap 0.25, gate 0.9 | `.../mp_on_7/p25_gated` |
| 12 | overlap experiment: #7 then stage 2, partial 0.10 + units in phrases, n_min 3 | `~rivo/whale_runs/014845_overlap/stage2/p10_n50` |
| 13 | overlap experiment: #7 then stage 2, partial 0.25 + units in phrases, n_min 3 | `.../stage2/p25_n50` |
| 14 | overlap experiment: #7 then stage 2, partial 0.05 + units in phrases, 10 seeds/pass, n_min 3 | `.../stage2/p05_n50_m10` |
| 15 | overlap experiment: whole file, stage 1 no overlap, n_min 3 | `~rivo/whale_runs/014845_overlap/full/s1_n3` |
| 16 | overlap experiment: whole file, stage 2 partial 0.05 + units in phrases, n_min 3 | `.../full/s2_n3_p05` |
| 17 | overlap experiment: whole file, stage 2 partial 0.10 + units in phrases, n_min 3 | `.../full/s2_n3_p10` |
| 18 | (no label: stage 1, n_min 5, registered as 19's parent) | `.../full/s1_n5` |
| 19 | overlap experiment: whole file, stage 2 partial 0.05 + units in phrases, n_min 5 | `.../full/s2_n5_p05` |
| 21 | overlap experiment: 5 components, stage 1 no overlap, target 0.7 | `~rivo/whale_runs/014845_overlap/pc5/t70_k5_s1` |
| 22 | overlap experiment: rank 1, stage 1 no overlap, target 0.7 (same result as #7) | `.../pc5/t70_k1_s1` |
| 23 | overlap experiment: rank 1, stage 2 partial 0.05 + units in phrases, target 0.7 | `.../pc5/t70_k1_p05` |

## Reproducing

```bash
# The grid of section 3 (about 10 min on 2 cores); folders and logs in $R
R=~/whale_runs/014845_overlap/grid bash analysis/dict_learning/experiments/overlap_and_speed/run_grid.sh
# The tighter limits of section 4 (about 10 min on 2 cores)
R=~/whale_runs/014845_overlap/less bash analysis/dict_learning/experiments/overlap_and_speed/run_less.sh
# The re-placements of section 5 (under a minute)
bash analysis/dict_learning/experiments/overlap_and_speed/run_mp_on_7.sh
# The two-stage builds of section 6 (about 5 min)
bash analysis/dict_learning/experiments/overlap_and_speed/run_stage2.sh
# The whole recording of section 7 (about 30 min)
bash analysis/dict_learning/experiments/overlap_and_speed/run_full_file.sh
# New atoms of one build against another (section 7)
python analysis/dict_learning/experiments/overlap_and_speed/new_atoms.py ~/whale_runs/014845_overlap/full/s2_n3_p10 \
  ~/whale_runs/014845_overlap/stage2/p05_n50_m10
# The measures, for any build folders (FOLDER.log holds the /usr/bin/time -v output)
python analysis/dict_learning/experiments/overlap_and_speed/compare.py \
  pan_limited_n3=$R/pan_limited_n3 rand_limited_n3_s0=$R/rand_limited_n3_s0
```

The old-code runs of sections 1 and 2 used commit `9863e86`, with the same
options and `--max-outer 3 --min-coverage 0.5 --n-min 10`.
