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

Recommended: `pan-build --overlap limited --n-min 3`. To remove every
overlap except units inside phrases, add `--max-partial-overlap 0`. That gives
0.429 coverage (section 4).


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
(before 2–5) and `catalog.sqlite.bak.20261109T120729_before_less_overlap`
(before 6–8).

| Dictionary id | Label | Folder |
|---|---|---|
| 2 | overlap experiment: pan-build, free overlap | `~rivo/whale_runs/014845_overlap/grid/pan_free_n10` |
| 3 | overlap experiment: pan-build, limited overlap, n_min 3 | `.../grid/pan_limited_n3` |
| 4 | overlap experiment: randomized-build seed 0, free overlap | `.../grid/rand_free_n10_s0` |
| 5 | overlap experiment: randomized-build seed 0, limited overlap, n_min 3 | `.../grid/rand_limited_n3_s0` |
| 6 | overlap experiment: pan-build, no partial overlap, units inside phrases allowed, n_min 3 | `~rivo/whale_runs/014845_overlap/less/pan_p00_n50` |
| 7 | overlap experiment: pan-build, no overlap at all, n_min 3 | `.../less/pan_p00_n00` |
| 8 | overlap experiment: randomized-build seed 0, no overlap at all, n_min 3 | `.../less/rand_p00_n00_s0` |

## Reproducing

```bash
# The grid of section 3 (about 10 min on 2 cores); folders and logs in $R
R=~/whale_runs/014845_overlap/grid bash analysis/dict_learning/experiments/overlap_and_speed/run_grid.sh
# The tighter limits of section 4 (about 10 min on 2 cores)
R=~/whale_runs/014845_overlap/less bash analysis/dict_learning/experiments/overlap_and_speed/run_less.sh
# The measures, for any build folders (FOLDER.log holds the /usr/bin/time -v output)
python analysis/dict_learning/experiments/overlap_and_speed/compare.py \
  pan_limited_n3=$R/pan_limited_n3 rand_limited_n3_s0=$R/rand_limited_n3_s0
```

The old-code runs of sections 1 and 2 used commit `9863e86`, with the same
options and `--max-outer 3 --min-coverage 0.5 --n-min 10`.
