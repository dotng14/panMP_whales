# dict_learning/

Dictionary learning on GPL whitened spectrogram windows: **Mallat matching
pursuit (MP)**, the **randomized dictionary build** built on it, and the
**pan build**, which takes the randomized build's seeds from the pan matrix
profile instead (the panMP + MP design in
[`../../plans/stage1_panmp_to_mp_design.md`](../../plans/stage1_panmp_to_mp_design.md)).

The other builders (greedy seed trials, `build_dictionary`, `extend_dictionary`,
`match_to_dictionary`, `split_template`, the similarity tools, the DP and
subspace MP solvers, the old catalog persistence) were removed on 2026-10-25;
see [Removed](#removed). Git history keeps them. A new catalog came back on
2026-10-26: `--catalog DB` registers a run in
[`../../database/`](../../database/README.md).

## Layout

```
dict_learning/
├── __main__.py           python -m analysis.dict_learning → code/cli.py
├── code/
│   ├── core.py           Window, Segment, Template, Matching, Dictionary; patch distance; knobs
│   ├── io.py             GPL windows, detection CSVs, detections → patches
│   ├── mp.py             Mallat MP: shift scoring (numba / NumPy / CuPy), loop, workers, counters
│   ├── tools.py          matching_pursuit; rebuild atoms from their patches (mean, PC1, top-k PCA)
│   ├── cli.py            randomized-build, pan-build and matching-pursuit commands
│   └── learn/
│       ├── randomized_dictionary.py   Create → Refine, coverage-gated outer loop (any seeder)
│       ├── pan_seeds.py               pan matrix profile seeds for that loop (pan-build)
│       ├── refine.py                  refine_dictionary_by_averaging (used by learn_from_motifs)
│       └── learn_from_motifs.py       panMP → MP trial, one pair per round (later "mp-greedy" mode)
├── experiments/  notebooks/  scripts/  reports/   (see Removed: many no longer run)
```

Tests live in [`../test/`](../test/README.md): `unit/dict_learning/` and
`integration/`.

## Command line

Every option and its default:
[Command-line tools](../../README.md#command-line-tools) in the root README.
This section covers what the commands read and write.

The commands read a GPL output folder (`spectrogram_npy/` with whitened
windows and JSON sidecars, plus `CSV_*.csv` detections, which `pan-build`
does not need). With `--catalog DB`, every command also registers its output
folder in the SQLite catalog after
writing it (a matching-pursuit run registers its dictionary folder first if
needed). `summary.json` records absolute `gpl_dir` / `dictionary` paths and the
`window_ids` used, so a folder can be registered later with
`python -m database register`.

### `randomized-build`

```bash
# Recipe defaults are enough; override only what you need.
python -m analysis.dict_learning randomized-build \
  --gpl-dir GPL_OUT --out DICT_DIR --seed 0 --verbose
# e.g. override coverage / outer cap:
#   ... --min-coverage 0.5 --max-outer 20 --epsilon 0.5
```

Defaults: `eps-step-fraction=0.01`, `epsilon=0.9`, `m-per-outer=5`,
`max-outer=10`, `min-coverage=0.2`, template widths `10`–`20`,
`n-components=1`, `n-min=10`, `post-refine-max-relative-distance=0.9`,
`refine-iters=1`, `n-windows=all`.

Writes, in the layout the `reports/randomized_*` folders and
`notebooks/randomized_leidata_wav3_014845_results.ipynb` read:

| File | Content |
|---|---|
| `atoms/template_NNN.npy` | One atom, `(F, W)` or `(k, F, W)` |
| `summary.json` | Inputs, every knob, stop reason, coverage, energies, and the atom list (`template_id`, `width`, `n_freq`, `n_components`, `n_patches`, `source_segment_id`, `path`) |
| `matchings.json` | One row per placement: `template_id`, `window_id`, `t_start`, `amplitude`, `energy` (+ `coeffs` for subspace atoms) |
| `outer_logs.jsonl` | One `RandomizedOuterPassLog` per outer pass |
| `loop_stats.json` | MP counters, with `--loop-stats` |

Exit codes: 0 success, 1 no atoms survived, 2 bad input (message on stderr).
`--n-windows` / `--window-offset` pick a slice of windows. `python -m
analysis.dict_learning randomized-build --help` lists every knob with its
default.

### `pan-build`

```bash
python -m analysis.dict_learning pan-build --gpl-dir GPL_OUT --out DICT_DIR \
  --signal pca --lengths-s 1,2,3,4 --m-per-outer 5 --n-min 10 --verbose
```

The randomized build's outer loop with a different seeder
(`learn/pan_seeds.py`). Each pass:

1. **Search signal.** The residual windows, joined in time order, as `pca`
   (projected onto the residual's top `--pca-rows` frequency patterns),
   `full`, `bands` or `energy` (`analysis.matrix_profile.signals.gpl_rows`),
   with every `--decimate` columns averaged. Positions map back to
   `(window_id, t_start)` through the windows' column counts
   (`io.ColumnMap`), not a time grid: a GPL window is not a whole number of
   hops, so joined columns drift off any grid at every boundary.
2. **One pan matrix profile** at the `--lengths-s` lengths.
3. **Pairs, best first,** by the z-score of their distance within their
   length, the lengths taking turns, longest first. Z-scores of different
   lengths don't compare: ranked together on the 014845 recording, the 1 s
   length took 10 of 13 atoms, and 1 s atoms match 2–3 times as often, so
   each pass's refinement took 2–3 times as long. A long seed taken first
   also masks the shorter cuts of its call. Skipped: a pair with a masked copy, a copy that spans two windows
   (per-window MP could never place it), or a flat stretch (two flat windows
   are a perfect z-normalized match).
4. **Copies:** one distance profile (`matrix_profile.distance_profile`, the
   matrix profile's own measure, by FFT) from the pair's first copy. Copies
   are the starts under `d + copy_threshold × (median − d)`, where `d` is the
   pair's distance and the median is the profile's, taken best first and no
   closer together than the profile's exclusion zone. Fewer than
   `--min-copies`: the pair is dropped. A multiple of `d` does not work here:
   per-row z-normalization gives frequency rows the call never reaches full
   weight, so `d` is large even for a perfect pair, and `2d` takes in the
   whole background (55 "copies" of one pair on the synthetic test folder).
5. **Seed:** the copies cut from the full residual (re-aligned within
   `--decimate` columns), all shifted together by up to half a width to
   where their mean has the most energy, averaged, scaled to unit norm. The
   shift matters: z-normalized distance ranks the onset or tail of a loud
   call (half noise, half call) as a strong motif, and without the shift one
   call gave two seeds, one per edge.
6. **Duplicates:** a seed whose best shifted cosine (at least half the
   narrower width overlapping) with a seed or atom already taken reaches
   `--dedupe-corr` is dropped.
7. **Mask** every copy, before and after the shift, at every length.

Steps 3–7 repeat until `--m-per-outer` seeds are taken, every start is used,
or 25 pairs per seed asked for have been tried. Refinement, pruning, merging,
coverage and the stop reasons are the randomized build's, with one addition:

8. **Fold refined twins.** Seeds that differ at step 6 can still refine to the
   same shape (on 014845, four seeds of one pass ended 0.93–0.99 alike). After
   refinement, an atom whose best shifted cosine with an atom that has more
   placements reaches `--dedupe-corr` hands its placements to that atom,
   moved by the shift (`randomized_dictionary.merge_similar_atoms`); the
   atom is rebuilt from all of them and the residual re-deflated from the
   pass's input, before the post-refine gate. `n_merged` in
   `outer_logs.jsonl` counts the folded atoms. `randomized-build` doesn't
   fold (`merge_corr=None`).
 One matrix profile
per pass: masking uses neighbours found before anything was masked, so a call
whose nearest neighbour got masked waits for the next pass, whose profile
runs on the new residual.

**Which signal.** On the synthetic tests `full` seeds each planted motif once
with exactly its copies, even when one motif is twice as frequent and twice
as loud; `pca` sometimes lets a copy of another motif in or misses a motif in
the first pass, and refinement repairs it (all motifs above 0.95 after
refining, in every layout tried). `pca` is the default because the matrix
profile's cost grows with the row count: 8 rows instead of 1637 on the
200 kHz LeiData files.

Writes the `randomized-build` files (with `"command": "pan-build"` and a
`seeder` block in `summary.json`) plus `pan_seeds.jsonl`, one `PanPass` per
outer pass: signal rows and columns, pairs tried, how many were skipped as
masked, too few copies or duplicate, and each pick's pair, distance,
z-score and copies.

### `matching-pursuit`

```bash
python -m analysis.dict_learning matching-pursuit \
  --gpl-dir GPL_OUT --dictionary DICT_DIR --out MP_DIR [--mp-mode global] [--max-repeats N]
```

Places the atoms of a `randomized-build` or `pan-build` output on the windows and writes
`matchings.json` and `summary.json` (energy before and after, stop reason,
time and window coverage).

**How a window is pursued** (`mp._run_mp_loop_one_window`). Each atom's
shift scores and each width's patch energies are computed once. A placement
at `[t, t + W)` changes only the patches that overlap it, so after each step
only those positions are rescored: the same placements as rescoring the
whole window every step, at a fraction of the cost. A step the
`--eps-step-fraction` rule rejects is not applied, so the residual matches
the matchings. The GPU scorer does one batched FFT over all frequency rows.

**Overlap** (`--overlap`, `mp.OverlapRule`). Matching pursuit subtracts the
best fit at one position; the leftover of a call still correlates with the
same atom one column over, so with `free` the same call is placed again and
again (on 014845, each covered column was covered 3–4 times). With `limited`,
two placements in a window may overlap by at most a quarter of the narrower
width, or one may lie wholly inside the other when it is at most half its
width (a unit in a phrase). Both limits are options; with both at 0 no two
placements share a column. The builds apply the rule to every refine pass,
to the spans of earlier outer passes, and to the placements a merge moves.
Measurements: [`experiments/overlap_and_speed/`](experiments/overlap_and_speed/README.md).

**Two stages** (`--init-dictionary DIR`, both builds). The loop starts from
a dictionary folder: its atoms and placements stay fixed, are refitted and
subtracted, and the passes look for new atoms in what is left. Their spans
count as earlier passes' for `--overlap`, and the pan seeder treats the
atoms as already taken. On 014845 the best dictionary so far comes from a
no-overlap build continued with `--max-partial-overlap 0.1`.

### Shared MP knobs

| Flag | Env fallback | Default | Meaning |
|---|---|---|---|
| `--mp-mode` | `DICT_LEARNING_MP_MODE` | `per_window` | `per_window`: independent Mallat per window (parallel; atoms never straddle windows). `global`: one best atom per step over the whole stack. |
| `--overlap` | | `free` | `free` or `limited` (see above). Recorded in `summary.json`. |
| `--max-partial-overlap` | | 0.25 | `limited` only: the fraction of the narrower width two placements may share. |
| `--max-nested-ratio` | | 0.5 | `limited` only: a placement may lie inside another at most this fraction of its width; 0 allows no nesting. |
| `--workers` | `DICT_LEARNING_MP_WORKERS` | all CPUs | Worker processes for `per_window`. `1` runs serially. |
| `--compute` | `DICT_LEARNING_MP_COMPUTE`, then `GPL_COMPUTE` | `auto` | `score_shifts` backend. `auto`: CuPy when it works and the problem is large (`DICT_LEARNING_MP_GPU_MIN_OPS`, default 2e6), else numba, else NumPy. `cuda` forces the GPU when usable. `cpu` never uses it. |
| `--eps-step-fraction` | | 0.01 | Stop when a step removes less than this fraction of the first step's energy. |
| `--max-relative-distance` | | off | Reject placements that leave more than this fraction of the patch energy unexplained; MP falls through to the next-best shift. |
| `--loop-stats` | `DICT_LEARNING_LOOP_STATS=1` | off | Count MP loops, steps and wall time. |
| `--progress` | | off | tqdm bars on stderr. |

### `split`

```bash
python -m analysis.dict_learning split --gpl-dir GPL_OUT --dictionary DICT_DIR \
  --template 3 --out SPLIT_DIR [--matchings MP_DIR] [--seed 0]
```

Replaces one atom by two (`learn/split_template.py`, ported unchanged from
gplrev3):

1. Cut the atom's patches: from its placements in the dictionary's
   `matchings.json`, or with `--matchings` from a `matching-pursuit` run of
   that dictionary.
2. For each of `--n-trials` starts, perturb the atom by a random ±ε (norm
   `--epsilon` × the atom's). Assign each patch to the candidate that explains
   more of its energy, then alternate refitting each side's first principal
   component and reassigning, up to `--max-refine` rounds.
3. Keep the start with the smallest residual error. Accept it when it cuts
   the error of the single atom by at least `--min-relative-reduction` and
   each side has at least `--min-size` patches.

When the split is accepted, the output is a dictionary folder in the build
layout:

- The split atom becomes half `a`, keeping its id; half `b` gets a new id.
- Each half keeps the placements assigned to it.
- Everything else carries over.

`matching-pursuit`, another `split` and the catalog read the folder like a
build's. The catalog records it as an `edit` dictionary of its parent. A
rejected split exits 1 and writes only `summary.json`, which gives the
reduction it found.

## Python API

### Randomized build (`code/learn/randomized_dictionary.py`)

Create → Refine, after
`documentation/Docs/randomized-dictionary.tex` in the old `gplrev3-webserver`
repo:

1. **`create_dictionary`**: sample `m` poorly explained detection patches from
   the residual (keep a crop iff `‖r‖ / ‖x‖ > epsilon`, within the width
   limits) and turn each into a unit-norm rank-1 atom.
2. **`refine_dictionary`**: `iter` rounds of whole-dictionary MP (overlaps
   allowed), drop atoms with fewer than `max(n_min, K)` patches, rebuild each
   survivor as the top-`K` PCA subspace of its patches. Then drop placements
   with unexplained fraction above `post_refine_max_relative_distance` and
   prune again.
3. **`learn_randomized_outer`**: repeat Create → Refine on the deflating
   residual, merging survivors, until time coverage exceeds `min_coverage`.
   Stop reasons: `coverage`, `max_outer`, `no_survivors`, `empty_pool`.

```python
from analysis.dict_learning import learn_randomized_outer

result = learn_randomized_outer(
    windows, detections,
    # defaults: m_per_outer=5, n_min=10, post_refine=0.9, min_coverage=0.2,
    # max_outer=10, refine_iters=1, epsilon=0.9, eps_step_fraction=0.01, …
    rng=0, mp_mode="per_window", n_workers=8,
    progress_callback=lambda log: print(log.outer, log.n_atoms, log.time_coverage_fraction),
)
result.dictionary, result.matchings, result.outer_logs, result.stop_reason
```

`build_randomized_dictionary` runs one Create → Refine. `create_dictionary`
also accepts pre-built `Segment` partitions and a parent-dictionary gate
(`parent`, `alpha`); the CLI uses neither.

`seeder=` replaces Create: any callable
`(residual_stack, m, accumulated_dictionary) -> RandomizedCreateResult`
(the `Seeder` protocol). `partition` may then be `None`.

### Pan seeds (`code/learn/pan_seeds.py`)

```python
from analysis.dict_learning.code.learn.pan_seeds import PanSeedConfig, PanSeeder

seeder = PanSeeder(PanSeedConfig(lengths=(8, 16, 24, 33),     # STFT columns
                                 start_freq=10.0, end_freq=10000.0, signal="pca"))
result = learn_randomized_outer(windows, None, seeder=seeder, m_per_outer=5, n_min=10)
seeder.passes      # one PanPass per outer pass
```

`shape_similarity(a, b)` is the duplicate check's best shifted cosine.

### Mallat MP (`code/tools.py`, `code/mp.py`)

```python
from analysis.dict_learning import matching_pursuit
from analysis.dict_learning.code.core import as_stack

result = matching_pursuit(as_stack(windows), dictionary, eps_step_fraction=0.01,
                          mp_mode="global", max_relative_distance=0.5)
result.loop.matchings, result.loop.residual_stack, result.loop.energies, result.loop.stopped_reason
```

`as_stack` mean-centres each window once. Atoms may be rank-1 `(F, W)` or a
`(k, F, W)` subspace; subspace atoms force global MP without normalization.

### `refine_dictionary_by_averaging` (`code/learn/refine.py`)

Average every atom's patches (or take their leading PC with
`template_method="pca"`), then rerun multi-atom MP, `iter` times.
`pinned_matchings` are refitted in place, deflated before MP, and withheld from
it. Used by `learn_from_motifs`; not part of the baseline CLIs.

## Removed

Removed on 2026-10-25 to make the randomized build the starting point:

- `code/learn/`: `build_dictionary.py`, `extend_dictionary.py`,
  `learn_dictionary.py` (greedy seed → average → MP; its
  `refine_dictionary_by_averaging` moved to `refine.py`), `match_to_dictionary.py`,
  `seed_from_gpl.py` (`detections_to_segments` moved to `io.py`),
  `split_template.py`, `similarity.py`.
- `code/catalog.py` and every `database.catalog` call (the import had been
  resolving to another checkout). Replaced on 2026-10-26 by this repo's own
  [`database/`](../../database/README.md), written through `--catalog`.
- From `mp.py`: candidate and seed-average trial machinery, the DP solver
  (`algorithm="dp"`), the subspace MP loop, and unused helpers.
- From `tools.py`: `build_dictionary_from_detections`, `remove_template`,
  `merge_templates`, `pca_templates`, catalog persistence.
- The old `code/cli.py` (seed, MP, average, edit tools on the catalog).

**These files were left in place but import or call removed code, so expect
them to fail:**

- `GPL_Python/preprocess.py` (and the `backend/preprocess.py` shim that calls
  it): runs the greedy builder and registers it in the catalog.
- `run_build_dictionary.sh`; every script in `scripts/`.
- `experiments/`: `bench_mp.py`, `bench_mp_dp.py`, `compare_grow_k1_k2.py`,
  `run_extend_compare_pca_w2.py`, `run_extend_compare_pca_w2_r20.py`,
  `run_extend_compare_pca_w4_r8.py`, `run_match_to_dictionary_leidata_w2.py`,
  `split_one_template.py`, `split_replace_refine.py`, `subspace_mp_compare.py`.
- `notebooks/`: `build_dictionary_leidata_wav1_results`,
  `build_dictionary_leidata_wav1_w2_mainlike`,
  `build_dictionary_leidata_wav1_w2_mainlike_results`,
  `compare_build_vs_extend_leidata_wav1_w2`,
  `compare_build_vs_extend_leidata_wav1_w2_results`,
  `extend_dictionary_leidata_wav1_chain`, `extend_dictionary_leidata_wav1_w1_cold`,
  `extend_dictionary_leidata_wav1_w1_cold_results`,
  `incremental_build_leidata_wav1_first_window`,
  `patch_template_distance_distribution`, `seed_from_gpl_LeiData` (+ its
  `_setup.py`), `show_saved_templates`, `split_template_patches`,
  `subspace_mp_k1_k2_compare`.

Still working: `notebooks/randomized_leidata_wav3_014845_results.ipynb` and
`notebooks/learn_from_motifs_leidata_014845_10kHz.ipynb`.
