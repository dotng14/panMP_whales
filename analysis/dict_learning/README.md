# dict_learning/

Dictionary learning on GPL whitened spectrogram windows: **Mallat matching
pursuit (MP)** and the **randomized dictionary build** built on it. This is the
baseline for combining panMP with MP (see
[`../../plans/stage1_panmp_to_mp_design.md`](../../plans/stage1_panmp_to_mp_design.md)).

The other builders (greedy seed trials, `build_dictionary`, `extend_dictionary`,
`match_to_dictionary`, `split_template`, the similarity tools, the DP and
subspace MP solvers, catalog persistence) were removed on 2026-10-25; see
[Removed](#removed). Git history keeps them.

## Layout

```
dict_learning/
├── __main__.py           python -m analysis.dict_learning → code/cli.py
├── code/
│   ├── core.py           Window, Segment, Template, Matching, Dictionary; patch distance; knobs
│   ├── io.py             GPL windows, detection CSVs, detections → patches
│   ├── mp.py             Mallat MP: shift scoring (numba / NumPy / CuPy), loop, workers, counters
│   ├── tools.py          matching_pursuit; rebuild atoms from their patches (mean, PC1, top-k PCA)
│   ├── cli.py            randomized-build and matching-pursuit commands
│   └── learn/
│       ├── randomized_dictionary.py   Create → Refine, coverage-gated outer loop
│       ├── refine.py                  refine_dictionary_by_averaging (used by learn_from_motifs)
│       └── learn_from_motifs.py       panMP → MP trial, one pair per round (later "mp-greedy" mode)
├── experiments/  notebooks/  scripts/  reports/   (see Removed: many no longer run)
```

Tests live in [`../test/`](../test/README.md): `unit/dict_learning/` and
`integration/`.

## Command line

Both commands read a GPL output folder (`spectrogram_npy/` with whitened
windows and JSON sidecars, plus `CSV_*.csv` detections). No catalog database.

### `randomized-build`

```bash
python -m analysis.dict_learning randomized-build \
  --gpl-dir GPL_OUT --out DICT_DIR \
  --m-per-outer 5 --n-min 10 --min-coverage 0.5 --max-outer 100 \
  --refine-iters 3 --post-refine-max-relative-distance 0.9 --seed 0 --verbose
```

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

### `matching-pursuit`

```bash
python -m analysis.dict_learning matching-pursuit \
  --gpl-dir GPL_OUT --dictionary DICT_DIR --out MP_DIR [--mp-mode global] [--max-repeats N]
```

Places the atoms of a `randomized-build` output on the windows and writes
`matchings.json` and `summary.json` (energy before and after, stop reason,
time and window coverage).

### Shared MP knobs

| Flag | Env fallback | Default | Meaning |
|---|---|---|---|
| `--mp-mode` | `DICT_LEARNING_MP_MODE` | `per_window` | `per_window`: independent Mallat per window (parallel; atoms never straddle windows). `global`: one best atom per step over the whole stack. |
| `--workers` | `DICT_LEARNING_MP_WORKERS` | all CPUs | Worker processes for `per_window`. `1` runs serially. |
| `--compute` | `DICT_LEARNING_MP_COMPUTE`, then `GPL_COMPUTE` | `auto` | `score_shifts` backend. `auto`: CuPy when it works and the problem is large (`DICT_LEARNING_MP_GPU_MIN_OPS`, default 2e6), else numba, else NumPy. `cuda` forces the GPU when usable. `cpu` never uses it. |
| `--eps-step-fraction` | | 0.01 | Stop when a step removes less than this fraction of the first step's energy. |
| `--max-relative-distance` | | off | Reject placements that leave more than this fraction of the patch energy unexplained; MP falls through to the next-best shift. |
| `--loop-stats` | `DICT_LEARNING_LOOP_STATS=1` | off | Count MP loops, steps and wall time. |
| `--progress` | | off | tqdm bars on stderr. |

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
    m_per_outer=5, n_min=10, post_refine_max_relative_distance=0.9,
    min_coverage=0.5, max_outer=100, refine_iters=3, n_components=1,
    rng=0, mp_mode="per_window", n_workers=8,
    progress_callback=lambda log: print(log.outer, log.n_atoms, log.time_coverage_fraction),
)
result.dictionary, result.matchings, result.outer_logs, result.stop_reason
```

`build_randomized_dictionary` runs one Create → Refine. `create_dictionary`
also accepts pre-built `Segment` partitions and a parent-dictionary gate
(`parent`, `alpha`); the CLI uses neither.

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
- `code/catalog.py` and every `database.catalog` call (this repo has no catalog;
  the import had been resolving to another checkout).
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
