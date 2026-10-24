# dict_learning/

Dictionary-learning tools on GPL whitened spectrogram windows. Design: [`../../documentation/Docs/analysis.md`](../../documentation/Docs/analysis.md) §3b.

## Table of contents

- [Layout](#layout)
- [Core library (`code/`)](#core-library-code)
- [Experiments / scripts / notebooks](#experiments--scripts--notebooks)
  - [Compatibility contract (subspace)](#compatibility-contract-subspace)
- [Matching Pursuit performance](#matching-pursuit-performance)
  - [Run efficiently (recommended)](#run-efficiently-recommended)
  - [Low-level knobs](#low-level-knobs)
- [`seed_from_GPL` (all candidates → min residual among repeat-qualified)](#seed_from_gpl-all-candidates--min-residual-among-repeat-qualified)
  - [Python API](#python-api)
  - [`Seed` getters](#seed-getters)
- [Multi-round learning (`code/learn/learn_dictionary.py`)](#multi-round-learning-codelearnlearn_dictionarypy)
  - [`learn_from_gpl` (seed only)](#learn_from_gpl-seed-only)
  - [`build_dictionary` (seed + refine → catalog)](#build_dictionary-seed--refine--catalog)
  - [`extend_dictionary` (coverage-gated stage → catalog)](#extend_dictionary-coverage-gated-stage--catalog)
  - [`match_to_dictionary` (matchings + widths → dictionary → refine)](#match_to_dictionary-matchings--widths--dictionary--refine)
  - [`split_template` (1→2 bipartition of one atom’s patches)](#split_template-12-bipartition-of-one-atoms-patches)
  - [`calc_divergences` (asymmetric atom×atom residual divergences)](#calc_divergences-asymmetric-atomatom-residual-divergences)
  - [`calc_template_similarity` (response correlation on a fixed matching)](#calc_template_similarity-response-correlation-on-a-fixed-matching)
  - [`calc_template_group_similarity` (group-mean responses on labeled matchings)](#calc_template_group_similarity-group-mean-responses-on-labeled-matchings)
  - [`algorithm="dp"` (exact non-overlapping matching pursuit)](#algorithmdp-exact-non-overlapping-matching-pursuit)
  - [`learn_averaged_templates_from_gpl` (seed → average → MP)](#learn_averaged_templates_from_gpl-seed--average--mp)
  - [`refine_dictionary_by_averaging` (average every atom → multi-atom MP)](#refine_dictionary_by_averaging-average-every-atom--multi-atom-mp)
  - [`LearnedDictionary` getters](#learneddictionary-getters)
- [Tests](#tests)
- [Types](#types)
  - [`Window`](#window-codecoreobjectspy)
  - [`Segment`](#segment-codecoreobjectspy)
  - [`Template`](#template-codecoreobjectspy)
  - [`Matching`](#matching-codecoreobjectspy)
  - [`Dictionary`](#dictionary-codecoreobjectspy)
  - [`Detection`](#detection-codeioio_gplpy)
  - [`Seed`](#seed-codelearnseed_from_gplpy)
  - [`LearnedDictionary`](#learneddictionary-codelearnlearn_dictionarypy)
  - [`DictionaryRefinementResult`](#dictionaryrefinementresult-codelearnlearn_dictionarypy)
  - [`MpStepResult`](#mpstepresult-codempmp_steppy)
  - [`MpLoopResult`](#mploopresult-codempmp_steppy)

## Layout

```
dict_learning/
├── code/                # library implementation
│   ├── core.py          # windows, atoms, distance, defaults
│   ├── io.py            # GPL window / CSV loaders and spectrogram stitch
│   ├── mp.py            # matching pursuit
│   ├── learn/           # seed, build, extend, motifs, similarity, split
│   ├── catalog.py       # repeats payload for the frontend
│   ├── tools.py         # elementary tools A–E
│   └── cli.py
├── experiments/         # LeiData harnesses (compare grow, subspace, match-to-dict, split, bench_mp)
├── scripts/             # bash runners for LeiData experiments
├── notebooks/           # visualization / exploration notebooks
├── tests/
├── __init__.py          # public API re-exports from code/
└── __main__.py          # → code.cli
```

## Core library (`code/`)

| File | Role |
|---|---|
| `code/core.py` | `Window`, `Segment`, `Template`, `Matching`, `Dictionary`, coverage, patch distance, and shared knobs (`defaults`). |
| `code/io.py` | Load GPL windows (`.npy` + `.json`) and detection CSVs; stitch the whitened spectrogram. |
| `code/mp.py` | Matching pursuit: score every shift, greedy or exact placement, subtract the atom. |
| `code/tools.py` | Elementary tools **A–E**: `matching_pursuit`, `average_templates` / `pca_templates` / `pca_basis`, grow/edit (+ catalog persist for `k=1`). |
| `code/learn/seed_from_gpl.py` | Repeat-qualified single-template selection (`Seed.from_gpl` / `Seed.from_paths`). |
| `code/learn/learn_dictionary.py` | Orchestrator: seed→average→MP rounds; optional `n_components>1` subspace rebuild. |
| `code/learn/extend_dictionary.py` | Single stage: peel parent → while coverage < 0.9: +1 atom → refine; optional pinned matchings. |
| `code/learn/match_to_dictionary.py` | Build dictionary from labeled matchings + explicit widths, then optional average→MP refine. |
| `code/learn/split_template.py` | Randomized 1→2 rank-1 split of one atom’s patches. |
| `code/learn/similarity.py` | Compare atoms: residual-energy divergence, response correlation, and group-mean similarity. |
| `code/learn/build_dictionary.py` | Catalog WAV → one-shot build → register (`k=1`) or manifest-only (`k>1`). |
| `code/learn/learn_from_motifs.py` | Grow a dictionary from matrix-profile motifs until half the frames are covered. |
| `code/learn/randomized_dictionary.py` | Sample unexplained patches as atoms and refine until coverage. |
| `run_build_dictionary.sh` | Generic efficient wrapper (threads + CPU Numba defaults). LeiData: `scripts/run_build_dictionary_leidata_wav1.sh`. |
| `code/catalog.py` | Frontend repeats JSON from catalog + panel.bin. |
| `code/cli.py` / package `__main__.py` | CLI: `matching-pursuit`, `average-templates`, `match-to-dictionary`, `build-from-detections`, `remove-template`, `merge-templates`. |
| `tests/` | Unit tests for seeding, MP stopping rules, tools, subspace, extend, split, relative-distance. |

## Experiments / scripts / notebooks

| Path | Role |
|---|---|
| `experiments/compare_grow_k1_k2.py` | Side-by-side grow for two `n_components` values; writes `compare_summary.json`. |
| `experiments/subspace_mp_compare.py` | Post-hoc k=1 vs subspace MP on a fixed built dictionary. |
| `experiments/run_match_to_dictionary_leidata_w2.py` | Labeled re-estimate harness. |
| `experiments/split_one_template.py` | Catalog or synthetic 1→2 split. |
| `experiments/split_replace_refine.py` | Split atom → remap → pinned refine → register `source=edit`. |
| `experiments/bench_mp.py` | Microbenchmarks for `score_shifts` / MP loops. |
| `scripts/run_*.sh` | LeiData wav1 bash runners (build, chained extend, w1→w2 K=1, compare). |
| `notebooks/` | Overlays (subspace compare, build/extend viewers, split, relative-distance). |

Run experiments via `python -m analysis.dict_learning.experiments.<name>`.

Design docs: [`documentation/Docs/dictionary-operations.md`](../../documentation/Docs/dictionary-operations.md).

### Compatibility contract (subspace)

* `Matching(amplitude=a)` ↔ `coeffs=[a]`; `amplitude` is signed for `k=1`, else `‖coeffs‖₂`.
* `Template.data` is `(F,W)` or `(k,F,W)`; use `.basis` / `.as_rank1`.
* `n_components=1` (default) keeps catalog registration; `n_components>1` writes a subspace manifest and skips catalog dictionary rows.
* Math: [`documentation/Docs/subspace-mp-math.md`](../../documentation/Docs/subspace-mp-math.md). Plan: [`documentation/Docs/dict-learning-subspace-refactor-plan.md`](../../documentation/Docs/dict-learning-subspace-refactor-plan.md).

## Matching Pursuit performance

### Run efficiently (recommended)

Use the generic wrapper (any indexed WAV). It sets same-logic speed defaults
unless you already exported overrides:

```bash
bash analysis/dict_learning/run_build_dictionary.sh \
  --db /path/to/dict_learning_catalog.sqlite \
  --wav /path/to/clip.wav \
  --out-dir /path/to/out \
  --name my_run \
  --n-windows 4 --n-rounds 20 --eps-step-fraction 0.5 \
  --min-repeats 3 --min-template-width 12 --max-template-width 24 \
  --template-method pca --refine-iters 0
```

LeiData wav1 shortcut (same recipe): `run_build_dictionary_leidata_wav1.sh`.

| Env | Recommended | Meaning |
|---|---|---|
| `DICT_LEARNING_MP_CANDIDATE_POOL` | `threads` (script default) | Candidate trials share the residual; avoids per-round process spawn / pickle. Set `process` to A/B. |
| `DICT_LEARNING_MP_COMPUTE` | `cpu` (script default) | Numba `score_shifts`. One-shot CuPy is often slower for typical patch sizes; use `auto`/`cuda` only after `bench_mp`. |
| `DICT_LEARNING_MP_WORKERS` | `$(nproc)` or try **8–16** | Candidate + nested window parallelism. On a 32-CPU host, 8 was slightly faster than 32 for the 4-window recipe. |
| `DICT_LEARNING_LOOP_STATS` | `1` (optional) | Print/write loop counters for profiling. |

Do **not** pass `--template-out-dir` unless you need per-round dumps (extra I/O). Round-log SQLite commits are batched until the run finishes.

Details / measurements: [`documentation/Docs/mp-performance.md`](../../documentation/Docs/mp-performance.md), [`documentation/plans/13-same-logic-mp-speedups.md`](../../documentation/plans/13-same-logic-mp-speedups.md).

### Low-level knobs

Defaults (override with CLI `--mp-mode` / `--mp-workers` / `--mp-compute`, or env
`DICT_LEARNING_MP_MODE` / `DICT_LEARNING_MP_WORKERS` / `DICT_LEARNING_MP_COMPUTE`;
`GPL_COMPUTE=cuda` also enables CUDA preference):

| Knob | Default | Meaning |
|---|---|---|
| `mp_mode` | `per_window` | Independent Mallat per window (parallelizable). Atoms never straddle window boundaries. Each window uses its own first-step ΔE for `eps_step_fraction`. |
| `mp_mode=global` | opt-in | Classic stack-wide Mallat: one global best atom per step; shared first-step ΔE. |
| `mp_workers` | **all CPUs** (`sched_getaffinity` / `cpu_count`) | Parallel GPL **candidate trials** and parallel windows (threads when Numba/CuPy). Nested work splits CPUs across outer×inner. Use `1` to force serial. |
| `mp_compute` | `auto` | `score_shifts` backend: CUDA via CuPy when kernels work **and** the patch is large enough (`DICT_LEARNING_MP_GPU_MIN_OPS`, default 2e6 fused ops); else Numba/NumPy on CPU. `cuda` forces GPU when usable; `cpu` disables GPU. |
| `DICT_LEARNING_MP_CANDIDATE_POOL` | `threads` | `threads` = ThreadPool (shared residual); `process` = spawn ProcessPool + initializer. |

`score_shifts` prefers CuPy (when beneficial), then Numba (`@njit(nogil=True)`), then NumPy. Candidate trials default to a **thread** pool (`DICT_LEARNING_MP_CANDIDATE_POOL=threads`) with shared residual; set `process` to A/B. Per-window MP uses threads when Numba/CuPy is available. Microbench: `python -m analysis.dict_learning.experiments.bench_mp`.

A C/Cython rewrite of the correlate kernel is **deferred**: after Numba +
`per_window` / candidate parallelism (+ optional CuPy), revisit C only if
LeiData-scale profiles still pinpoint the kernel.

## `seed_from_GPL` (all candidates → min residual among repeat-qualified)

**First step:** mean-center each raw window **once** (subtract its global mean) so the signal has zero mean. This is always done and is not optional. When `stack` is already a residual `dict[window_id → array]` it is used as-is and never re-centered.

Then trial **every** GPL unit patch with a single-template Mallat loop (on that same residual, no further centering), discard trials with fewer than `min_repeats` matchings, and select the qualifying candidate with the **minimum** residual energy.

```bash
# Via the shared CLI:
python3 -m analysis.dict_learning.code.cli seed-from-gpl \
  --npy-dir /path/to/spectrogram_npy \
  --gpl-output-dir /path/to/gpl_outputs \
  --eps-step-fraction 0.5 \
  --min-repeats 3 \
  --out seed_out

# Or run the module directly (same flags, plus --no-normalize):
python3 -m analysis.dict_learning.code.learn.seed_from_gpl \
  --npy-dir /path/to/spectrogram_npy \
  --gpl-output-dir /path/to/gpl_outputs \
  --eps-step-fraction 0.5 --min-repeats 3 --out seed_out
```

### Python API

```python
from analysis.dict_learning import Seed
from analysis.dict_learning.code.io import load_window, load_detections_csv

window = load_window("…/spwhiten_….npy")  # sibling .json required
detections = load_detections_csv("…/CSV_….csv")

seed = Seed.from_gpl(
    [window],              # or dict[window_id → array]; with a dict, pass windows_for_units=…
    detections,            # or pre-built list[Segment] (e.g. width-filtered)
    eps_step_fraction=0.5, # stop Mallat when ΔE_step < this × first ΔE_step
    min_repeats=3,         # require at least this many matchings
)

# Same algorithm from paths (what the CLI wraps)
seed = Seed.from_paths(
    npy_dir="…/spectrogram_npy",   # or window_paths=[…]
    gpl_output_dir="…/output",     # or csv_paths=[…]
    min_repeats=3,
)
seed.write("seed_out")  # template.npy + seed_manifest.json
```

`seed_from_gpl(...)` and `seed_from_gpl_paths(...)` remain as aliases that return a `Seed`.

### `Seed` getters

| Method | Returns | Meaning |
|---|---|---|
| `is_selected()` | `bool` | Whether a template was chosen. |
| `get_template()` | `Template \| None` | Winning patch as a dictionary atom. |
| `get_matchings()` | `list[Matching]` | MP activations from the winning trial. |
| `get_residual_after()` | `dict[str, ndarray] \| None` | Residual spectrograms after the winning trial (`window_id` → `(F, T)`). |
| `get_energy_before()` | `float` | Total Frobenius energy before any trial: $\sum_w \lVert R_w \rVert_F^2$. |
| `get_energy_after()` | `float` | Total Frobenius energy after the winning trial. |
| `get_delta_energy_total()` | `float` | `energy_before - energy_after`. |
| `get_mp_energies()` | `list[float]` | Stack energy after each MP step of the **winning** trial (index 0 = before any atom). |
| `get_mp_delta_energies()` | `list[float]` | Per-iteration ΔE for that trial: `E[k-1] - E[k]`. |
| `get_n_candidates_tried()` | `int` | How many candidates were trialled. |
| `get_selected_segment_id()` | `str \| None` | ID of the winning `Segment`. |
| `get_candidates()` | `list[Segment]` | All patches considered (energy-sorted for display). |
| `write(out_dir)` | `Path` | Write `template.npy` + `seed_manifest.json` (+ residuals). |

**`get_template()` (`Template`):** `template_id`, `data` `(F, W)`, `source_segment_id`; `.width` is the time-bin width.

**Each `Matching` from `get_matchings()`:** `template_id`, `window_id`, `t_start`, `amplitude`.

**Each `Segment` from `get_candidates()`:** `segment_id`, `data`, `window_id`, `t_start`/`t_end`, global sample bounds, `detection_index`, `boundary_cross_flag`; `.energy` is $\lVert U \rVert_F^2$.

If nothing is selected, `get_template()`, `get_matchings()`, and `get_residual_after()` return empty/`None`, but `get_candidates()` and `get_energy_before()` are still populated.

## Multi-round learning (`code/learn/learn_dictionary.py`)

Two related APIs build a dictionary by residual deflation. Both return a `LearnedDictionary` whose per-round results are ordinary `Seed` objects (same getters as above).

### `learn_from_gpl` (seed only)

Each round runs `Seed.from_gpl` on the current residual and keeps that seed's residual for the next round. No averaging step.

```python
from analysis.dict_learning import learn_from_gpl

result = learn_from_gpl(
    windows,
    detections,                 # or list[Segment]
    max_templates=5,
    min_seed_delta_fraction=0.1,  # stop when round ΔE < this × first-round ΔE
    eps_step_fraction=0.5,
    min_repeats=1,
)
```

Path variant: `learn_from_gpl_paths(npy_dir=…, gpl_output_dir=…, …)`.

### `build_dictionary` (seed + refine → catalog)

End-to-end CLI for one WAV already indexed in the SQLite catalog: load its GPL
segmentation + first `n_windows`, run `learn_averaged_templates_from_gpl`,
optionally refine with `refine_dictionary_by_averaging`, write template `.npy`
files, and register via `database.catalog.register_dictionary_run`.

```bash
python -m analysis.dict_learning.code.learn.build_dictionary \
  --db catalog.sqlite \
  --wav /data/recordings/sub/clip.wav \
  --n-rounds 20 --n-windows 10 --min-repeats 3 \
  --refine-iters 1 \
  --out-dir dictionaries/clip
```

`--wav` accepts a path, stem, or filename. Each run writes under a UTC-stamped
subdirectory of `--out-dir`:

```text
--out-dir/
  20260722T083925Z/
    template_000.npy
    manifest.json
```

Re-running with the same `--out-dir` appends a new stamped subdirectory and a
new learning run / dictionary / templates / matchings in the catalog (prior
rows and files are never deleted). To move an older flat layout
(`template_*.npy` + `manifest.json` directly under `--out-dir`) into a stamped
child and rewrite catalog paths:

```bash
python -m database.migrate_flat_dict_dirs --db catalog.sqlite
```

### `extend_dictionary` (coverage-gated stage → catalog)

Start from an optional `--input-dictionary-id` (any prior WAV), apply it to
**target** `--wav` windows (may differ from the parent’s WAV), peel, then add
one atom and refine the whole dictionary until time-bin coverage ≥
`--min-coverage` (default `0.9`).

```bash
python -m analysis.dict_learning.code.learn.extend_dictionary \
  --db catalog.sqlite \
  --wav /data/recordings/sub/clip.wav \
  --input-dictionary-id 42 \
  --n-windows 40 \
  --min-coverage 0.9 \
  --min-repeats 3 \
  --out-dir dictionaries/clip_extend
```

Catalog tool: `extend_dictionary`. Details: [`extend-dictionary.md`](../../documentation/Docs/extend-dictionary.md).

#### CLI parameters

Defaults come from `code/core/defaults.py` where noted (`MIN_COVERAGE`,
`EPS_STEP_FRACTION`, `MIN_REPEATS`, `MAX_REPEATS`, width filters, etc.).
Python entrypoints: `extend_dictionary_for_wav` (catalog I/O; what the CLI
calls) and `extend_dictionary_on_windows` (in-memory only).

| Flag | Default | Description |
|---|---|---|
| `--db` | *(required)* | Path to catalog SQLite. |
| `--wav` | *(required)* | Target WAV (windows/detections); may differ from the parent dictionary’s WAV. |
| `--input-dictionary-id` | `None` | Optional parent `dictionary_id` (any prior WAV). Omit for cold start. |
| `--input-dictionary-dir` | `None` | Optional parent stamped run dir (`manifest.json` + `template_*.npy` / `subspace_*.npy`). Use to chain `n_components>1` stages that skip the catalog. |
| `--window-offset` | `0` | Skip this many windows before taking `--n-windows`. |
| `--n-windows` | `None` | Max windows from the target segmentation after offset. `None` = all remaining. |
| `--window-ids` | `None` | Comma-separated window ids (subset of the target WAV); when set, offset/limit load is skipped and the full segmentation is filtered to these ids. |
| `--out-dir` | `None` | Base directory for run artifacts (UTC-stamped subdir). Default under CWD-style path chosen by the learner when omitted. |
| `--name` | `None` | Dictionary / run name. |
| `--label` | `None` | Alias for `--name` (either may be used). |
| `--segmentation-id` | `None` | GPL segmentation for the target WAV (default: first for that WAV). |
| `--min-coverage` | `0.9` (`MIN_COVERAGE`) | Stop growing when time-bin coverage reaches this fraction in `(0, 1]`. |
| `--max-new-atoms` | `None` | Cap on atoms added this stage. `None` = grow until coverage (or seed fails). `0` = peel/refine only. |
| `--eps-step-fraction` | `0.2` (`EPS_STEP_FRACTION`) | Matching-pursuit stop: relative residual improvement threshold. |
| `--min-repeats` | `3` (`MIN_REPEATS`) | Require at least this many matches in both seed trial and averaged re-MP. |
| `--max-repeats` | `None` (`MAX_REPEATS`) | Cap accepted Mallat matchings per MP run (per window in `per_window` mode). `None` = no cap. Must be `>= min_repeats` when set. |
| `--min-template-width` | `12` (`INCREMENTAL_MIN_TEMPLATE_WIDTH`) | Drop GPL seed candidates narrower than this (time bins). |
| `--max-template-width` | `24` (`INCREMENTAL_MAX_TEMPLATE_WIDTH`) | Drop GPL seed candidates wider than this (time bins). |
| `--n-components` | `1` (`N_COMPONENTS`) | PCA modes per atom. `1` = rank-1 catalog path; `>1` = subspace (manifest-only; use `--input-dictionary-dir` to chain). |
| `--max-relative-distance` | `None` → `MAX_RELATIVE_DISTANCE` | Reject patches leaving more than this fraction of energy unexplained (0–1). CLI `None` falls back to the module default (also `None` = gate off). |
| `--refine-iters` | `1` | Multi-atom average→MP polish passes on the original stack after growth (`0` = no refine). |
| `--pin-matchings` | `None` | Path to `matchings.json` whose activations keep template id and `t_start` through peel, growth, and refine (hand labels). |
| `--seed-window-policy` | `global` (`SEED_WINDOW_POLICY`) | Where to pick seed patches: `global` (all residual GPL candidates) or `low_coverage` (least-covered window ± neighborhood). |
| `--seed-window-neighborhood` | `0` (`SEED_WINDOW_NEIGHBORHOOD`) | With `low_coverage`, also seed from ±N adjacent windows. |
| `--no-normalize` | off | Disable template L2 normalization in MP / rebuilds. |
| `--no-progress` | off | Disable tqdm progress bars. |
| `--mp-mode` | `None` | Matching-pursuit mode override (e.g. `per_window`, `global`). `None` = env / `MP_MODE` default. |
| `--n-workers` | `None` | MP worker count. `None` = env / `MP_WORKERS` default. |
| `--no-register-catalog` | off | Write artifacts only; skip catalog dictionary registration (forced off behavior also applies when `n_components>1`). |

#### Python-only parameters

`extend_dictionary_for_wav(...)` accepts the same knobs as the CLI (keyword
names without flags: `input_dictionary_id`, `pin` → `pinned_matchings` as a
sequence of `Matching`, `normalize=True|False`, `register_catalog=True|False`,
`show_progress`, …). Extra:

| Parameter | Default | Description |
|---|---|---|
| `pinned_matchings` | `None` | In-memory pinned activations (same role as `--pin-matchings`). |
| `register_catalog` | `True` | When `False`, same as `--no-register-catalog`. |

`extend_dictionary_on_windows(stack, detections, ...)` has no catalog I/O. Its
arguments are the learning-stage knobs above plus:

| Parameter | Default | Description |
|---|---|---|
| `stack` | *(required)* | Target windows. |
| `detections` | *(required)* | GPL detections / segments for seeding. |
| `dictionary` | `None` | Parent atoms (or empty / `None` for cold start). |
| `disallow_overlap` | `True` | Non-overlapping placements in MP. |
| `show_progress` | `None` | Progress bars. |
| `mp_mode` / `n_workers` | `None` | Same as CLI. |

**Chained prefixes (LeiData wav1):** run
[`scripts/run_extend_dictionary_leidata_wav1.sh`](scripts/run_extend_dictionary_leidata_wav1.sh)
to extend across `n_windows = 1 → 2 → 4 → 8 → 16 → 30` with **K=1** on spans
`1–2` and **K=10** afterward (subspace stages chain via `--input-dictionary-dir`).
Viewer: [`notebooks/extend_dictionary_leidata_wav1_chain.ipynb`](notebooks/extend_dictionary_leidata_wav1_chain.ipynb).
See “Chained prefixes on one WAV” in the doc above.


### Randomized dictionary building (`create_dictionary` + `refine_dictionary`)

This branch (documented in [`documentation/Docs/randomized-dictionary.tex`](../../documentation/Docs/randomized-dictionary.tex))
implements the **Create→Refine** architecture:

1. **CreateDictionary**: sample `m` *poorly explained* patches from a residual
   partition and turn each sample into an order-1 (rank-1) unit-norm atom.
2. **RefineDictionary**: run a **whole-dictionary** matching pursuit loop
   (overlaps allowed), rebuild each surviving atom via top-`K` PCA of the
   patches assigned to that atom, then prune/drop too-far placements.

Python lives in `code/learn/randomized_dictionary.py` and exposes:
`create_dictionary`, `refine_dictionary`, the convenience wrapper
`build_randomized_dictionary`, and the coverage-gated outer loop
`learn_randomized_outer`.

#### `create_dictionary`

```python
from analysis.dict_learning.code.learn.randomized_dictionary import (
    create_dictionary,
)

d0 = create_dictionary(
    residual,   # Sequence[Window] or dict[window_id -> array]
    partition,  # Sequence[Detection] or Sequence[Segment] intervals
    m=10,        # number of seed atoms to sample
    epsilon=0.1,  # keep pool patches iff ||r||/||x|| > epsilon
    min_template_width=10,
    max_template_width=20,
    original=None,  # denominator stack; defaults to residual
    parent=None,  # optional parent dictionary for the poor-explanation gate
    alpha=None,   # keep only if g_parent(x) <= alpha
    rng=0,
    # plus: windows_for_units=... when partition is Detection inputs
)
```

Return type is `RandomizedCreateResult`, with:
`dictionary` (order-1 `Dictionary`), `pool_size`, `n_sampled`, and the residual
stack used.

#### `refine_dictionary`

```python
from analysis.dict_learning.code.learn.randomized_dictionary import (
    refine_dictionary,
)

result = refine_dictionary(
    residual,
    d0.dictionary,
    iter=3,                    # default refine budget I=3
    n_components=1,           # `K` after refine (1 = rank-1, >1 = subspace)
    n_min=3,                  # prune if < max(n_min, K) assigned patches
    eps_step_fraction=0.2,   # MP stop threshold (paper epsilon)
    max_repeats=None,
    max_relative_distance=None,
    post_refine_max_relative_distance=0.5,  # paper tau
    normalize=True,
    mp_mode=None,
    n_workers=None,
    show_progress=False,
)
```

Return type is `RandomizedRefineResult` with:
`dictionary`, `matchings`, `residual_after`, `energies`,
`iterations_completed`, plus pruning/filter counters.

Notable details:
- Whole-dictionary MP assigns patches to exactly one atom per MP step.
- Overlaps are allowed (`disallow_overlap=False`).
- After refine, placements with unexplained energy fraction
  `dnorm(x, A) > tau` are dropped, and atoms with too few remaining patches
  are pruned.

#### `build_randomized_dictionary` (convenience)

```python
from analysis.dict_learning.code.learn.randomized_dictionary import (
    build_randomized_dictionary,
)

result = build_randomized_dictionary(
    residual,
    partition,
    m=10,
    refine_iters=3,
    n_components=1,
    n_min=3,
    epsilon=0.1,
    min_template_width=10,
    max_template_width=20,
    eps_step_fraction=0.01,
)
```

#### `learn_randomized_outer` (coverage-gated residual outer loop)

Repeated Create→Refine on a deflating residual. Each outer pass samples
`m_per_outer` new seeds, refines them (prune atoms with `< n_min` patches;
drop placements with `dnorm > post_refine_max_relative_distance`), merges
survivors into an accumulating dictionary, and stops when cumulative **time
coverage** exceeds `min_coverage` (or on empty pool / no survivors /
`max_outer`).

```python
from analysis.dict_learning import learn_randomized_outer

result = learn_randomized_outer(
    windows,
    detections,
    m_per_outer=5,                         # new random seeds / outer
    n_min=10,                              # drop atoms with fewer patches
    post_refine_max_relative_distance=0.9, # relative-distance tolerance
    min_coverage=0.5,                      # stop when time coverage > 50%
    max_outer=100,
    refine_iters=3,
    n_components=1,
    epsilon=0.01,
    min_template_width=10,
    eps_step_fraction=0.2,
    rng=0,
    show_progress=True,
    progress_callback=lambda log: print(
        f"outer {log.outer}: atoms={log.n_atoms} cov={log.time_coverage_fraction:.3f}"
    ),
)
# result.dictionary, result.matchings, result.outer_logs, result.stop_reason, …
```

Return type is `RandomizedOuterResult` (`dictionary`, `matchings`,
`residual_after`, `original_stack`, `outer_logs`, coverage fractions,
`n_outer_completed`, `stop_reason`, energies).

### `match_to_dictionary` (matchings + widths → dictionary → refine)

Build a dictionary from **labeled matchings** when you already know which
patches belong to which atom. Callers supply explicit template widths
(`Matching` has no width); `n_freq` comes from the windows. Flow:

1. Placeholder atoms (zeros) for each `template_id` in the matchings.
2. `average_templates` rebuilds each atom from its matched patches (PCA by default).
3. Optional `refine_dictionary_by_averaging` for `refine_iters` average→MP passes
   on the same windows (`refine_iters=0` keeps the input matchings).

```python
from analysis.dict_learning import match_to_dictionary

result = match_to_dictionary(
    windows,                 # Sequence[Window] or {window_id: array}
    matchings,               # Sequence[Matching] labeled by template_id
    widths={0: 21, 1: 20},   # or a shared int, e.g. widths=21
    refine_iters=2,          # 0 = rebuild only
    template_method="pca",
    n_components=1,
    eps_step_fraction=0.2,
    pin_labels=True,         # default: refine keeps the input labels put
)

result.dictionary            # rebuilt Dictionary
result.matchings             # input matchings if refine_iters=0, else rematched
result.residual_after
result.energies              # residual-energy trace
result.iterations_completed  # refine iterations run
result.n_pinned_matchings    # labels carried through refine
```

`pin_labels=True` keeps each input matching on its own atom at its own
`t_start` through every refine iteration, and reserves its span so MP cannot
claim it for another atom. Without it, refine is free to reassign hand labels
and to leave a faint atom with no patches at all (`NoMatchingPatchesError`).
See [`match-to-dictionary.md`](../../documentation/Docs/match-to-dictionary.md#pinned-labels).

CLI (module or `cli.py` subcommand):

```bash
python -m analysis.dict_learning.code.learn.match_to_dictionary \
  --npy-dir /path/to/windows \
  --matchings matchings.json \
  --widths '{"0":21,"1":20}' \
  --refine-iters 2 \
  --out /path/to/out
```

Use `--width 21` for a shared width. Matchings JSON matches `average-templates`
(`{"matchings":[{"template_id","window_id","t_start","amplitude"},…]}`).
With `--out`, writes a stamped dir (`template_*.npy`, `manifest.json`,
`matchings.json`) and prints a JSON summary.

Negative / non-matching patches are not used yet. Details:
[`match-to-dictionary.md`](../../documentation/Docs/match-to-dictionary.md),
future discriminative rebuild:
[`negative-patches-in-template-estimation.md`](../../documentation/Docs/negative-patches-in-template-estimation.md).
Experiment: [`experiments/run_match_to_dictionary_leidata_w2.py`](experiments/run_match_to_dictionary_leidata_w2.py).

### `split_template` (1→2 bipartition of one atom’s patches)

Randomized rank-1 split when one template mixes two call shapes:

```python
from analysis.dict_learning.code.learn.split_template import split_one_template

result = split_one_template(
    patches, base_template=parent_atom, n_trials=20, n_refine=3, epsilon=0.1, min_size=2, rng=0
)
# result.template_a / template_b (unit L2), patches_a / patches_b, relative_reduction
```

| Harness | Role |
|---|---|
| `experiments/split_one_template.py` | Catalog or `--demo` diagnostics |
| `experiments/split_replace_refine.py` | Replace atom → pinned refine → register `source=edit` |

Doc: [`split-template.md`](../../documentation/Docs/split-template.md). Ops map: [`dictionary-operations.md`](../../documentation/Docs/dictionary-operations.md).

### `calc_divergences` (asymmetric atom×atom residual divergences)

After joint MP to residual $R$, for each atom $A$ inflate its matchings to form
$R[-A]$. If $\Delta_A = E(R[-A])-E(R)$ is large enough, for each $B$ re-MP
only $B$ on $R[-A]$ and set:

$$
D(A,B)=\frac{E(R[-A+B])-E(R)}{\Delta_A}
$$

$D\approx 0$ means $B$ substitutes for $A$. Not symmetric. Cost $O(n^2)$ single-atom MPs.

```python
from analysis.dict_learning import calc_divergences

result = calc_divergences(windows, dictionary, min_delta_fraction=1e-6)
# result.divergences  # (n, n); nan on insignificant rows of A
# result.significant, result.template_ids, result.joint_matchings
```

See [`dictionary-operations.md`](../../documentation/Docs/dictionary-operations.md) § Analytical.

Negative entries deserve care: by default the $B$ re-peel may claim spans that
atoms other than $A$ still hold, which lowers the residual for reasons that have
nothing to do with substitutability. Pass `respect_occupancy=True` to restrict
$B$ to what $A$ vacated.

### `calc_template_similarity` (response correlation on a fixed matching)

A cheaper, symmetric alternative that **does not run matching pursuit**. Templates
are unit-$L^2$ rank-1 atoms. For each patch $P_m$ under a *caller-supplied*
matching and each atom $T_i$:

$$
\alpha(T_i,P_m)=\langle T_i, P_m\rangle
$$

Similarity is Pearson correlation of the response vectors:

$$
S_{ij}=\mathrm{corr}(\alpha_i,\alpha_j)
$$

High $S$ means the atoms fire together on those patches. Matching labels /
amplitudes are ignored — only coordinates are used, and every atom is scored on
every patch.

```python
from analysis.dict_learning import calc_template_similarity

sim = calc_template_similarity(windows, dictionary, matchings)
# sim.similarity  # (n, n)
# sim.alphas, sim.n_patches, sim.template_ids
```

### `calc_template_group_similarity` (group-mean responses on labeled matchings)

Partition a **labeled** matching into groups $G_A$ by `template_id`. For unit-$L^2$
atoms:

$$
\mu(A,B)=\frac{1}{|G_A|}\sum_{P\in G_A}\langle T_B,P\rangle,\qquad
S_{AB}=\frac{\mu(A,B)}{\mu(A,A)}
$$

Near 1 means $B$ scores about as well as $A$ on the patches assigned to $A$.
Unlike Pearson similarity, this **uses** labels; empty groups get a `nan` row.

```python
from analysis.dict_learning import calc_template_group_similarity

grp = calc_template_group_similarity(windows, dictionary, matchings)
# grp.similarity, grp.mean_alpha, grp.group_sizes
```

### `algorithm="dp"` (exact non-overlapping matching pursuit)

Greedy Mallat takes the single best placement per step, so an early pick can
block two better ones. With non-overlapping placements the spans are disjoint,
deflating one never touches another's patch, and the energy drops become
**additive** — the window is then maximum-weight interval scheduling, which a DP
solves exactly in $O(T\cdot K)$:

$$
f[t]=\max\big(f[t-1],\ \max_a f[t-w_a]+W[a][t-w_a]\big)
$$

Since there are no steps, `eps_step_fraction` has nothing to measure; sparsity
comes from a per-placement penalty $\lambda = \texttt{lambda\_fraction}\times \max_{a,t}\delta(a,t)$ computed per window, so it is scale-free. Higher
`lambda_fraction` means fewer matchings and higher $E(R)$.

```python
from analysis.dict_learning import matching_pursuit

result = matching_pursuit(
    windows, dictionary, algorithm="dp", lambda_fraction=0.2
)
```

On dictionary 122 (20 windows, 15 atoms) DP reaches $E(R)/E(S)=0.706$ versus
greedy's $0.828$ — 12% more of the stack explained, with fewer matchings and
about 8× less wall time. Overlap must be disallowed for exactness, so the DP
path forces it. Benchmark:
[`experiments/bench_mp_dp.py`](experiments/bench_mp_dp.py); design notes:
[`mp-performance.md`](../../documentation/Docs/mp-performance.md).

### `learn_averaged_templates_from_gpl` (seed → average → MP)

Notebook pipeline as a library function. Each outer round on the **current residual**:

1. **Seed trials** — every width-eligible GPL candidate; require `>= min_repeats` matchings.
2. **Average** — cut every matching from that residual and average into a new template.
3. **Re-MP** — Mallat again with the averaged template; again require `>= min_repeats`.
4. **Select** — among candidates that pass both checks, keep the lowest residual energy.

Rounds continue until no candidate qualifies (`below_min_repeats`), no candidates remain, or the optional `n_rounds` safety cap is hit. Returned seeds are the **refined** (averaged-template) rounds.

```python
from analysis.dict_learning import learn_averaged_templates_from_gpl

result = learn_averaged_templates_from_gpl(
    windows,                   # Sequence[Window] (or residual dict)
    detections,                # or list[Segment]
    n_windows=4,               # use n windows starting at window_offset
    window_offset=0,
    n_rounds=None,             # optional max rounds; None = grow until stop
    eps_step_fraction=0.2,     # shared MP stop for seed and averaged stages
    min_repeats=3,             # required in both seed trial and averaged re-MP
    min_template_width=None,   # filter GPL candidate widths (time bins)
    max_template_width=24,
    normalize=True,
    # After each outer round, write round_XXX/template.npy + matchings.json:
    template_out_dir="/path/to/dict_learning_templates/run",
)
```

`build_dictionary` / preprocess expose the same knobs (plus `--refine-iters` for an optional multi-atom average→MP pass after the outer rounds). See `python -m analysis.dict_learning.code.learn.build_dictionary --help`.

Per-round on-disk layout (written immediately when each round finishes):

```text
template_out_dir/
  manifest.json
  round_000/template.npy
  round_000/matchings.json
  round_001/...
```

```python
# Same algorithm from paths
from analysis.dict_learning import learn_averaged_templates_from_gpl_paths

result = learn_averaged_templates_from_gpl_paths(
    npy_dir="…/spectrogram_npy",
    gpl_output_dir="…/output",
    n_windows=4,
    n_rounds=5,
    min_repeats=3,
    template_out_dir="…/templates",
)
```

Single-round helper (used inside the loop, also callable alone):

```python
from analysis.dict_learning import Seed, refine_seed_by_averaging

seed = Seed.from_gpl(windows, segments, …)
refined = refine_seed_by_averaging(windows, seed, eps_step_fraction=0.2)
# refined.get_template()  → averaged Template
# refined.get_matchings() → locations after re-MP
```

### `refine_dictionary_by_averaging` (average every atom → multi-atom MP)

Refine an existing multi-atom dictionary from a set of matchings. In each
iteration, matchings are grouped by `template_id`, their patches are averaged
from the original once-centered windows, and a fresh MP loop is run with the
whole averaged dictionary. The resulting matchings feed the next iteration.

```python
from analysis.dict_learning import refine_dictionary_by_averaging

result = refine_dictionary_by_averaging(
    windows,                    # Sequence[Window] or centered residual dict
    dictionary,                 # Dictionary containing the referenced atoms
    initial_matchings,          # Matching locations grouped by template_id
    iter=5,                     # average → MP repetitions
    eps_step_fraction=0.2,
)

result.dictionary            # final averaged Dictionary
result.matchings             # final multi-atom MP matchings
result.residual_after        # residual after the final MP loop
result.energies              # final MP loop's residual-energy trace
result.iterations_completed  # 5
```

Pass `template_method="pca"` (to either refine helper, `learn_averaged_templates_from_gpl`,
`build_dictionary_for_wav`, or `incremental_*`, or via the `--template-method pca` CLI flag)
to rebuild each atom from the leading principal component of its patches instead of the
mean. PCA maximizes single-atom energy reduction and is robust to sign/phase cancellation
that would zero out a plain average.

Every atom must have at least one valid matched patch in every iteration;
otherwise the function raises `ValueError`. `iter=0` returns copies of the
inputs without averaging or MP.

Pass `pinned_matchings=` (e.g. hand labels) to exempt activations from
reassignment: each is re-fitted to its atom at its own `t_start`, deflated
before MP, and its span withheld from MP. Pinned atoms therefore always have
patches to average, and pinned patches stay on the atom they were given to.

### `LearnedDictionary` getters

| Method | Returns | Meaning |
|---|---|---|
| `len(result)` / `__len__` | `int` | Number of completed rounds (seeds kept). |
| `get_seeds()` | `list[Seed]` | One `Seed` per round. For `learn_averaged_templates_from_gpl`, each seed is the **averaged-template** round (template + matchings + residual). |
| `get_templates()` | `list[Template]` | Averaged (or seed-only) template from each round, in round order. |
| `get_matchings()` | `list[Matching]` | All matchings across all rounds (concatenated). |
| `get_matchings_for(template_id)` | `list[Matching]` | Locations for one template (`template_id` equals the outer round index `0…m-1`). |
| `get_energy_before()` | `float` | Stack energy at the start of learning (after mean-centering). |
| `get_energy_after()` | `float` | Stack energy of the final residual. |
| `get_delta_energy_total()` | `float` | `energy_before - energy_after`. |
| `get_residual_after()` | `dict[str, ndarray] \| None` | Final residual spectrograms (`window_id` → `(F, T)`). |

Reading results in the same style as a single `Seed`:

```python
result = learn_averaged_templates_from_gpl(windows, detections, n_windows=4, n_rounds=5)

for seed in result.get_seeds():
    tmpl = seed.get_template()          # Template: data (F, W), template_id, source_segment_id
    for m in seed.get_matchings():      # Matching: template_id, window_id, t_start, amplitude
        print(m.window_id, m.t_start, m.amplitude)

# Or flatten across rounds
templates = result.get_templates()
all_locs = result.get_matchings()
locs_round0 = result.get_matchings_for(0)
```

## Tests

```bash
python3 analysis/dict_learning/tests/test_seed_from_gpl.py
python3 analysis/dict_learning/tests/test_learn_dictionary.py
python3 -m pytest analysis/dict_learning/tests/ -q
```

## Types

Core dataclasses live in `code/core/objects.py` and `code/io/io_gpl.py`. Result wrappers for MP and seeding are in `code/mp/mp_step.py` and `code/learn/seed_from_gpl.py`. Multi-round results are in `code/learn/learn_dictionary.py` (`LearnedDictionary`).

### `Window` (`code/core/objects.py`)

One GPL process-window spectrogram tile with STFT metadata.

| Field | Type | Meaning |
|---|---|---|
| `window_id` | `str` | Stable id (usually the `.npy` stem). |
| `data` | `ndarray` | Whitened spectrogram, shape `(F, T)`. |
| `start_sample`, `end_sample` | `int` | File-global 1-based sample span of this window. |
| `overlap` | `int` | STFT hop length in samples (used to map samples ↔ time bins). |
| `fftl`, `sample_freq`, `start_freq`, `end_freq` | optional | Spectrogram geometry from the sibling `.json` sidecar. |
| `npy_path`, `meta` | optional | Source path and full sidecar dict. |

Properties: `n_freq`, `n_time`.

### `Segment` (`code/core/objects.py`)

A spectrogram patch cropped from a `Window` where a GPL detection overlaps.

| Field | Type | Meaning |
|---|---|---|
| `segment_id` | `str` | Unique patch id. |
| `data` | `ndarray` | Patch array `(F, W)`. |
| `window_id` | `str` | Source window. |
| `t_start`, `t_end` | `int` | Window-local time bins (`t_end` exclusive). |
| `global_start_sample`, `global_end_sample` | `int` | File-global 1-based sample bounds from the CSV. |
| `detection_index` | `int` | Row index in the detection CSV. |
| `boundary_cross_flag` | `int` | GPL boundary-cross flag. |

Property: `.energy` = $\lVert U \rVert_F^2$ of `data` (used to rank seed candidates).

### `Template` (`code/core/objects.py`)

A dictionary atom: fixed-width pattern copied from a seed segment (or refined later).

| Field | Type | Meaning |
|---|---|---|
| `template_id` | `int` | Integer atom id. |
| `data` | `ndarray` | Pattern `(F, W)`. |
| `source_segment_id` | `str \| None` | Segment this template was seeded from. |

Properties: `.width` (`W`), `.n_freq` (`F`).

### `Matching` (`code/core/objects.py`)

One activation of a template inside a window residual.

| Field | Type | Meaning |
|---|---|---|
| `template_id` | `int` | Which template fired. |
| `window_id` | `str` | Which window it landed in. |
| `t_start` | `int` | Left time-bin index of the placement. |
| `amplitude` | `float` | Scalar weight used when subtracting the template. |

### `Dictionary` (`code/core/objects.py`)

Active ordered set of templates for MP.

| Field | Type | Meaning |
|---|---|---|
| `templates` | `dict[int, Template]` | Atoms keyed by `template_id`. |
| `order` | `list[int]` | Insertion order used when iterating. |

Methods: `add(template)`, `__iter__`, `__len__`.

### `Detection` (`code/io/io_gpl.py`)

One row from a GPL `CSV_*.csv` (file-global sample times). Frozen dataclass.

| Field | Type | Meaning |
|---|---|---|
| `start_time`, `end_time` | `int` | 1-based file-global sample interval. |
| `index` | `int` | Row index in the CSV. |
| `fname` | `str` | Source WAV name from the CSV (if present). |
| `boundary_cross_flag` | `int` | GPL flag from the CSV. |
| `raw` | `dict \| None` | Original CSV row (optional). |

Mapped into `Segment`s via `detections_to_segments`.

### `Seed` (`code/learn/seed_from_gpl.py`)

Result of energy-ranked single-template Mallat selection. Construct with `Seed.from_gpl(...)` or `Seed.from_paths(...)`; access state only through getters (`is_selected`, `get_template`, `get_matchings`, `get_mp_delta_energies`, …). See [`Seed` getters](#seed-getters) above.

### `LearnedDictionary` (`code/learn/learn_dictionary.py`)

Multi-round residual-deflation result from `learn_from_gpl` or `learn_averaged_templates_from_gpl`. Holds a list of per-round `Seed`s plus stack-level energies and the final residual. See [`LearnedDictionary` getters](#learneddictionary-getters) above.

### `DictionaryRefinementResult` (`code/learn/learn_dictionary.py`)

Result of `refine_dictionary_by_averaging`: the final averaged `dictionary`,
final MP `matchings`, `residual_after`, final MP energy trace, and
`iterations_completed`.

### `MpStepResult` (`code/mp/mp_step.py`)

Outcome of one Mallat `mp_step` on a residual stack.

| Field | Type | Meaning |
|---|---|---|
| `matching` | `Matching \| None` | Best activation this step, or `None` if no improvement. |
| `energy_before`, `energy_after` | `float` | Stack Frobenius energy before/after the subtraction. |
| `delta_energy` | `float` | `energy_before - energy_after`. |

### `MpLoopResult` (`code/mp/mp_step.py`)

Outcome of `run_mp_loop` (repeated `mp_step` until a stop rule).

| Field | Type | Meaning |
|---|---|---|
| `matchings` | `list[Matching]` | All activations recorded by the loop. |
| `residual_stack` | `dict[str, ndarray]` | Final residuals (`window_id` → `(F, T)`). |
| `energies` | `list[float]` | Energy after each step (and initial). |
| `stopped_reason` | `str` | Why the loop stopped (`eps_step_fraction`, `no_improvement`, `energy_floor`, …). |
