---
name: Baseline: CLIs, pruning, and tests
overview: "Before merging panMP into Matching Pursuit: prune dict_learning to the randomized build plus the Mallat MP it needs, give the randomized build, Mallat MP and the pan matrix profile CLI commands, and test them in two layers: unit tests for maximum function coverage, and integration tests through the CLIs at 100% line and branch coverage of everything the pipelines call. The suite runs in a few minutes."
todos:
  - id: b0-env
    content: Make the repo self-contained. Add pyproject.toml with test extras, remove the hidden dependency on another checkout's `database` package, and record the current test baseline.
    status: completed
  - id: b1-prune
    content: Prune dict_learning code to the randomized build, Mallat MP, and what learn_from_motifs needs. Leave notebooks, experiments, scripts and reports in place.
    status: completed
  - id: b2-cli
    content: CLI commands randomized-build, matching-pursuit, and pan (pan matrix profile), reading GPL output folders, WAVs or .npy series directly, with no catalog DB.
    status: completed
  - id: b3-layout
    content: Move tests into analysis/test/unit/ and analysis/test/integration/, with shared synthetic fixtures.
    status: completed
  - id: b4-tests
    content: Integration tests through the CLIs at 100% line and branch coverage of the pipeline code. Unit tests for maximum coverage of individual functions.
    status: completed
  - id: b5-docs
    content: Write analysis/test/README.md (setup and how to run), and update analysis/README.md, analysis.md and dict_learning/README.md.
    status: completed
isProject: false
---

# Baseline: CLIs, pruning, and tests

Comes before [stage1_motif_dictionary.md](stage1_motif_dictionary.md). The goal is a small, known-good starting point with three pipelines, each runnable from the command line and fully covered:

| Pipeline | What it does | CLI |
|---|---|---|
| **Randomized build** | Learns a dictionary: random seeds → Mallat MP + PCA refine → coverage-gated outer loop (`learn_randomized_outer`) | `python -m analysis.dict_learning randomized-build` |
| **Mallat MP** | Places a saved dictionary on a recording and reports matchings and residual | `python -m analysis.dict_learning matching-pursuit` |
| **Pan matrix profile** | Runs panMP on a 1-row or multi-row signal and reports the surface and the top motif pairs | `python -m analysis.matrix_profile pan` |

`learn_from_motifs` stays in the code (it's the `mp-greedy` mode for later) but is not part of the baseline CLI or the integration coverage. Its existing unit tests stay.

## Where things stand (checked 2026-10-25)

- **The randomized build has no CLI.** `learn_randomized_outer` was only run by the old webserver's `randomized_builder` terminal command (`gplrev3-webserver/backend/app/services/command_service.py`). That command loaded windows from the catalog DB and saved results back into it. The reports under `dict_learning/reports/randomized_*` and the results notebook read the files it wrote: `summary.json`, `matchings.json`, `outer_logs.jsonl`, and atoms.
- **panMP has no CLI, and is 1-D only.**
  - `matrix_profile.panMP()` does `np.asarray(TS).ravel()`, so a 2-D array would be silently flattened rather than rejected. It uses the pure-Python reference `mpxForPan` (MATLAB parity, slow) and shows its plot with `plt.show()`. Only parity tests call it.
  - The HMM doesn't call `panMP()`. It has its own length sweep (`pan_prototypes`) on `mpx_fast`, over the whitened spectrogram summed over frequency at 10 Hz.
  - The matrix-profile notebooks use `rms_envelope` of the raw WAV, also 1-D.
  - A multi-row matrix profile exists, but only at a single length: `mpx_channels`, used by `learn_from_motifs`. There's no multi-row length sweep.
- **There's a hidden dependency on another checkout.** `code/tools.py`, `code/cli.py` and `code/catalog.py` do `from database import catalog`. The import resolves to `/home/rivo/production-gplrev3-webserver/database`, installed into the Python environment; on a clean machine these fail. `tools.py` is needed by the randomized build.
- **No `pyproject.toml`,** although `analysis/README.md` says `pip install -e .`. `pytest-cov` and `coverage` aren't installed.
- **Tests are spread over** `analysis/test/`, `analysis/dict_learning/tests/`, `analysis/GPL_Python/tests/` and `analysis/HMM/test_soft_hmm.py`. Now: 208 passed, 1 failed (`test_learn_dictionary.py::test_save_round_template_each_outer_iteration`), 1 collection error (`test/test_matlab_report_193500.py` imports a missing `report_193500`).
- **GPU:** 2× RTX 3080 Ti and CuPy 14.1.1 here, so the CUDA path in `mp.py` can be tested on this machine.

## b0 — Self-contained environment

- `pyproject.toml` with runtime dependencies (numpy, numba, tqdm, matplotlib, plus scipy if the kept code uses it), an optional `[gpu]` extra (cupy) and a `[test]` extra (pytest, pytest-cov).
- Remove every `database` import from the kept code (see b1). The CLIs read files directly.
- Record the current pass/fail baseline before deleting anything.

## b1 — Prune dict_learning to the randomized build

Only **library code and its tests** are pruned. **Notebooks, experiments, scripts and reports stay in place**, because others may be using them (see Open questions about the ones that import removed code).

What the randomized build uses, traced from its imports:

| Module | Keep | Remove |
|---|---|---|
| `learn/randomized_dictionary.py` | All: `create_dictionary`, `refine_dictionary`, `build_randomized_dictionary`, `learn_randomized_outer` | — |
| `learn/seed_from_gpl.py` | `detections_to_segments`, `mean_center`, moved into `core.py` or `io.py` | The seed trial (`Seed`, `seed_from_gpl*`, its CLI); then delete the file |
| `learn/learn_dictionary.py` | `refine_dictionary_by_averaging` and `NoMatchingPatchesError` (needed by `learn_from_motifs`), moved to `learn/refine.py` | The greedy builder (`learn_averaged_templates_from_gpl*`), `learn_from_gpl*`, `refine_seed_by_averaging`, `save_round_template`, its CLI; then delete the file |
| `learn/build_dictionary.py`, `extend_dictionary.py`, `match_to_dictionary.py`, `split_template.py`, `similarity.py` | — | Delete |
| `learn/learn_from_motifs.py` | Keep (later `mp-greedy` mode) | — |
| `tools.py` | `matching_pursuit`, `rebuild_template_from_matchings`, `average_templates`, `pca_basis`, the patch helpers they call | `build_dictionary_from_detections` (greedy), `remove_template`, `merge_templates`, catalog persistence (`catalog.persist_*`, `tool_run_id` arguments) |
| `mp.py` | `run_mp_loop` (global and per-window), `score_shifts` (numba, numpy, CuPy), `refit_and_deflate`, `residual_energy`, the worker and progress helpers that loop uses | Candidate and seed-average trial machinery (about lines 791–1112), the DP solver `run_mp_dp_loop` (about 2186–2586), the subspace MP loop (about 2587–end). The randomized build calls none of them. |
| `core.py`, `io.py` | Whatever kept code calls | Helpers nothing kept calls (seed-window selection etc.) |
| `catalog.py`, old `cli.py` | — | Delete `catalog.py` (web viewer payload); replace `cli.py` (b2) |
| `__init__.py` | Re-exports of kept code | Exports of removed functions |
| `dict_learning/tests/` | Tests of kept code, moved to `analysis/test/unit/` | Tests of removed builders |

Rule: remove code only once it's confirmed that no kept entry point reaches it (by the integration coverage report and a search for callers).

## b2 — CLI commands

Each command has `--help` and a `main(argv)`, so tests can run it in-process.

### `python -m analysis.dict_learning randomized-build`

- **Input:** `--gpl-dir`, a GPL output folder (`spectrogram_npy/` plus `CSV_*.csv`). `--n-windows`/`--window-offset` pick a slice.
- **Knobs**, mirroring `learn_randomized_outer` and the old `randomized_builder` flags: `--m-per-outer`, `--min-coverage`, `--max-outer`, `--refine-iters`, `--n-components`, `--n-min`, `--epsilon`, `--min-template-width`, `--max-template-width`, `--eps-step-fraction`, `--max-relative-distance`, `--post-refine-max-relative-distance`, `--seed`, `--mp-mode`, `--workers`, `--compute {cpu,cuda,auto}`.
- **Output** (`--out DIR`), in the layout the existing reports and notebook read: `atoms/template_*.npy`, `matchings.json`, `summary.json`, `outer_logs.jsonl`. Exit code 0 on success, non-zero on bad input or when no atoms survive.

### `python -m analysis.dict_learning matching-pursuit`

- Runs Mallat MP with a saved dictionary (`--dictionary DIR`, e.g. `randomized-build`'s output) on a GPL folder.
- Knobs: `--eps-step-fraction`, `--max-relative-distance`, `--mp-mode`, `--workers`, `--compute`.
- Writes `matchings.json` and `summary.json` (energy before and after, coverage).

### `python -m analysis.matrix_profile pan`

Runs the pan matrix profile (matrix profile over a sweep of motif lengths) on a **1-row or multi-row** signal of shape `(rows, T)`.

- **Input, pick one:**
  - `--series FILE.npy`: 1-D, or 2-D `(rows, T)`;
  - `--wav FILE.wav --rate HZ --signal {rms,bands}`: `rms` is the RMS envelope (`rms_envelope`, 1 row); `bands` is band energy from raw audio (`bands.band_energy`);
  - `--gpl-dir DIR --rate HZ --signal {energy,bands,full}`, built from the whitened spectrogram: `energy` is summed over frequency (1 row, the HMM's signal); `bands` is summed within frequency ranges (`--band-edges`, default 100 Hz ranges over GPL's frequency span); `full` is every frequency row.
- **Knobs:** `--min-len`, `--max-len`, `--step` (panMP's `MinMotifLen`/`MaxMotifLen`/`StepSize`), `--top-k`, `--threads`, `--engine {fast,reference}`.
  - `fast` uses numba: `mpx_fast` for 1 row, `mpx_channels` for several (sum of per-row z-normalized distances).
  - `reference` is the MATLAB-parity `mpxForPan`. It is 1-row only and slow, so it's for small inputs; on multi-row input the command exits with a clear error.
- **Output** (`--out DIR`):
  - `pan.npy` (the surface: time × motif length);
  - `pairs.json` (the top non-overlapping pairs across lengths: start, neighbour, length, distance, z-score, via `top_pairs_by_zscore`);
  - `signal.json` (input type, rows, rate, band edges);
  - optionally `pan.png` (`--plot`).
- **Code changes:**
  - New `pan_profile(signal, lengths, engine, nthreads)` in `analysis/matrix_profile` that accepts `(rows, T)`.
    - For 1 row with the reference engine, it reproduces `panMP()` exactly; a parity test checks this.
    - For several rows, it uses `mpx_channels` at each length, with the same length normalization and z-scoring.
  - `panMP()` becomes a thin wrapper over it that keeps today's behaviour for notebooks, but raises on 2-D input instead of silently flattening.
  - Plots are saved with the Agg backend instead of `plt.show()`, so a CLI or test run never blocks.
  - A shared `motif_signal()` builder for the GPL and WAV signal options, reused later by Stage 1's search signal (s2 in stage1_motif_dictionary.md).

## b3 — Test layout

```
analysis/test/
├── README.md                 setup and how to run (b5)
├── conftest.py               shared fixtures: synthetic GPL folder, WAV, series
├── unit/
│   ├── dict_learning/        kept tests moved from analysis/dict_learning/tests/
│   └── matrix_profile/       moved from analysis/test/ (test_mpxForPan, test_panMP, test_motifs, …)
└── integration/
    ├── test_randomized_build_cli.py
    ├── test_matching_pursuit_cli.py
    └── test_pan_cli.py
```

`GPL_Python/tests/` and `HMM/test_soft_hmm.py` stay where they are for now; nothing in this plan touches them.

**Fixtures** (written into `tmp_path`):
- **Synthetic GPL folder:** 2–3 whitened windows of about 16 frequency rows × 300–400 time steps, with `spwhiten_*.json` sidecars (the fields `io.py` reads). 2–3 planted motif shapes repeat in low noise, and `CSV_*.csv` has detections over the planted copies plus a few over noise.
- **Edge-case variants:** no detections, detections too narrow for the width limits, a single window.
- **Short WAV, 1-D and 2-D `.npy` series** with planted repeats, for `pan`, including one too short for the requested `--max-len`.

## b4 — Tests and coverage

### Two layers

| Layer | Purpose | Drives code through | Coverage target |
|---|---|---|---|
| **Unit** (`analysis/test/unit/`) | Each function behaves correctly, including edge cases | Direct function calls | As high as practical per function; reported, not gated |
| **Integration** (`analysis/test/integration/`) | Each pipeline works end to end, and every line and branch it runs is exercised | The CLI `main(argv)`, plus one real `python -m` subprocess per command | **100% line and branch** of the pipeline code, gated |

### What "pipeline code" means for the integration gate

It's every function the three CLIs can call, directly or indirectly, after pruning. Kept functions that only other callers use (e.g. `learn_from_motifs` helpers such as `band_motif_profile`, notebook helpers such as `format_clock`) are listed by name in `analysis/test/integration/.coveragerc` under `exclude_also`. Excluding a function's `def` line excludes its whole body, so the list stays short and reviewable.

Gate command:

```bash
pytest analysis/test/integration --cov=analysis.dict_learning --cov=analysis.matrix_profile \
  --cov-branch --cov-config=analysis/test/integration/.coveragerc --cov-fail-under=100
```

### Integration cases

**`randomized-build`:**
- **Stop reasons:** coverage reached, `max_outer` hit, a pass adding no atoms, empty seed pool.
- **Options:**
  - `--n-components 1` and `>1` (PCA rebuild, global MP);
  - post-refine gate on and off;
  - width limits on and off;
  - `--mp-mode` global and per-window;
  - `--workers 1` and `>1` (process pool);
  - `--compute cpu` and `cuda` (CUDA skipped when no GPU).
- **Bad inputs:** missing folder, no CSV, empty CSV, and an out-of-range value for each validated argument.

**`matching-pursuit`:** a dictionary from a `randomized-build` run, an empty or missing dictionary, mismatched frequency rows, and the same mode, worker and compute matrix.

**`pan`:**
- each input type and each `--signal` option (1-row and multi-row), both engines (and the reference engine refusing multi-row input), and `--plot` on and off;
- step size 1 and greater than 1;
- input too short, `max-len` smaller than `min-len`, and bad `--rate`.

### Leftover branches

When a pipeline branch can't be reached through a CLI, the default is to delete it as dead code. Branches that depend only on the environment (numba or CuPy missing) are forced through flags or environment variables where the code supports that. Anything still left gets `# pragma: no cover` with a one-line reason and is listed in the testing README.

### Run time

Parameters are tuned so the **whole suite (unit + integration) runs in a few minutes** on this machine:
- tiny fixtures; `--max-outer` 1–3; `--refine-iters 1`; short motif lengths; fixed `--seed`;
- the `reference` engine only on a series of a few hundred samples.

numba compiles on first use, and `cache=True` makes later runs faster. The README states both cold and warm timings. A test-time budget check in CI is optional.

## b5 — Docs

**`analysis/test/README.md`:**
- install (`pip install -e '.[test]'`, plus `[gpu]` for the CUDA cases);
- how to run unit tests, integration tests, the coverage gate, one file or one test;
- what the fixtures contain, and expected run time;
- the coverage exclusion and pragma list with reasons;
- how to run each CLI on real data.

**Other docs:** update `analysis/README.md`, `analysis/analysis.md` and `dict_learning/README.md`. Remove the pruned builders (`build_dictionary`, `extend`, …), document the three CLI commands, and point to the testing README.

## Results (2026-10-25)

All six steps are done; the code isn't committed yet. How to run everything: [analysis/test/README.md](../analysis/test/README.md).

- **Suites:** all of `analysis/` gives 214 passed, 7 skipped (MATLAB not installed), 1 collection error (`test_matlab_report_193500.py`, which predates this work), in about 9 s cold and 6 s warm. Integration: 85 tests in about 7 s, **100% line and branch coverage** of the pipeline code. Unit coverage report: 67% overall; the main gap is the numba kernels, which coverage can't trace.
- **CLIs:** `python -m analysis.dict_learning randomized-build | matching-pursuit` and `python -m analysis.matrix_profile pan` (1-row and multi-row; series, WAV, or GPL folder).
- **Pruned:** see `analysis/dict_learning/README.md` § Removed. `dict_learning/code` went from about 12,300 to about 4,900 lines.

How this differs from the plan above:

- **Coverage markers, not `# pragma: no cover`.** Unreachable lines are tagged `# no-pipeline-cover: <reason>` or `# no-pipeline-branch: <reason>`. Only the integration config skips them, so the unit report still counts that code. 153 + 11 markers. This replaced the plan's per-function `exclude_also` list.
- **numba kernels are excluded from the gate** (coverage can't trace compiled code). Running the gate with `NUMBA_DISABLE_JIT=1` was considered and not used.
- **CUDA lines are tagged `# gpu-only`.** They count on GPU machines. On a CPU-only machine, set `COVERAGE_EXCLUDE_GPU` / `COVERAGE_PARTIAL_GPU` (see the testing README).
- **Small code changes for the CLIs:**
  - `learn_randomized_outer` gained `mp_mode` and `n_workers` and passes them to every refine (before, the old command set them only through environment variables).
  - `panMP` gained `engine`, `nthreads`, `plot_path` and `verbose`, and rejects 2-D input. New: `length_profile`, `pan_lengths`, `pan_profile`, `pan_top_pairs`, `matrix_profile/signals.py`.
  - Two behaviour-preserving restructures so blocks can be marked whole: the partial-crop path in `tools._extract_patches`, and `mp._progress_by_default`.
- **Found while testing:** `top_pairs_by_zscore` blocks only a new pair's first copy, so `pan`'s pairs can share a copy. This is documented, not changed. It matters for Stage 1 (design doc §2).
- **Still broken, as decided:** 32 notebooks, experiments and scripts, and `GPL_Python/preprocess.py` (the `backend/preprocess.py` shim calls it). The HMM's `observations._mpx()` import bug is unchanged (Stage 1, s0).

## Decisions (2026-10-25)

1. The panMP pipeline is "run pan matrix profile on something", not `learn_from_motifs`.
2. Suite run time: a few minutes at most.
3. Unit tests aim for maximum function coverage. Integration tests are gated at 100% of the code the pipelines call.
4. Tests go in `analysis/test/unit/` and `analysis/test/integration/`, with the README in `analysis/test/`.
5. Notebooks, experiments, scripts and reports stay for now. The 32 that import removed code are listed as broken in `dict_learning/README.md` (option a).
6. `pan` accepts multi-row input (bands or full spectrogram) as well as 1-row.
