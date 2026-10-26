# analysis/test

Tests for the three baseline pipelines, the catalog, and the code under them:

| Entry point | Command |
|---|---|
| Randomized dictionary build | `python -m analysis.dict_learning randomized-build [--catalog DB]` |
| Mallat matching pursuit with a saved dictionary | `python -m analysis.dict_learning matching-pursuit [--catalog DB]` |
| Pan matrix profile | `python -m analysis.matrix_profile pan` |
| Catalog of runs and CSVs | `python -m database register` / `index` |
| Catalog HTTP API | `/api/catalog/*`, `/api/library/analyses`, `/api/jobs/{id}/song-structure?csvId=` |

There are two layers:

| Layer | Folder | Drives the code through | Coverage |
|---|---|---|---|
| **Unit** | `unit/` | Direct function calls | Reported, not enforced. Aim: as high as practical per function. |
| **Integration** | `integration/` | The CLIs: `main(argv)` in-process, plus one real `python -m` subprocess per command; the catalog API through FastAPI's `TestClient` | **100% of lines and branches the CLIs and the catalog API can run**, enforced |

## Layout

```
analysis/test/
├── README.md
├── conftest.py          shared fixtures (session-scoped, written to pytest's tmp dir)
├── synthetic.py         builders for synthetic GPL folders, series, WAVs, song-structure CSVs; run_module()
├── catalog_setup.py     helpers for the catalog tests (run the CLIs, copy the catalog)
├── unit/
│   ├── .coveragerc      unit coverage report
│   ├── database/        catalog helpers, validation, hand-made run folders
│   ├── dict_learning/   MP, refine, randomized build, learn_from_motifs, bands
│   └── matrix_profile/  mpxForPan, panMP, pan_profile, signals, MATLAB parity
└── integration/
    ├── .coveragerc      the 100% gate
    ├── conftest.py      catalog_runs: a build + MP registered with --catalog, plus one CSV
    ├── test_randomized_build_cli.py
    ├── test_matching_pursuit_cli.py
    ├── test_pan_cli.py
    ├── test_catalog_cli.py   --catalog, python -m database register / index
    └── test_catalog_api.py   the backend's catalog routes (TestClient)
```

`GPL_Python/tests/` and `HMM/test_soft_hmm.py` are separate suites and stay
where they are.

## Setup

Python 3.11 or newer (developed on 3.13). From the repository root, in a
virtual environment **used only for this checkout**:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'          # numpy, scipy, matplotlib, numba, tqdm, FastAPI, pytest, pytest-cov, httpx2, …
pip install -e '.[test,gpu]'      # also CuPy, to run the CUDA test
```

Don't install this repo into an environment that another checkout is also
installed into, such as the production webserver's. Both checkouts provide a
top-level `analysis` package, and the last install wins for every program using
that environment.

You can also skip the install: `pyproject.toml` tells pytest to put the
repository root and `backend/` first on `sys.path` (`pythonpath = [".", "backend"]`),
so the tests always import this checkout's `analysis`, `database` and the
backend's `app`. You then only need the packages listed in `pyproject.toml`,
plus `pytest`, `pytest-cov` and `httpx2` (FastAPI's `TestClient` needs it).

## Running

Always run from the repository root.

```bash
# Everything under analysis/ (unit + integration + GPL_Python + HMM). The flag
# keeps one known collection error (below) from stopping the run.
python -m pytest analysis --continue-on-collection-errors

# One layer (unit/ holds the known collection error too)
python -m pytest analysis/test/unit --continue-on-collection-errors
python -m pytest analysis/test/integration

# The coverage gate (fails below 100%)
python -m pytest analysis/test/integration \
  --cov --cov-config=analysis/test/integration/.coveragerc

# The unit coverage report
python -m pytest analysis/test/unit --continue-on-collection-errors \
  --cov --cov-config=analysis/test/unit/.coveragerc --cov-report=term

# One file, one test, or skip the CUDA test
python -m pytest analysis/test/integration/test_pan_cli.py
python -m pytest analysis/test/integration -k stops_at_coverage
python -m pytest analysis/test/integration -m "not gpu"
```

### Run time

Measured on the development server (2× RTX 3080 Ti, Python 3.13):

| Run | Wall time |
|---|---|
| All of `analysis/`, warm numba cache | about 10 s |
| Integration suite with the coverage gate | about 17 s (about 9 s with the GPUs hidden) |

A cold numba cache adds a few seconds.

numba compiles its kernels on first use and caches them (`cache=True`) in
`__pycache__/`, so the first run on a machine is the slow one.

### Known results

- **7 skipped:** `unit/matrix_profile/test_matlab_equivalence.py` needs MATLAB
  on `PATH`.
- **1 collection error:** `unit/matrix_profile/test_matlab_report_193500.py`
  imports `report_193500`, which isn't in the repository. This predates the
  baseline work.
- **1 warning:** CuPy prints a `FutureWarning` about `cupyx.jit.rawkernel` when
  the CUDA test runs.

## Fixtures

Built by `synthetic.py`, written once per test session under pytest's temporary
directory. Nothing in the repository is read or written.

- **`gpl_dir` / `gpl_folder`:** a GPL output folder, as `GPL_Python` writes it.
  - 3 windows of 12 frequency rows × 240 columns; hop 10 samples at 1 kHz, so
    100 columns per second.
  - 3 planted time–frequency shapes, 4 copies per window, in low noise.
  - The last 8 columns of each window are exactly zero.
  - `CSV_synthetic.csv` has one detection on every planted copy, plus per
    window: one over noise, one wider than the default width limit, one on the
    zero stretch, and one shorter than a hop.
  - Also present: `_display.npy` arrays and a stray `.npy` without a sidecar,
    which loaders must skip.
- **`gpl_variant(**kwargs)`:** other folders, built once per variant: no CSV
  (`detections="none"`), a header-only CSV (`"empty"`), detections only 2
  columns wide (`"narrow"`), sidecars without `sample_freq`, fewer windows.
- **`series_files`:** `.npy` series with a shape repeated at known starts: 1-D,
  3 rows, one too short for the motif lengths, and a 3-D array (invalid).
- **`wav_file` / `stereo_wav_file`:** 6 s at 2 kHz, noise plus repeated 150 Hz
  and 350 Hz bursts, mono and stereo.
- **`catalog_runs`** (integration only): a GPL folder whose window ids carry the
  stem `SOCAL_T_230426_193500`, a randomized build and a matching-pursuit run
  registered with `--catalog`, and a 6-unit song-structure CSV for that stem.
  Tests that write to the catalog take a copy (`catalog_runs.copy_db`). The API
  tests point `app.config` at the copy and at a one-WAV library.

## Test parameters

The CLI tests don't use the commands' defaults, which are sized for real
recordings. They use one small shared set, sized to the fixtures above, and
then each test changes one option to make a particular path run. Defaults are
listed in [Command-line tools](../../README.md#command-line-tools).

### `randomized-build`: the shared set

`FAST` in `integration/test_randomized_build_cli.py` ("small, fast settings
that still find the planted motifs"):

| Option | Test | Default | Why |
|---|---|---|---|
| `--m-per-outer` | 3 | 5 | The fixture plants 3 shapes; sampling more per pass only adds time |
| `--n-min` | 2 | 10 | Each shape has only 12 copies (4 per window × 3 windows). With 10, no atom survives one short refinement and the build exits 1 |
| `--refine-iters` | 1 | 3 | Clean planted copies converge in one round, and each round is a full MP run |
| `--seed` | 0 | random | Same atoms every run, so exact assertions (stop reason, counts) never flake |
| `--workers` | 1 | all CPUs | No process pool per test; the parallel path has its own test with `--workers 2` |

The catalog fixture (`catalog_setup.BUILD_ARGS`) is `FAST` plus
`--max-outer 2 --min-coverage 1.0`: exactly two passes, so the catalog tests
can assert fixed counts.

### `randomized-build`: forcing each stop reason

| Options | Stop reason | Why it happens |
|---|---|---|
| `--min-coverage 1.0 --max-outer 2` | `max_outer` | Coverage can't reach 100%, so the pass limit ends the run, after exactly 2 passes |
| `--min-coverage 0.05` | `coverage` | Reached after the first pass |
| `--epsilon 0.95 --min-coverage 1.0 --max-outer 20` | `empty_pool` | Every patch must keep over 95% of its energy unexplained, so the pool empties once the first pass explains the planted copies |
| `--max-outer 1` | `max_outer` | Used by tests that only need one pass to check one behaviour |

### `randomized-build`: values that make one path run

| Options | Path |
|---|---|
| `--n-min 1000` | No atom has enough patches: exit code 1 |
| `--post-refine-max-relative-distance 0` | Every placement is dropped after refining: exit code 1 |
| `--post-refine-max-relative-distance none` | The post-refine filter is off |
| `--max-relative-distance 0.5 --n-min 1` | The rebuild gate, with atoms kept even when one patch supports them |
| `--max-relative-distance 1` | A gate that keeps every patch |
| `--n-components 2` | Subspace atoms, shape `(k, F, W)` |
| `--mp-mode global` vs `per_window --workers 2` | Both MP modes, and the worker pool |
| `--n-windows 2 --window-offset 1` | A slice of the windows |
| `--compute cpu` / `cuda` | Each backend (the CUDA test skips without a GPU) |
| `--verbose --loop-stats` | The per-pass log and `loop_stats.json` |
| Invalid values (`--n-windows 0`, `--max-relative-distance 1.5`, `--m-per-outer 0`, …) | Each input check and its message: exit code 2 |

### `matching-pursuit`

Every run gets `--workers 1` (a test about the pool passes `--workers 2`). The
dictionaries come from `BUILD` in `integration/test_matching_pursuit_cli.py`:
`FAST` plus `--max-outer 2 --min-coverage 1.0`, built once per module, and
again with `--n-components 2` for the subspace tests.

| Options | What it checks |
|---|---|
| `--max-relative-distance 0` | A gate that rejects every placement: 0 matchings, stop reason `no_improvement`, in both MP modes; with `--catalog`, the run still registers |
| `--max-repeats 3` (global) | Stops at exactly 3 placements (`max_repeats`) |
| `--max-repeats 2 --disallow-overlap --workers 2` (per window) | The limits inside the worker pool: at most 2 placements per window |
| `--max-relative-distance 0.5 --eps-step-fraction 1e-12` | The step-size stop is effectively off, so per-window MP runs until no placement passes the gate (`completed`); global MP stops on `no_improvement` |
| `--eps-step-fraction 0.001` on the subspace dictionary | Subspace placements carry 2 coefficients each |

### `pan`

`LENGTHS` in `integration/test_pan_cli.py` is `--min-len 16 --max-len 32
--step 8` (defaults 32 / 128 / 32):

| Option | Test | Why |
|---|---|---|
| `--min-len` / `--max-len` / `--step` | 16 / 32 / 8 | The series is 600 samples with a length-24 shape at 40, 220 and 420; 16, 24 and 32 bracket it. The defaults start at 32, above the shape: they still find two of the pairs, but with weaker matches, and their third pair is noise |
| `--top-k` | 3 | The 3 planted copies |
| `--threads` | 2 | With the 3-row series: the multi-row, multithreaded path |

Use `--engine reference` (pure Python) only on series of a few hundred samples.

## The coverage gate

`integration/.coveragerc` measures `analysis.dict_learning`,
`analysis.matrix_profile`, `analysis.bands`, `database` and the backend's
catalog modules (`app.services.catalog_service`, `app.routes.catalog`), with
branch coverage, and fails below 100%. It also counts the `python -m` subprocesses (`patch = subprocess`).
`learn_from_motifs.py` and `refine.py` are left out: they aren't part of the
baseline pipelines.

**What 100% means here:** every line and branch that one of the CLIs or the
catalog API can run. Code the CLIs can't reach is marked in the source, on the line, with the
reason:

```python
if not partition:  # no-pipeline-cover: the CLI rejects an empty detection list
if tid not in self.templates:  # no-pipeline-branch: the CLIs never re-add a template id
```

- `# no-pipeline-cover: <reason>` drops that line from the gate. If the line
  starts a block (`def`, `if`, `else:`, a decorator), the whole block is
  dropped.
- `# no-pipeline-branch: <reason>` keeps the line but doesn't require both
  directions of its branch.

The unit report ignores these markers, so marked code still has to be tested
there.

What gets marked:

- input checks for library callers, where the CLI or API validates first or
  can't pass the bad value;
- library input types the CLIs never use (array templates, lists of windows,
  pre-built segments);
- helpers used only by `learn_from_motifs`, notebooks or unit tests;
- environment fallbacks (no numba, no tqdm, no CuPy, no `sched_getaffinity`);
- the numba-compiled kernels (`_sliding_mean_invnorm`, `_mpx_parallel`,
  `_mpx_channels_parallel`, `_score_shifts_numba`). Coverage can't trace
  compiled code. Unit tests check them against `mpxForPan` and the NumPy
  scorer.

To review them:

```bash
grep -rn "no-pipeline-cover:\|no-pipeline-branch:\|gpu-only" analysis database --include=*.py
```

At the time of writing there are 160 `no-pipeline-cover` and 11
`no-pipeline-branch` markers, in `dict_learning/code/{core,io,mp,tools}.py`,
`learn/randomized_dictionary.py`, `dict_learning/__init__.py`,
`matrix_profile/__init__.py`, `bands.py` and `database/catalog.py`.

**When the gate fails after a change:** add an integration test through a CLI if
the new code is reachable that way. Mark a line only when no CLI input can reach
it, and say why on the line.

### Machines without CUDA

The CUDA path (`--compute cuda`, CuPy shift scoring) is part of the pipeline and
counts on a GPU machine. Its lines are tagged `# gpu-only: …`. Without a GPU,
the CUDA test skips itself, so tell the gate to skip those lines too:

```bash
COVERAGE_EXCLUDE_GPU=gpu-only: COVERAGE_PARTIAL_GPU=gpu-only-branch: \
  python -m pytest analysis/test/integration \
  --cov --cov-config=analysis/test/integration/.coveragerc
```

## Running the CLIs on real data

Point them at a GPL output folder, for example one written by
`python -m analysis.GPL_Python.process_v4 clip.wav --output-dir GPL_OUT`:

```bash
python -m analysis.dict_learning randomized-build --gpl-dir GPL_OUT --out DICT \
  --n-windows 10 --m-per-outer 5 --n-min 10 --min-coverage 0.5 --seed 0 --verbose
python -m analysis.dict_learning matching-pursuit --gpl-dir GPL_OUT --dictionary DICT --out MP
python -m analysis.matrix_profile pan --gpl-dir GPL_OUT --rate 10 --signal bands \
  --min-len 10 --max-len 50 --step 10 --threads 8 --out PAN --plot
```

Each command has `--help`. Every option and its default:
[Command-line tools](../../README.md#command-line-tools) in the root README.
