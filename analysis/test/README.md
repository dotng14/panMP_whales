# analysis/test

Tests for the three baseline pipelines and the code under them:

| Pipeline | Command |
|---|---|
| Randomized dictionary build | `python -m analysis.dict_learning randomized-build` |
| Mallat matching pursuit with a saved dictionary | `python -m analysis.dict_learning matching-pursuit` |
| Pan matrix profile | `python -m analysis.matrix_profile pan` |

There are two layers:

| Layer | Folder | Drives the code through | Coverage |
|---|---|---|---|
| **Unit** | `unit/` | Direct function calls | Reported, not enforced. Aim: as high as practical per function. |
| **Integration** | `integration/` | The CLIs: `main(argv)` in-process, plus one real `python -m` subprocess per command | **100% of lines and branches the CLIs can run**, enforced |

## Layout

```
analysis/test/
├── README.md
├── conftest.py          shared fixtures (session-scoped, written to pytest's tmp dir)
├── synthetic.py         builders for synthetic GPL folders, series, WAVs; run_module()
├── unit/
│   ├── .coveragerc      unit coverage report
│   ├── dict_learning/   MP, refine, randomized build, learn_from_motifs, bands
│   └── matrix_profile/  mpxForPan, panMP, pan_profile, signals, MATLAB parity
└── integration/
    ├── .coveragerc      the 100% gate
    ├── test_randomized_build_cli.py
    ├── test_matching_pursuit_cli.py
    └── test_pan_cli.py
```

`GPL_Python/tests/` and `HMM/test_soft_hmm.py` are separate suites and stay
where they are.

## Setup

Python 3.11 or newer (developed on 3.13). From the repository root, in a
virtual environment **used only for this checkout**:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'          # numpy, scipy, matplotlib, numba, tqdm, pytest, pytest-cov, …
pip install -e '.[test,gpu]'      # also CuPy, to run the CUDA test
```

Don't install this repo into an environment that another checkout is also
installed into, such as the production webserver's. Both checkouts provide a
top-level `analysis` package, and the last install wins for every program using
that environment.

You can also skip the install: `pyproject.toml` tells pytest to put the
repository root first on `sys.path` (`pythonpath = ["."]`), so the tests always
import this checkout's `analysis` package. You then only need the packages
listed in `pyproject.toml`, plus `pytest` and `pytest-cov`.

## Running

Always run from the repository root.

```bash
# Everything under analysis/ (unit + integration + GPL_Python + HMM)
python -m pytest analysis

# One layer
python -m pytest analysis/test/unit
python -m pytest analysis/test/integration

# The coverage gate (fails below 100%)
python -m pytest analysis/test/integration \
  --cov --cov-config=analysis/test/integration/.coveragerc

# The unit coverage report
python -m pytest analysis/test/unit \
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
| All of `analysis/`, cold numba cache | about 9 s |
| All of `analysis/`, warm numba cache | about 6 s |
| Integration suite with the coverage gate | about 7 s |

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

Test parameters are kept small for speed: 1–3 outer passes, 1 refine
iteration, fixed `--seed 0`, `--workers 1` unless the test is about workers,
and short motif lengths. Use the reference panMP engine (pure Python) only on
series of a few hundred samples.

## The coverage gate

`integration/.coveragerc` measures `analysis.dict_learning`,
`analysis.matrix_profile` and `analysis.bands`, with branch coverage, and fails
below 100%. It also counts the `python -m` subprocesses (`patch = subprocess`).
`learn_from_motifs.py` and `refine.py` are left out: they aren't part of the
baseline pipelines.

**What 100% means here:** every line and branch that one of the three CLIs can
run. Code the CLIs can't reach is marked in the source, on the line, with the
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

- input checks for library callers, where the CLI validates first or can't pass
  the bad value;
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
grep -rn "no-pipeline-cover:\|no-pipeline-branch:\|gpu-only" analysis --include=*.py
```

At the time of writing there are 153 `no-pipeline-cover` and 11
`no-pipeline-branch` markers, in `dict_learning/code/{core,io,mp,tools}.py`,
`learn/randomized_dictionary.py`, `dict_learning/__init__.py`,
`matrix_profile/__init__.py` and `bands.py`.

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

Each command has `--help`. Full flag tables: `dict_learning/README.md` and the
`pan` section of `analysis/README.md`.
