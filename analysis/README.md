# analysis/

GPL **analysis** science: reference detection (WAV → whitened spectrograms + detections) and dictionary learning on those products. This tree is **not** “offline only.”

The product’s **offline** path is **preprocessing**: [`backend/preprocess.py`](../backend/preprocess.py) (usually via [`backend/scripts/preprocess_cron.sh`](../backend/scripts/preprocess_cron.sh)), which batch-builds spectrogram / `.gpl` caches for the library. Production interactive detection uses the CUDA-capable fork in [`backend/`](../backend/). FastAPI does not import this package directly.

## Song analysis pipeline

Three stages. Payne and McVay (1971) supply the hierarchy the model is meant to recover: **subunit < unit < phrase < theme < song**. A song of type A is six themes in a fixed cycle. Phrase-repeat count varies; theme order does not, and no theme is dropped. See [`../papers/payne_mcvay_1971_song_structure.md`](../papers/payne_mcvay_1971_song_structure.md).

A **motif** here is a convolution kernel. A **unit** is a latent state that emits a distribution over those motifs. The labeled `motif` column (`U1`…`U11`) in [`../backend/data/song_structure/`](../backend/data/song_structure/) is a reference annotation, loaded by [`../backend/app/services/song_structure.py`](../backend/app/services/song_structure.py) for display. How the two motif methods are combined, and the exact HMM topology, stay open.

```
whitened sequence → motif dictionary → trained HMM → Viterbi path
```

### Stage 1 — Motif dictionary

**Input.** Whitened window spectrograms in `spectrogram_npy/`, written by [`GPL_Python/gpl_outputs.py`](GPL_Python/gpl_outputs.py).

**Method.** Two existing pieces to combine:

- Matrix Profile over motif length: [`panMP.py`](panMP.py) on [`mpxForPan.py`](mpxForPan.py). Finds repeated subsequences without a fixed length.
- Matching Pursuit: the Mallat loop in [`dict_learning/`](dict_learning/) (seed a patch, average its matches, re-pursue, deflate the residual). See [How `build_dictionary` works](#how-build_dictionary-works) below.

**Combination (sketch).** Use pan-MP peaks to propose candidate kernels (location and length). Use Matching Pursuit to place, average, and deflate them into a small dictionary.

**Output.** A dictionary of convolution kernels, plus the score of each kernel convolved along the sequence.

**Quality.**

- **Coverage.** Fraction of the sequence where at least one motif score is more than 2 standard deviations above that motif’s mean.
- **Specificity.** For each motif, the ratio of the score at a peak to the scores obtained when that motif is shifted by half its length to either side.

Repeat count, residual energy, and time-bin coverage stay internal stopping rules of `build_dictionary` / `extend_dictionary`. The stage score is coverage and specificity above.

### Stage 2 — Train the HMM

**Input.** The motif dictionary and its score sequence from stage 1.

**States.** One hierarchical HMM, trained with Baum–Welch:

- **6 themes** in a cycle (1→2→3→4→5→6→1). Each theme is present; only the number of repeats changes.
- **Phrases** repeat inside a theme. A phrase is a short chain of units.
- **Units** are the emitting states. Each unit has a distribution over motifs.

HMM training is still to be written.

**Output.** Fitted transition and emission parameters, and the posterior over states on the training sequence.

**Quality.**

- **Posterior-weighted likelihood** of the sequence.
- **Parameter sensitivity:** how much that likelihood moves when any single parameter is perturbed.

### Stage 3 — Inference

**Input.** A trained HMM and a new sequence scored with the same motif dictionary.

**Method.** Viterbi: the most likely state path (theme, phrase, unit).

**Output.** That path, aligned to the sequence.

**Quality.**

- **Likelihood** of the sequence.
- **Boundary sensitivity:** how much that likelihood moves when the boundaries between themes are shifted.

## Layout

```
analysis/
├── GPL_Python/         # CPU-only Python reference detector (parity baseline)
├── panMP.py            # pan matrix profile over motif length
├── mpxForPan.py        # matrix profile used by panMP
├── test/               # panMP tests
└── dict_learning/      # Seed / Mallat MP / dictionary build
    ├── code/           # library (core, mp, learn, io, catalog, tools, cli)
    ├── experiments/    # compare / bench harnesses
    ├── scripts/        # LeiData bash runners
    ├── notebooks/      # result / explore notebooks
    └── tests/
```

| Directory                          | Language | Role                                                                                                                   |
| ---------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------- |
| [`GPL_Python/`](GPL_Python/)       | Python   | Reference detector (STFT, whitening, detect, exports); CLI + tests                                                     |
| [`panMP.py`](panMP.py)             | Python   | Length sweep of the matrix profile; motif peaks are candidate kernels                                                 |
| [`dict_learning/`](dict_learning/) | Python   | Dictionary tools: `Seed`, Mallat MP, `build_dictionary` / `extend_dictionary` (design: [`../documentation/Docs/analysis.md`](../documentation/Docs/analysis.md)) |

## Responsibilities

| Subtree          | Owns                                                                 | Does not own                          |
| ---------------- | -------------------------------------------------------------------- | ------------------------------------- |
| `GPL_Python/`    | Reference numerical detector + exports (`CSV_*`, `spectrogram_npy/`) | Web API, CUDA, cron preprocess caches |
| `dict_learning/` | Template seeding, MP, and catalog dictionary builds on those exports | Writing library `result/*.gpl` caches |

## Relationship to the web stack

```
GPL_Python ──port science──► backend/gpl_* (+ CUDA) ──┬──► backend/app/ (interactive)
     │                                                 └──► preprocess.py / cron (offline spectrograms)
     └── CSV + sp_whiten ──► dict_learning (Seed.from_gpl, MP, build_dictionary)
```

When detector math or exports change for correctness, land and validate them in `GPL_Python/`, then port into `backend/` so the web app **and** the preprocess cron pick them up. See [`../documentation/Docs/architecture.md`](../documentation/Docs/architecture.md) (repo layout), [`../documentation/Docs/analysis.md`](../documentation/Docs/analysis.md) (dictionary / peel design), and plans `08` / `10` under [`../documentation/plans/`](../documentation/plans/).

## How `build_dictionary` works

One-shot path: learn a dictionary on a fixed contiguous slice of GPL windows (e.g. all 30 at once), then register templates + matchings in the catalog.

```bash
python -m analysis.dict_learning.code.learn.build_dictionary \
  --db catalog.sqlite --wav /path/to/clip.wav \
  --n-windows 30 --n-rounds 4 \
  --eps-step-fraction 0.2 --min-repeats 3 \
  --min-template-width 12 --max-template-width 24 \
  --refine-iters 2 --out-dir …
```

Shell example: [`dict_learning/scripts/run_build_dictionary_leidata_wav1.sh`](dict_learning/scripts/run_build_dictionary_leidata_wav1.sh).

### What it does

1. **Load** the WAV’s indexed GPL segmentation from the catalog: whitened window `.npy` files + detection CSV.
2. **Select** `window_offset` … `window_offset + n_windows` (mean-center each window once).
3. **Learn** with `learn_averaged_templates_from_gpl` on that full stack — residual deflation, one new atom per outer round.
4. Optionally **refine** all atoms (`--refine-iters` average→MP passes).
5. **Write** `template_*.npy` + `manifest.json` under a UTC-stamped out dir and register `tool_runs.tool='orchestrated_build'`.

### Outer loop (one atom per round)

On a deflating residual `R` (starts as the mean-centered stack):

```text
R ← mean_center(windows)
loop until stop:
    candidates ← crop width-eligible GPL patches from R
    for each candidate U:                    # often the expensive nest
        seed ← Mallat MP with {U} only       # tool A
        if seed matches < min_repeats: skip
        T̄ ← average seed's matched patches  # tool B
        avg ← Mallat MP with {T̄} only       # tool A again
        if avg matches < min_repeats: skip
        keep best avg by lowest residual energy
    if none qualify: stop
    append T̄; R ← avg residual               # next round sees what's left
    if n_rounds reached: stop
```

- **Seed:** which GPL patch, used alone, best explains `R` and repeats enough times?
- **Average:** consensus shape over that seed’s match locations.
- **Re-MP:** place the averaged template; its matchings are the catalog “repeats”; residual energy left is the next round’s input.

Matchings are **non-overlapping** within a window. `--n-rounds` is an optional cap; omit it to grow until no candidate meets `--min-repeats`. Candidate width is filtered by `--min-template-width` / `--max-template-width`.

Most wall time is the **candidate trial nest** (every eligible patch × seed MP + avg MP each round), not a single final MP.

### Extend alternative

[`extend_dictionary`](dict_learning/code/learn/extend_dictionary.py) uses the **same** grow engine in a **single stage**: peel an optional parent → while time-bin coverage < 0.9, add one atom and refine all. Target windows may come from a different WAV than the parent. See [`../documentation/Docs/extend-dictionary.md`](../documentation/Docs/extend-dictionary.md).

Full algorithm: [`../documentation/Docs/dictionary-from-scratch.md`](../documentation/Docs/dictionary-from-scratch.md). Operator flags: [`../documentation/Docs/users-guide.md`](../documentation/Docs/users-guide.md). Package details: [`dict_learning/README.md`](dict_learning/README.md).

## Setup

From the repo root, into your Python env (editable install — code changes
under `analysis/` take effect immediately, no reinstall needed):

```bash
pip install -e .            # analysis.* + FastAPI/uvicorn stack (CPU)
pip install -e '.[gpu]'     # + cupy / nvidia-* for CUDA (GPU servers only)
```

All runtime deps (science + webserver) live in the root [`pyproject.toml`](../pyproject.toml); GPU stays optional via `[gpu]`.

## Quick start

```bash
# From repo root — run GPL_Python (module invocation: internal imports are
# relative, so it can no longer be run as a bare script path)
python3 -m analysis.GPL_Python.process_v4 /path/to/input.wav \
  --output-dir GPL_Python_outputs

# Select one template: trial all GPL candidates, keep min residual among those with enough repeats
python3 -m analysis.dict_learning.code.cli seed-from-gpl \
  --npy-dir GPL_Python_outputs/spectrogram_npy \
  --gpl-output-dir GPL_Python_outputs \
  --eps-step-fraction 0.5 \
  --min-repeats 3 \
  --out seed_out

# Mallat MP using the selected template
python3 -m analysis.dict_learning.code.cli matching-pursuit \
  --npy-dir GPL_Python_outputs/spectrogram_npy \
  --seed-manifest seed_out/seed_manifest.json \
  --eps-step-fraction 0.5 \
  --out mp_out

# Full dictionary on a catalog WAV (fixed window slice — not incremental)
# Requires WAV indexed in --db (preprocess or database.index_existing).
python3 -m analysis.dict_learning.code.learn.build_dictionary \
  --db /path/to/catalog.sqlite --wav /path/to/clip.wav \
  --n-windows 30 --n-rounds 4 --out-dir dictionaries/clip
```

MATLAB parity helpers (optional): see [`GPL_Python/README.md`](GPL_Python/README.md).
