# analysis/

GPL **analysis** science: reference detection (WAV → whitened spectrograms + detections) and dictionary learning on those products. This tree is **not** “offline only.”

The product’s **offline** path is **preprocessing**: [`backend/preprocess.py`](../backend/preprocess.py) (usually via [`backend/scripts/preprocess_cron.sh`](../backend/scripts/preprocess_cron.sh)), which batch-builds spectrogram / `.gpl` caches for the library. Production interactive detection uses the CUDA-capable fork in [`backend/`](../backend/). FastAPI does not import this package directly.

## Layout

```
analysis/
├── GPL_Python/         # CPU-only Python reference detector (parity baseline)
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
