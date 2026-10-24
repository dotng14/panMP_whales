# `analysis.dict_learning.code`

Library implementation for dictionary learning on GPL whitened spectrogram windows.

Parent package layout (experiments, scripts, notebooks, tests): see [`../README.md`](../README.md).  
Design overview: [`../../../documentation/Docs/analysis.md`](../../../documentation/Docs/analysis.md).

## Tree

```
code/
├── core/       # shared types, knobs, helpers
├── io/         # load GPL windows + detection CSVs
├── mp/         # matching pursuit (scoring, Mallat loop, parallelism)
├── tools.py    # elementary tools A–E (MP, average/PCA, edit)
├── learn/      # seed → grow → build / extend / match / split
├── catalog/    # frontend repeats payload from catalog rows
└── cli.py      # argparse entry for tools A–E (+ seed, match-to-dictionary)
```

Import from these paths directly, e.g.:

```python
from analysis.dict_learning.code.core import Template, Matching
from analysis.dict_learning.code.core import defaults
from analysis.dict_learning.code.learn.build_dictionary import build_dictionary_for_wav
```

```bash
python -m analysis.dict_learning.code.learn.build_dictionary --help
python -m analysis.dict_learning.code.cli --help
python -m analysis.dict_learning   # same as code.cli
```

The package root `__init__.py` re-exports a small public API for convenience; prefer `code.*` for new callers.

## Layering

Dependencies flow **down** the stack (higher layers may import lower; not the reverse):

```text
catalog, cli, learn
        ↓
     tools.py
        ↓
        mp
        ↓
    io , core
```

| Layer | Package / file | Responsibility |
|---|---|---|
| **core** | `core.py` | Domain types (`Window`, `Segment`, `Template`, `Matching`, `Dictionary`), shared knobs (`defaults`), coverage / residual / distance helpers. |
| **io** | `io.py` | Read whitened `.npy` windows + GPL detection CSVs, and stitch the whitened spectrogram. |
| **mp** | `mp.py` | Shift scoring, greedy and exact pursuit, worker pools, subspace wrappers, progress and loop timing. |
| **tools** | `tools.py` | Elementary operators **A–E** used by learning and the CLI (see below). |
| **learn** | `learn/` | How dictionaries are grown and curated: seed selection, outer seed→average→MP rounds, catalog build, coverage-gated extend, labeled match-to-dictionary, 1→2 template split, atom comparison. |
| **catalog** | `catalog.py` | Build the web “repeats” JSON from catalog templates + `panel.bin` (used by the backend). |
| **cli** | `cli.py` | Thin argparse front-end over seed + tools A–E + `match-to-dictionary`. |

## Modules

`core.py` holds the dataclasses, the shared knobs, and the patch-distance helpers. `io.py` loads GPL windows and stitches the whitened spectrogram. `mp.py` places atoms and subtracts them: shift scores, the greedy loop, the exact non-overlapping loop, worker pools, and the older subspace names.

### `tools.py` (elementary A–E)

| Tool | Function(s) | Meaning |
|---|---|---|
| **A** | `matching_pursuit` | Mallat MP of a dictionary on a residual stack. |
| **B** | `average_templates` / `pca_*` / `pca_basis` | Rebuild atoms from matched patches (mean or PCA / subspace). |
| **C** | `build_dictionary_from_detections` | Greedy single-atom growth from GPL detections. |
| **D** | `remove_template` | Drop an atom (new dictionary). |
| **E** | `merge_templates` | Merge two atoms’ detections into one. |

Also: `rebuild_template_from_matchings`, optional catalog persistence for `n_components=1`.

### `learn/`

| File | Role |
|---|---|
| `seed_from_gpl.py` | Trial GPL patches with single-template MP; pick the best repeat-qualified seed (`Seed`). |
| `learn_dictionary.py` | Outer rounds: seed → average/PCA → re-MP on a deflating residual; refine-by-averaging; round logs. |
| `build_dictionary.py` | Catalog WAV → load windows → `learn_*` → optional refine → write `.npy` / manifest → register (`k=1`) or manifest-only (`k>1`). |
| `extend_dictionary.py` | Single stage: peel parent (optional) → while time coverage < 0.9: add one atom → end refine. Cross-WAV windows; optional pinned matchings. |
| `match_to_dictionary.py` | Build dictionary from labeled matchings + explicit widths; optional pinned average→MP refine. |
| `split_template.py` | Randomized 1→2 rank-1 bipartition of one atom’s patches (`split_one_template`, `split_template_matchings`). |
| `similarity.py` | Compare atoms by residual-energy divergence, response correlation, and group-mean similarity. |
| `learn_from_motifs.py` | Grow a dictionary from matrix-profile motifs until half the frames are covered. Also builds seeds from the 5-band energy grid. |
| `randomized_dictionary.py` | Sample unexplained patches as atoms and refine until coverage. |

### `catalog.py`

`build_repeats_payload_from_catalog` for the frontend overlay API.

### `cli.py`

Subcommands: `seed-from-gpl`, `matching-pursuit`, `average-templates`,
`match-to-dictionary`, `build-from-detections`, `remove-template`,
`merge-templates`.

Production builds usually go through `learn.build_dictionary` /
`learn.extend_dictionary` rather than this CLI. Labeled rebuilds use
`learn.match_to_dictionary` (also available here as a subcommand).

## Typical call paths

**One-shot dictionary (catalog WAV):**

```text
build_dictionary_for_wav
  → load windows/detections (io + database.catalog)
  → learn_averaged_templates_from_gpl
       → seed trials (seed_from_gpl / mp_workers + mp_step)
       → rebuild atom (tools: average / pca_basis)
       → re-MP (tools.matching_pursuit → mp_step)
  → optional refine_dictionary_by_averaging
  → write templates + register
```

**Extend (coverage-gated grow):**

```text
extend_dictionary_on_windows
  → matching_pursuit peel of existing atoms (optional)
  → while time_coverage < min_coverage:
       learn_averaged_templates_from_gpl (n_rounds=1) on residual
       → joint peel + end refine on full stack
```

**Match-to-dictionary / reviewed rebuild:**

```text
matchings + widths → average_templates → optional pinned refine → catalog
```

**Split atom:**

```text
patches of one atom → split_one_template (±ε trials)
  → optional split_replace_refine (remap + match_to_dictionary + register source=edit)
```

**Atom divergences:**

```text
joint MP → R
  for each A: R[-A] = inflate matches of A
  if ΔA large: for each B: MP(B on R[-A]) → D(A,B)
```

## Related docs

| Doc | Topic |
|---|---|
| [`../README.md`](../README.md) | Package overview, knobs, API examples |
| [`../../../documentation/Docs/dictionary-operations.md`](../../../documentation/Docs/dictionary-operations.md) | Ops map |
| [`../../../documentation/Docs/extend-dictionary.md`](../../../documentation/Docs/extend-dictionary.md) | Extend stage / coverage loop |
| [`../../../documentation/Docs/match-to-dictionary.md`](../../../documentation/Docs/match-to-dictionary.md) | Labeled rebuild |
| [`../../../documentation/Docs/split-template.md`](../../../documentation/Docs/split-template.md) | 1→2 split |
| [`../../../documentation/Docs/subspace-mp-math.md`](../../../documentation/Docs/subspace-mp-math.md) | Subspace MP math |
| [`../../../documentation/Docs/dictionary-from-scratch.md`](../../../documentation/Docs/dictionary-from-scratch.md) | Grow-from-empty algorithm |
