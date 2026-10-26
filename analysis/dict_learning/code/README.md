# `analysis.dict_learning.code`

Library for Mallat matching pursuit and the randomized dictionary build on GPL
whitened spectrogram windows. Commands, knobs and what was removed:
[`../README.md`](../README.md).

## Tree

```
code/
├── core.py     shared types (Window, Segment, Template, Matching, Dictionary),
│               patch–atom distance, knobs (`defaults`)
├── io.py       load GPL windows and detection CSVs; detections → patches
├── mp.py       matching pursuit: shift scoring, one step, the loop (global and
│               per-window), worker pools, progress bars, loop counters
├── tools.py    matching_pursuit; rebuild atoms from their patches
├── cli.py      randomized-build and matching-pursuit commands
└── learn/
    ├── randomized_dictionary.py   create_dictionary, refine_dictionary, learn_randomized_outer
    ├── refine.py                  refine_dictionary_by_averaging
    └── learn_from_motifs.py       panMP → MP trial (python -m …learn.learn_from_motifs)
```

## Layering

`cli` → `learn` → `tools` → `mp` → `io`, `core`. `learn_from_motifs` also uses
`analysis.matrix_profile` and `analysis.bands`.

`mp.py` concatenates what used to be separate modules (loop counters, progress
helpers, shift scoring, worker pools, the MP step and loop), in that order.

## Coverage markers

Lines ending in `# no-pipeline-cover: <reason>` or
`# no-pipeline-branch: <reason>` are code the CLIs cannot reach (library input
checks, helpers for notebooks or `learn_from_motifs`, environment fallbacks,
numba-compiled kernels). The integration coverage gate skips them; the unit
coverage report does not. See [`../../test/README.md`](../../test/README.md).
