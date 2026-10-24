# Analysis

Two stages, from a recording to repeated calls.

1. **Find motifs.** `matrix_profile` finds repeated stretches without a template. The search runs on the whitened spectrogram from `GPL_Python`, not on the raw samples.
2. **Learn templates.** `dict_learning` turns those motif rectangles into seed atoms, then matching pursuit places each atom and removes it from the residual.

```
WAV → GPL whitened spectrogram → matrix profile → motif rectangles → seed atoms → dictionary
```

Both stages use the same rectangle: the spectrogram's frequency rows by the motif's time span.

## Layout

```
analysis/
├── bands.py                 five bands of 100 Hz, 0–500 Hz
├── matrix_profile/          matrix profile and motif pairs
├── GPL_Python/              whitened spectrogram and detections
│   ├── gpl_signal.py
│   ├── gpl_detector.py
│   ├── gpl_outputs.py
│   ├── process_v4.py
│   ├── gpl_parity.py
│   ├── decimate_wav.py
│   └── preprocess.py
├── dict_learning/
│   ├── code/
│   │   ├── core.py          windows, atoms, patch distance
│   │   ├── io.py            load and stitch GPL spectrograms
│   │   ├── mp.py            matching pursuit
│   │   ├── tools.py         average, add, remove, merge
│   │   ├── cli.py
│   │   ├── catalog.py       repeats payload for the viewer
│   │   └── learn/           grow a dictionary
│   ├── experiments/
│   ├── notebooks/
│   └── tests/
└── test/
```

Inside `dict_learning/code`, `learn` and `cli` use `tools`, `tools` uses `mp`, and `mp` uses `io` and `core`. `learn_from_motifs` turns a motif rectangle into a seed atom.

## Modules

### Top level

- `bands.py` — energy in five 100 Hz bands from 0 to 500 Hz.
- `matrix_profile/__init__.py` — matrix profile for one series or many frequency rows, and the pairs that do not overlap.

### `GPL_Python/`

- `gpl_signal.py` — read a WAV, take its STFT, and whiten the noise floor.
- `gpl_detector.py` — find calls in one whitened window.
- `gpl_outputs.py` — write and load the whitened spectrogram, the display panel, and the detection tables.
- `process_v4.py` — run the detector on each window of a recording.
- `gpl_parity.py` — compare the Python detector with the MATLAB export.
- `decimate_wav.py` — resample a 200 kHz WAV to 10 kHz.
- `preprocess.py` — run detection, then register an averaged-template dictionary in the catalog.

### `dict_learning/`

- `__init__.py` — public exports. The heavier learn modules load on first use.
- `__main__.py` — `python -m analysis.dict_learning` runs the tool CLI.
- `code/core.py` — windows, templates, matchings, and how much of a patch an atom leaves unexplained.
- `code/io.py` — load GPL windows and stitch the whitened spectrogram into one array.
- `code/mp.py` — place dictionary atoms on a spectrogram and subtract them.
- `code/tools.py` — average matched patches, and add, remove, or merge atoms.
- `code/cli.py` — command line for those tools.
- `code/catalog.py` — repeat-match payload the viewer reads from the catalog.
- `code/learn/learn_from_motifs.py` — grow a dictionary from 5 s and 10 s motifs until half the frames are covered.
- `code/learn/seed_from_gpl.py` — keep the GPL detection whose single atom leaves the smallest residual.
- `code/learn/learn_dictionary.py` — grow a dictionary by deflating those seeds, then average each atom's patches.
- `code/learn/build_dictionary.py` — run that learner for one WAV and register the dictionary.
- `code/learn/extend_dictionary.py` — add atoms until a coverage target.
- `code/learn/match_to_dictionary.py` — rebuild atoms from a labeled matching.
- `code/learn/randomized_dictionary.py` — sample unexplained patches as atoms and refine until coverage.
- `code/learn/split_template.py` — split one atom's patches into two atoms.
- `code/learn/similarity.py` — compare atoms by the patches they explain.

## Motif signal

`GPL_Python` writes one whitened spectrogram per 75 s window (`spectrogram_npy/spwhiten_*.npy`). `stitch_whitened_spectrogram` joins those windows in time. A column is one STFT hop (`overlap` samples).

The matrix profile runs on that `(frequency, time)` array. `mpx_channels` measures a pair by the sum of the z-normalized distances of the frequency rows. `minlag` equals the motif length, so the two copies start at least one motif length apart. `closest_pairs` skips a later pair when either copy overlaps a pair already chosen.

The seed atom is the mean of the two rectangles, with one frequency row per spectrogram bin.

## Run

From the repo root, with this directory on `PYTHONPATH`:

```bash
python -m pytest analysis/test/test_mpxForPan.py analysis/test/test_motifs.py -q
python -m analysis.dict_learning.code.learn.build_dictionary --help
```

Dictionary commands and knobs: `dict_learning/README.md`.
