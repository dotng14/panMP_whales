# Pipeline scripts — SOCAL_H_76_230427_014845.x.wav

Four shell scripts, one per command, that run the GPL → dictionary → matching
pursuit / pan matrix profile pipeline on one recording on **ensemble**
(`ensemble.ucsd.edu`).

## Files

```
analysis/scripts/SOCAL_H_76_230427_014845/
├── README.md                    this document
├── config.sh                    shared paths + helpers (sourced, not run)
├── 01_gpl.sh                    step 1: GPL detector
├── 02_randomized_build.sh       step 2: randomized dictionary build
├── 03_matching_pursuit.sh       step 3: matching pursuit with the dictionary
└── 04_pan_matrix_profile.sh     step 4: pan matrix profile
```

## Steps and the commands they run

| Script | Command | Reads | Writes |
|---|---|---|---|
| `01_gpl.sh` | `python -m analysis.GPL_Python.process_v4 $WAV --output-dir $GPL_DIR` | WAV | `gpl/` |
| `02_randomized_build.sh` | `python -m analysis.dict_learning randomized-build --gpl-dir $GPL_DIR --out $DICT_DIR --m-per-outer 5 --n-min 10 --min-coverage 0.5 --seed 0 --verbose` | `gpl/` | `dictionary/` |
| `03_matching_pursuit.sh` | `python -m analysis.dict_learning matching-pursuit --gpl-dir $GPL_DIR --dictionary $DICT_DIR --out $MP_DIR` | `gpl/`, `dictionary/` | `matching_pursuit/` |
| `04_pan_matrix_profile.sh` | `python -m analysis.matrix_profile pan --gpl-dir $GPL_DIR --rate 10 --signal bands --min-len 10 --max-len 50 --step 10 --out $PAN_DIR --plot` | `gpl/` | `pan_mp/` |

Dependencies: 1 → 2 → 3, and 1 → 4. Step 4 does not need steps 2 or 3, so it
can run in parallel with them once step 1 has finished. Each script checks
that its input folders exist and stops with an error if they don't.

## Paths on ensemble (set in `config.sh`)

| Variable | Value |
|---|---|
| `REPO` | the checkout containing these scripts (auto-detected) |
| `PYTHON` | `/home/rivo/.pyenv/versions/3.13.5/envs/dev/bin/python` |
| `WAV` | `/home/yfreund/whale_processing/LeiData/SOCAL_H_76_230427_014845.x.wav` (200 kHz, 75 min) |
| `OUT_ROOT` | `/home/yfreund/whale_processing/result/LeiData/dat_pipeline/SOCAL_H_76_230427_014845.x` |
| `GPL_DIR` | `$OUT_ROOT/gpl` |
| `DICT_DIR` | `$OUT_ROOT/dictionary` |
| `MP_DIR` | `$OUT_ROOT/matching_pursuit` |
| `PAN_DIR` | `$OUT_ROOT/pan_mp` |
| `LOG_DIR` | `$OUT_ROOT/logs` |

Output tree after all four steps:

```
/home/yfreund/whale_processing/result/LeiData/dat_pipeline/SOCAL_H_76_230427_014845.x/
├── gpl/
│   ├── CSV_SOCAL_H_76_230427_014845.x.csv         detections (start/end per window)
│   ├── Detections_SOCAL_H_76_230427_014845.x.mat
│   ├── Labels_SOCAL_H_76_230427_014845.x.tlab
│   └── spectrogram_npy/                            30 whitened 75 s windows (.npy + .json sidecar each)
├── dictionary/
│   ├── atoms/template_*.npy                        learned atoms
│   ├── summary.json                                atom list, stop reason, coverage
│   ├── matchings.json                              placements found while learning
│   └── outer_logs.jsonl                            one line per outer pass
├── matching_pursuit/
│   ├── summary.json
│   └── matchings.json                              final placements of every atom
├── pan_mp/
│   ├── pan.npy                                     motif length × time surface
│   ├── pan.json                                    lengths + how the signal was built
│   ├── pairs.json                                  top motif pairs across lengths
│   └── pan.png                                     plot of pan.npy
└── logs/
    └── 01_gpl.log, 02_randomized_build.log, 03_matching_pursuit.log, 04_pan_matrix_profile.log
```

The pipeline writes into its own `dat_pipeline/` folder so it never touches the
existing `result/LeiData/SOCAL_H_76_230427_014845.x.gpl/` that the webserver
uses. If you want to skip step 1 and reuse that older GPL output, run steps 2–4
with `GPL_DIR=/home/yfreund/whale_processing/result/LeiData/SOCAL_H_76_230427_014845.x.gpl/output`.

## Running on ensemble

From the repository root of a checkout of this branch (`dat-script`):

```bash
bash analysis/scripts/SOCAL_H_76_230427_014845/01_gpl.sh
bash analysis/scripts/SOCAL_H_76_230427_014845/02_randomized_build.sh
bash analysis/scripts/SOCAL_H_76_230427_014845/03_matching_pursuit.sh
bash analysis/scripts/SOCAL_H_76_230427_014845/04_pan_matrix_profile.sh
```

Steps 1 and 2 take a long time (the earlier randomized-build on this file ran
about 1.5 h), so run them detached:

```bash
nohup bash analysis/scripts/SOCAL_H_76_230427_014845/02_randomized_build.sh > /dev/null 2>&1 &
```

The output still goes to `$LOG_DIR/<step>.log`; each run appends a start line,
the full program output and an `exit <code>` line.

### Overrides

- **Paths:** export any `config.sh` variable first, e.g.
  `OUT_ROOT=/tmp/try1 bash .../01_gpl.sh`.
- **Extra flags:** anything after the script name is passed to the Python
  command, e.g. `bash .../02_randomized_build.sh --workers 16 --compute cuda`
  or `bash .../04_pan_matrix_profile.sh --threads 16`.

## How a script is put together

Every step script has the same three parts:

1. `source config.sh` — sets the paths above, creates `OUT_ROOT` and `LOG_DIR`,
   and `cd`s to the repo root so `python -m analysis....` resolves.
2. `require_dir` checks for the folders produced by earlier steps.
3. `run_logged <name> -m <module> ...` runs the command with `$PYTHON` and tees
   output to `$LOG_DIR/<name>.log`, returning the program's exit code.
