# Pipeline scripts

One script per command, for any recording, with the options set. Run them
from anywhere; they `cd` to the repository root.

| Script | Command | Reads | Writes |
|---|---|---|---|
| `01_gpl.sh` | `process_v4 $WAV --freq-lo 10 --freq-hi 10000` | `$WAV` | `gpl/` |
| `02_randomized_build.sh` | `randomized-build --m-per-outer 5 --n-min 10 --max-outer 10 --min-coverage 0.2 --refine-iters 1 --seed 0` | `gpl/` (with detections) | `randomized_build/` |
| `02_pan_build.sh` | `pan-build --signal pca --pca-rows 8 --lengths-s 1,2,3,4`, plus the same outer-loop options | `gpl/` | `pan_build/` |
| `03_matching_pursuit.sh` | `matching-pursuit` with the `SOURCE` dictionary | `gpl/`, `pan_build/` or `randomized_build/` | `matching_pursuit_<source>/` |
| `04_pan_matrix_profile.sh` | `pan --rate 10 --signal bands --min-len 10 --max-len 50 --step 10 --top-k 10 --plot` | `gpl/` | `pan_mp/` |
| `05_split.sh TEMPLATE_ID` | `split --epsilon 0.1 --n-trials 10 --max-refine 10 --min-relative-reduction 0.1 --seed 0` | `gpl/`, the `SOURCE` dictionary, and `matching_pursuit_<source>/` if it exists | `split_<source>_t<id>/` |
| `06_register_catalog.sh` | `python -m database register` on every output folder | the folders above | `$CATALOG` |
| `run_randomized_pipeline.sh` | 01 → 02 randomized → 03 → 04 | `$WAV` | as above |
| `run_pan_pipeline.sh` | 01 → 02 pan → 03 → 04 | `$WAV` | as above |

The two `run_*` scripts are the two end-to-end runs in the root README. Steps
1 → 2 → 3 → 5 depend on each other in that order; step 4 needs only step 1.
Each script stops with exit 2 when a folder it needs is missing. Exit 1 from a
build means no atom survived, and from a split that the split was rejected
(its `summary.json` gives the reduction).

## Why these options

- **GPL 10 Hz – 10 kHz:** the `process_v4` defaults, and the range used for
  the LeiData recordings so far.
- **Builds:** 5 seeds a pass, an atom needs 10 placements, at most 10 passes,
  stop at 20% of the time bins covered. These are the command defaults; the
  only addition is `--seed 0` for randomized-build, so reruns match.
- **pan-build:** 1–4 s motifs on 8 PCA rows of the spectrogram, the defaults.
  It needs no detections, so it also works where GPL detects poorly.
- **Pan matrix profile:** 1–5 s motifs on 100 Hz band energy at 10 Hz, as the
  HMM uses.
- **Split:** gplrev3's defaults, plus `--seed 0`.
- **CPU:** `WORKERS=4` and `PAN_THREADS=4`. The server is shared, and the
  commands otherwise default to every CPU.

## Settings

Export any of these before running a script (all are in `config.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `WAV` | (none) | The recording. Needed for step 1; with `GPL_DIR` set, the other steps don't need it |
| `GPL_DIR` | `$OUT_ROOT/gpl` | GPL output folder. Point it at existing GPL output to skip step 1 |
| `OUT_ROOT` | `~/whale_runs/<stem>` | Where every step writes |
| `SOURCE` | `pan` | Which dictionary steps 3 and 5 use: `pan` or `randomized` |
| `N_WINDOWS` | all | Use only the first N GPL windows in steps 2–5, for a quick look |
| `WORKERS`, `PAN_THREADS` | 4, 4 | Matching-pursuit workers and matrix-profile threads |
| `CATALOG` | (none) | A catalog SQLite file; when set, steps 2, 3 and 5 register what they write |
| `PYTHON` | `python` | The Python with the project's requirements |

Anything after the script name goes to the command and overrides the options
above (the last value of a flag wins):

```bash
WAV=/data/rec.wav bash analysis/scripts/pipeline/run_pan_pipeline.sh
GPL_DIR=/data/result/rec.gpl/output N_WINDOWS=4 bash analysis/scripts/pipeline/02_pan_build.sh --lengths-s 2,3,4
SOURCE=pan bash analysis/scripts/pipeline/05_split.sh 3 --min-relative-reduction 0.2
CATALOG=backend/data/catalog.sqlite bash analysis/scripts/pipeline/06_register_catalog.sh --label "first try"
```

Each step appends its command, its full output and its exit code to
`$OUT_ROOT/logs/<step>.log`. Long steps can run detached:

```bash
WAV=/data/rec.wav nohup bash analysis/scripts/pipeline/run_pan_pipeline.sh > /dev/null 2>&1 &
```

The 014845 scripts in [`../SOCAL_H_76_230427_014845/`](../SOCAL_H_76_230427_014845/README.md)
run the randomized pipeline for that one recording, into its `dat_pipeline/`
folder.
