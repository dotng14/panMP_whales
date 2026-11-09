# Files the tools read and write

Every folder and file the command-line tools read or create, in pipeline
order, with what is inside and which command uses it. How to run each command
is in the root [`README.md`](README.md). Ready-made scripts that run the
pipeline are in [`analysis/scripts/pipeline/`](analysis/scripts/pipeline/README.md).

```
recording.wav                                  input
GPL_OUT/                                       1. process_v4
├── CSV_<stem>.csv                             detections
├── Detections_<stem>.mat, Labels_<stem>.tlab  the same detections for MATLAB / Triton
└── spectrogram_npy/
    ├── spwhiten_<stem>_w<k>_s<a>_e<b>.npy     whitened spectrogram of window k
    ├── spwhiten_<stem>_w<k>_s<a>_e<b>.json    its sidecar
    └── spwhiten_<stem>_w<k>_s<a>_e<b>_display.npy
DICT/                                          2. randomized-build, pan-build (and split)
├── atoms/template_<id>.npy
├── summary.json
├── matchings.json
├── outer_logs.jsonl                           builds only
├── pan_seeds.jsonl                            pan-build only
└── loop_stats.json                            with --loop-stats
MP/                                            3. matching-pursuit
├── summary.json
├── matchings.json
└── loop_stats.json                            with --loop-stats
PAN/                                           matrix_profile pan
├── pan.npy, pan.json, pairs.json
└── pan.png                                    with --plot
catalog.sqlite                                 python -m database, --catalog
```

## 1. Input: the recording

A WAV file, mono, any sample rate (the LeiData files are 200 kHz, the HMM
training file 10 kHz). `process_v4` takes one file or a folder of them. The
stem (file name without `.wav`) names everything GPL writes.

## 2. GPL output folder (`process_v4 --output-dir GPL_OUT`)

| File | Format | Read by |
|---|---|---|
| `spectrogram_npy/spwhiten_<stem>_w<k>_s<a>_e<b>.npy` | `float32` array `(F, T)`: the whitened spectrogram of window `k`, which covers samples `a`–`b` (1-based) of the WAV. `F` frequency rows from `start_freq` to `end_freq`, `T` STFT columns, one per `overlap` samples. | Every analysis command |
| `spectrogram_npy/…<same>.json` | Sidecar: `rows`, `cols`, `sample_freq`, `fftl`, `overlap` (hop in samples), `start_freq` / `end_freq` (Hz), `start_sample` / `end_sample`, `window_length_sec`, `noise_ceiling`, `thresh`, and the intensity clip and display settings | Every analysis command (window order, time per column, frequency range) |
| `spectrogram_npy/…_display.npy` | The display copy of the same window (clipped, cube-rooted) | The web viewer only |
| `CSV_<stem>.csv` | One row per detection: `start_time`, `end_time` (1-based samples in the WAV), `julian_start_time`, `julian_end_time`, `start_freq_hz`, `end_freq_hz`, `min_freq_hz`, `max_freq_hz`, `boundary_cross_flag`, `fname` | `randomized-build` (its seed crops); the catalog |
| `Detections_<stem>.mat`, `Labels_<stem>.tlab` | The detections in MATLAB and Triton label format | Not by these tools |
| `panel.bin`, `manifest.json`, `repeats.json` | Caches the web viewer adds to a GPL folder it shows | The web viewer only |

On 014845 (75 min at 200 kHz) GPL writes 30 windows of 75 s, each 1637 × 609.

## 3. Dictionary folder

Written by `randomized-build`, `pan-build` and `split`, in one layout, so
`matching-pursuit`, `split` and the catalog read any of them.

| File | Format |
|---|---|
| `atoms/template_<id>.npy` | `float64` atom: `(F, W)` (rank 1, unit norm) or `(k, F, W)` (a `k`-dimensional subspace, `--n-components k`). `F` matches the GPL windows; `W` is its width in columns. |
| `summary.json` | `command` (`randomized-build`, `pan-build` or `split`), `gpl_dir`, `n_windows`, `window_ids`, `window_offset`, `sec_per_bin`, every option of the run, `stop_reason`, `n_atoms`, `n_matchings`, `time_coverage_fraction`, `window_coverage_fraction`, `energy_before`, `energy_after`, and `atoms`: one entry per atom with `template_id`, `width`, `n_freq`, `n_components`, `n_patches` (its placements), `source_segment_id` (where it came from) and `path`. |
| `matchings.json` | List of placements: `template_id`, `window_id`, `t_start` (column in that window), `amplitude`, `energy`, and `coeffs` for subspace atoms. These are the placements the atom was learned from. |
| (in `summary.json`) | Builds also record `overlap` (and its two limits when `limited`) and `init_dictionary`: the folder a second-stage build started from, or `null`. |
| `outer_logs.jsonl` | Builds only. One JSON line per outer pass: `outer`, `pool_size`, `n_sampled`, `n_atoms_this_pass`, `n_matchings_this_pass`, `n_pruned`, `n_distance_filtered`, `n_merged`, running `n_atoms` / `n_matchings`, time and window coverage, `energy_before` / `energy_after`. |
| `pan_seeds.jsonl` | `pan-build` only. One line per pass: the search signal's `signal_rows` / `signal_columns`, `n_pairs_tried`, `n_masked`, `n_too_few_copies`, `n_duplicates`, and `picks` (each seed's pair, distance, z-score and copies as `(window_id, t_start)`). |
| `loop_stats.json` | With `--loop-stats`: `counts` (matching-pursuit steps, scoring calls), `series` and `timings_s` (wall time of each matching-pursuit loop). |

`source_segment_id` says where an atom came from:

- A detection crop id (`randomized-build`).
- `pan:<window>:<t>:<window>:<t>:w<width>`: the pan matrix profile pair it was
  seeded from (`pan-build`).
- `split:<parent's id>:a` or `…:b`: one half of a split atom.

### `split` adds

`summary.json` also holds `dictionary` (the parent folder), `matchings_from`
(where the patches came from), `split_template`, `new_template_ids` (`[a, b]`:
`a` reuses the split atom's id, `b` is new), `accepted`,
`relative_reduction`, `sse_one`, `sse_two`, `n_a`, `n_b`, `best_trial` and
the split options. A rejected split (exit 1) writes only `summary.json`, with
`accepted: false` and no atoms.

## 4. Matching-pursuit folder (`matching-pursuit --out MP`)

| File | Format |
|---|---|
| `summary.json` | `command: matching-pursuit`, `gpl_dir`, `dictionary` (the dictionary folder), `n_windows`, `window_ids`, `window_offset`, `n_atoms`, `n_matchings`, `stop_reason`, `energy_before`, `energy_after`, time and window coverage |
| `matchings.json` | Placements, as in a dictionary folder |
| `loop_stats.json` | With `--loop-stats` |

## 5. Pan matrix profile folder (`python -m analysis.matrix_profile pan --out PAN`)

| File | Format |
|---|---|
| `pan.npy` | `(n_lengths, n_samples)`: the normalized pan matrix profile, one row per motif length |
| `pan.json` | `lengths`, `rows`, `n_samples`, `rate`, `engine`, `disallow_overlap`, and `signal` (how the series was built: source, kind, bands or PCA rows) |
| `pairs.json` | The `--top-k` motif pairs: `length`, `start_a`, `start_b`, `distance`, `zscore` |
| `pan.png` | With `--plot`: `pan.npy` as an image |

## 6. Catalog (`python -m database`, or `--catalog DB` on any command)

One SQLite file (`database/schema.sql`). The web viewer reads it; its path is
`CATALOG_DB` in the backend's `.env`, by default `backend/data/catalog.sqlite`.

| Table | One row per |
|---|---|
| `wav_files` | WAV in the library roots |
| `gpl_segmentations` | GPL output folder |
| `windows` | GPL window, with its time span |
| `learning_runs`, `learning_round_logs` | Dictionary folder, and each of its outer passes |
| `tool_runs` | Command output folder: `randomized_build`, `pan_build`, `split` or `matching_pursuit` |
| `templates` | Atom (the `.npy` path stays on disk) |
| `dictionaries`, `dictionary_templates`, `dictionary_windows` | Dictionary: `seed` (a build), `matching` (a matching-pursuit run of a dictionary), `edit` (a split of its parent), `refinement` (a build from `--init-dictionary`); with its atoms and windows |
| `matchings`, `matching_feedback` | Placement, with its time in seconds; good/bad labels from the viewer |
| `annotation_csvs` | `*.motif_segments.csv` file (hand-labelled song units) |
| `action_history` | Viewer command (status, parameters) |

The catalog stores paths, not copies. Moving a run folder breaks its atoms
until it is registered again.

## 7. Pipeline scripts (`analysis/scripts/pipeline/`)

With `OUT_ROOT` (default `~/whale_runs/<stem>`) the scripts write:

```
OUT_ROOT/
├── gpl/                         step 1
├── randomized_build/            step 2a
├── pan_build/                   step 2b
├── matching_pursuit_<source>/   step 3, source = pan or randomized
├── pan_mp/                      step 4
├── split_<source>_t<id>/        step 5
└── logs/<step>.log              every step: start line, full output, exit code
```

## 8. HMM training (`analysis/HMM/`)

| File | Role |
|---|---|
| `SOCAL_H_76_230426_193500_df100.x.x_10kHz.wav` | The training recording (10 kHz, 10 h); path set in `run_train.py` |
| `work/gpl_193500/` | Its GPL output folder |
| `….motif_segments.csv` | Hand-cut songs, used to initialize the HMM |
| `work/motif_scores.npz` | Cache: 12 panMP prototype scores per second |
| `song_soft_hmm_t6.npz` | The trained HMM |
| `song_soft_hmm_t6_report.pdf`, `work/train_report.json` | Its report |
