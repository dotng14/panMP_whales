# How `energy_and_songs_by_year.md` was produced

This document describes the pipeline that generated `energy_and_songs_by_year.md`, `energy_and_songs_by_year.pdf`, and the per-year panels `figures/year_panels/year_2008.png` … `year_2026.png`, with pointers to the dictionary and HMM used in the scan.

## 1. `scan_dictionary` (35 motifs)

- **Location on disk (`scan_dictionary`):**
  `/home/yfreund/whale_processing/result/LeiData/SOCAL_H_76_230427_014845.x.limited_overlap_k1_w40_pc5_tau04_m8x32_refine2`
  (atoms stored in `atoms/template_*.npy`).
- **How it was built:** Learned from wav3 (`SOCAL_H_76_230427_014845.x.wav`) with limited-overlap dictionary learning (`k=1`, `w=40`, `pc=5`, `tau=0.4`, `m=8..32`, 2 refinement passes). See [`analysis/dict_HMM/01_build_dictionary.md`](../../01_build_dictionary.md).
- **Adaptation for 10 kHz Site H recordings:** `analysis/dict_HMM/find_songs/place.py` (`load_dictionary`) truncates `scan_dictionary` to the first `N_ROWS = 736` frequency rows (12 Hz to 4.5 kHz, matching a 10 kHz `df20` recording) and re-orthonormalizes the components. See [`analysis/dict_HMM/02_place_dictionary.md`](../../02_place_dictionary.md).

## 2. Theme HMM (4 themes)

- **HMM implementation & specification:** [`analysis/dict_HMM/03_themes_and_songs.md`](../../03_themes_and_songs.md) and the C implementation under `analysis/dict_HMM/c/` (binary: `analysis/dict_HMM/c/build/hmm_run`).
- **Reference fit:** `analysis/dict_HMM/find_songs/fit_theme_model.py` places `scan_dictionary` on the 75-minute reference span of 27 April 2023 (01:11:15–02:26:15 UTC, wav2 + wav3) in the 10 kHz file `SOCAL_H_76_230426_054845_df20.x.wav`, runs `hmm_run` to segment the four themes, and fits per-theme motif log-probabilities and visit-length parameters (`mu`, `sigma`) via `analysis/dict_HMM/find_songs/themes.py` (`ThemeModel.fit`).
- **Saved HMM parameters:** `analysis/dict_HMM/find_songs/results/theme_model.npz` (`motif_ids`, `log_p`, `mu`, `sigma`).
- **Combined C detector model:** `analysis/dict_HMM/find_songs/c/export_model.py` packs the 10 kHz-truncated `scan_dictionary` and `theme_model.npz` into a single binary:
  `analysis/dict_HMM/find_songs/c/build/song_model.bin` (used by `analysis/dict_HMM/find_songs/c/song_detector`).

## 3. Site H 10 kHz scan (songs + 600 s energy segments)

The scan ran over all eligible 10 kHz (`df20`) Site H WAV files (6,795 completed across 2007–2026) in three stages:

1. **Initial song-only pass (`run_all.py`, 2007–2012):**
   Ran `song_detector` (`song_model.bin`) directly on the earliest 1,470 files, writing conforming song detections (passing margin and order-shuffle tests) to `conforming_songs.csv` and recording processed stems in `done_stems.txt` / `manifest.csv`.
2. **Hybrid energy-gated pass (`hybrid_run.py`, Nov 14–15, 2026):**
   For each remaining eligible file:
   - Ran `pre_processing/c/energy_filter --cpu --segment-sec 600` to compute the whitened 50–500 Hz / overall energy ratio in 600 s segments and appended rows to `energy_segments.csv`.
   - If any 600 s segment had ratio ≥ `0.15`, ran `song_detector` (`song_model.bin`) on the file and appended conforming songs to `conforming_songs.csv` (`status = ok_song`).
   - Otherwise skipped song detection (`status = ok_energy_only`).
   Finished with **2,546** conforming songs across 6,795 completed files (2 failed).
3. **Energy backfill (`backfill_energy.py`, Nov 15, 2026):**
   Ran `energy_filter --cpu --segment-sec 600` on the 1,470 files from stage 1 (`missing_energy_files.txt`, covering 2007–2012) and appended them to `energy_segments.csv`, bringing the total to **879,704** 600 s energy segments across 2007–2026.

## 4. Generating `energy_and_songs_by_year.md` and `.pdf`

From `energy_segments.csv` and `conforming_songs.csv`:

- Each calendar year (2008–2026) was plotted as a dual-axis panel saved to `figures/year_panels/year_YYYY.png`:
  - **Left axis (blue):** daily mean whitened 50–500 Hz energy fraction from `energy_segments.csv` (with a dashed line at the `0.15` hybrid gate).
  - **Right axis (red):** daily conforming-song count (thin stems) and 14-day centered mean rate (songs/day, thick line) from `conforming_songs.csv`.
- `energy_and_songs_by_year.md` embeds all 19 year panels with per-year song and segment counts, and `energy_and_songs_by_year.pdf` is the rendered PDF version.

> **Note:** `energy_segments.csv` is >10 MB and is not tracked in git (and is no longer present in this directory). For per-minute 50–500 Hz energy ratios across the same 6,795 Site H recordings, see [`MINUTE_ENERGY_SCAN.md`](MINUTE_ENERGY_SCAN.md) (`/home/yfreund/whale_processing/12TB_results/`).
