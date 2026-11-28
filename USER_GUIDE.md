# Whale Song Viewer — User Guide

This guide explains how to browse Site H recordings, inspect spectrograms and detected songs/themes/motifs, review repeated motif matches, and jump back to previously visited times in the web app.

---

## 1. Overview

The app has one view, **Recordings**: pick a Site H WAV recording, listen to the audio, inspect the spectrogram, view detected motifs, themes, and songs, and browse repeated matches for each motif.

---

## 2. Finding and Opening a Recording

Use the picker at the top to find a file:

- **Site:** Fixed to `H` (Site H).
- **Date & time boxes (`YYYY-MM-DD HH:MM:SS`):** Type into any combination of **Year**, **Month**, **Day**, **Hour**, **Minute**, and **Second**. The matching list filters immediately as you type. You can use either the single text line or the individual boxes; they do the same thing.
- **Readiness filter (`All` / `Ready` / `Not ready`):**
  - **Ready** — Already processed and cached; opens immediately.
  - **Not ready** — Not yet processed; clicking it runs the GPU hybrid detector (`song_detector`, ~5–10 s for a 22-hour 10 kHz WAV) and caches the result.
- **Example recording with clear songs:** Filter to `2023-04-26 05:48:45` (`SOCAL_H_76_230426_054845_df20.x.wav`). It has 21 conforming songs (Song 1 starts around `264m`; Songs 11–21 around `1156m–1291m` have especially strong calls).

- **History (`🕒 History`):** Located next to the file picker. Whenever you interact with a recording (click the spectrogram to seek, step through songs, click a lane bar or repeated match, or pause playback), that `(file, time)` is saved automatically in your browser. Open **History** to filter and jump straight back to any previously observed recording and timestamp.

---

## 3. Spectrogram & Audio Controls

Once a recording loads:

- **Play / Pause & Follow:** Play the audio and toggle **Follow** to keep the spectrogram scrolled with the playhead.
- **Click to seek:** Click anywhere on the spectrogram to jump the playhead to that time.
- **`px/sec` slider:** Horizontal zoom (time scale). When zoomed in to a window of 10 minutes or less on a long recording, the viewer streams seamless full-resolution (`0.123 s/col`) tiles on demand and prefetches ahead as you scroll or listen.
- **`Max Hz` slider:** Vertical zoom (frequency cap, up to `4498 Hz` for 10 kHz recordings). Lowering this to `1500–2000 Hz` makes low-frequency humpback calls larger and easier to read.
- **`← Song` / `Song →` buttons:** Step directly through detected conforming songs. When a recording with songs first loads, the viewer automatically jumps to **Song 1**.

---

## 4. Song-Structure Lanes (Motif / Theme / Song)

Directly below the spectrogram are three aligned annotation lanes (each with a visibility checkbox on the left):

- **Motif** — Individual motif placements matched by the 2023 dictionary (`M0`–`M49`).
- **Theme** — Decoded 4-theme cyclic HMM segments (`0`, `1`, `2`, `3`) across all decoded windows.
- **Song** — Conforming songs (windows that pass both the margin check `min_margin > 0` and the shuffle test `shuffles_at_or_above == 0`).

Click any bar in a lane to highlight and jump to that span on the spectrogram.

> **Note:** Outside conforming songs (during quiet or non-song stretches), the cyclic HMM has no "silence" state, so it may produce long `theme 0` stretches (up to 30 minutes) followed by 1-minute `1 / 2 / 3` stubs. Use the **Song** lane or `← Song / Song →` buttons to jump straight to confirmed songs.

---

## 5. Repeated Matches (2023 Motif Dictionary)

Below the spectrogram, the **Repeated Matches** panel shows where each motif from the 2023 model (`song_model.bin`, 35 motifs `M0`–`M49`) fits in the recording, using the matching-pursuit placements produced by `song_detector`:

- Use **Prev / Next** to step through motifs. For each motif, the left side displays its average template spectrogram and a distance CDF plot, and the main strip displays the best-matching patches (sorted by lowest unexplained energy, up to 200 per motif).
- **Play** listens to the stitched matched patches; **Click any patch** (with **Original** enabled) to highlight and jump straight to that location on the main spectrogram above.
- When a per-recording catalog dictionary is selected in the analysis picker, **Repeated Matches** switches to that catalog dictionary and also enables catalog-specific review/rename/action tools.

**YF** We do need a way to communicate how the processing was done (which version of the dictionary + HMM). I think that visualization by the developers will be done using notebooks and when things are ready, the version of the dictionary and HMM will be updated and used for the visualization. The version will include a separate model for each year. It would be good to have several versions loaded, so that their performance can be compared visually.
