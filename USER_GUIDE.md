# Whale Song Viewer — User Guide

This guide explains how to browse Site H recordings, inspect spectrograms and detected songs/themes/motifs, and view the 2023 motif dictionary in the web app.

---

## 1. Overview

The app has one view, **Recordings**: pick a Site H WAV recording, listen to the audio, inspect the spectrogram, and view detected motifs, themes, and songs.

---

## 2. Finding and Opening a Recording

Use the picker at the top to find a file:

- **Site:** Fixed to `H` (Site H).
- **Date & time boxes (`YYYY-MM-DD HH:MM:SS`):** Type into any combination of **Year**, **Month**, **Day**, **Hour**, **Minute**, and **Second**. The matching list filters immediately as you type. You can use either the single text line or the individual boxes; they do the same thing.
- **Readiness filter (`All` / `Ready` / `Not ready`):**
  - **Ready** — Already processed and cached; opens immediately.
  - **Not ready** — Not yet processed; clicking it runs the GPU hybrid detector (`song_detector`, ~5–10 s for a 22-hour 10 kHz WAV) and caches the result.
- **Example recording with clear songs:** Filter to `2023-04-26 05:48:45` (`SOCAL_H_76_230426_054845_df20.x.wav`). It has 21 conforming songs (Song 1 starts around `264m`; Songs 11–21 around `1156m–1291m` have especially strong calls).

**YF** I think it would be nice to have a history of times observed, so one can easily jump back

---

## 3. Spectrogram & Audio Controls

Once a recording loads:

- **Play / Pause & Follow:** Play the audio and toggle **Follow** to keep the spectrogram scrolled with the playhead.
- **Click to seek:** Click anywhere on the spectrogram to jump the playhead to that time.
- **`px/sec` slider:** Horizontal zoom (time scale). When zoomed in to a window of 5 minutes or less on a long recording, the viewer automatically fetches a full-resolution (`0.123 s/col`) spectrogram slice for the visible area.
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

## 5. 2023 Motif Dictionary

Below the spectrogram (when no per-recording catalog dictionary is selected), the **2023 Motif Dictionary (35 motifs)** strip displays all 35 motifs from the fixed 2023 model:

- Each tile shows the motif ID (`M0`, `M1`, …) and its duration in seconds.
- **Click any motif** to open a zoomed popup with time (s), frequency (Hz), and relative power (dB) axes.
**YF** I'd like to bring back the examples of the locations where the motifs fit, like we used to have.
- Use **◀ Prev / Next ▶** (or `←` / `→` arrow keys) to step through motifs, and `×` or `Esc` to close.

> **Note:** This dictionary is a single fixed model trained on 2023 data (`song_model.bin`) and is used as the reference dictionary for all recordings processed by the hybrid detector.

---

## 6. Repeated Matches

**YF** We do need a way to communicate how the processing was done (which version of the dictionary + HMM). I think that visualization by the developers will be done using notebooks and when things are ready, the version of the dictionary and HMM will be updated and used for the visualization. The version will include a separate model for each year. It would be good to have several versions loaded, so that their performance can be compared visually.

- **Repeated Matches:** When a recording has a catalog dictionary selected in the analysis picker, the **Repeated Matches** panel appears below the spectrogram, showing matched patches grouped by template. Clicking a patch highlights its location on the main spectrogram.
