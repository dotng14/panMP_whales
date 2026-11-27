# Whale Song Viewer — User Guide

This guide explains how to browse Site H recordings, inspect spectrograms and detected songs/themes/motifs, view the 2023 motif dictionary, and explore the 20-year energy scan in the web app.

---

## 1. Main Tabs

At the top of the page, switch between two views:

- **Recordings** — Pick a Site H WAV recording (or upload one), listen to the audio, inspect the spectrogram, and view detected motifs, themes, and songs.
- **20-Year Scan** — Browse monthly hourly energy plots and long-term overview plots from the 20-year Site H scan.

---

## 2. Finding and Opening a Recording

In the **Recordings** tab, use the picker at the top to find a file:

- **Site:** Fixed to `H` (Site H).
- **Date & time boxes (`YYYY-MM-DD HH:MM:SS`):** Type into any combination of **Year**, **Month**, **Day**, **Hour**, **Minute**, and **Second**. The matching list filters immediately as you type.
- **Readiness filter (`All` / `Ready` / `Not ready`):**
  - **Ready** — Already processed and cached; opens immediately.
  - **Not ready** — Not yet processed; clicking it runs the GPU hybrid detector (`song_detector`, ~5–10 s for a 22-hour 10 kHz WAV) and caches the result.
- **Example recording with clear songs:** Filter to `2023-04-26 05:48:45` (`SOCAL_H_76_230426_054845_df20.x.wav`). It has 21 conforming songs (Song 1 starts around `264m`; Songs 11–21 around `1156m–1291m` have especially strong calls).

You can also upload a local `.wav` file using the upload control.

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
- Use **◀ Prev / Next ▶** (or `←` / `→` arrow keys) to step through motifs, and `×` or `Esc` to close.

> **Note:** This dictionary is a single fixed model trained on 2023 data (`song_model.bin`) and is used as the reference dictionary for all recordings processed by the hybrid detector.

---

## 6. Dictionary Actions & Repeated Matches

- **Actions button:** Located below the spectrogram when a recording does not yet have a catalog dictionary. Click **Actions** to run `randomized_builder` (builds a per-recording dictionary) or view past action history.
- **Repeated Matches:** When a recording has a catalog dictionary selected in the analysis picker, the **Repeated Matches** panel appears below the spectrogram, showing matched patches grouped by template. Clicking a patch highlights its location on the main spectrogram.

---

## 7. 20-Year Scan Tab

Switch to **20-Year Scan** to browse long-term Site H energy statistics:

- Use **Older / Newer** or the **Year** and **Month** dropdowns to inspect monthly hourly activity plots and summary statistics (recordings, hours, minutes, mean energy ratio, and 0.1–1 Hz FFT fraction).
- Use the overview tabs (**Monthly scatter**, **Minute scatter**, **2D contour**) to view multi-year summary plots.
