"""Draw one song: whitened spectrogram, theme boundaries, and motif placements.

usage: python3 figure.py WAV PLACEMENTS_CSV START_SEC THEME_STARTS_SEC END_SEC OUT_PNG TITLE
THEME_STARTS_SEC is the four theme starts joined by '/', in seconds from the file start.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spectro

ROWS = 410  # 12 Hz to 2.5 kHz
ROW_HZ = spectro.RATE / spectro.FFTL
THEME_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def song_spectrogram(wav: Path, start_sec: float, end_sec: float):
    first = int(start_sec // spectro.WINDOW_SEC)
    last = int(np.ceil(end_sec / spectro.WINDOW_SEC))
    pieces = []
    for window in range(first, last):
        audio = spectro.read_seconds(wav, window * spectro.WINDOW_SEC, spectro.WINDOW_SEC)
        white = spectro.window_spectrogram(audio, ROWS)
        full = np.zeros((ROWS, spectro.N_TIME + 1))  # 75 s is 610.35 steps
        full[:, : white.shape[1]] = white
        pieces.append((window * spectro.WINDOW_SEC, white))
    return pieces


def main() -> None:
    wav, csv = Path(sys.argv[1]), Path(sys.argv[2])
    start_sec, end_sec = float(sys.argv[3]), float(sys.argv[5])
    theme_starts = [float(v) for v in sys.argv[4].split("/")]
    out, title = Path(sys.argv[6]), sys.argv[7]
    edges = [*theme_starts, end_sec]
    placements = pd.read_csv(csv)
    placements = placements[(placements.start_sec >= start_sec) & (placements.start_sec < end_sec)]
    motif_ids = sorted(pd.read_csv(csv).motif.unique())
    row_of = {motif: i for i, motif in enumerate(motif_ids)}

    n_panels = int(np.ceil((end_sec - start_sec) / 240.0))
    span = (end_sec - start_sec) / n_panels
    fig, axes = plt.subplots(
        2 * n_panels, 1, figsize=(16, 4.6 * n_panels),
        gridspec_kw={"height_ratios": [3, 1.6] * n_panels},
    )
    pieces = song_spectrogram(wav, start_sec, end_sec)
    top = np.percentile(np.concatenate([p[1].ravel() for p in pieces]) ** (1 / 3), 99.9)
    for panel in range(n_panels):
        a = start_sec + panel * span
        b = a + span
        ax, raster = axes[2 * panel], axes[2 * panel + 1]
        for t0, white in pieces:
            t1 = t0 + white.shape[1] * spectro.HOP_SEC
            if t1 < a or t0 > b:
                continue
            ax.imshow(
                white ** (1 / 3), origin="lower", aspect="auto", cmap="jet", vmin=0, vmax=top,
                extent=((t0 - start_sec) / 60, (t1 - start_sec) / 60, spectro.FIRST_BIN * ROW_HZ, (spectro.FIRST_BIN + ROWS) * ROW_HZ),
            )
        for theme in range(4):
            lo, hi = max(edges[theme], a), min(edges[theme + 1], b)
            if hi <= lo:
                continue
            for axis in (ax, raster):
                axis.axvline((edges[theme] - start_sec) / 60, color="white" if axis is ax else "black", lw=1.5)
            ax.text((lo - start_sec) / 60 + 0.03, 2330, f"theme {theme}", color="white", fontsize=11, weight="bold", va="top")
            inside = placements[(placements.start_sec >= lo) & (placements.start_sec < hi)]
            raster.scatter(
                (inside.start_sec - start_sec) / 60, [row_of[m] for m in inside.motif],
                s=14, marker="|", color=THEME_COLORS[theme],
            )
        ax.set_xlim((a - start_sec) / 60, (b - start_sec) / 60)
        ax.set_ylim(0, 2500)
        ax.set_ylabel("Hz")
        ax.set_xticklabels([])
        raster.set_xlim((a - start_sec) / 60, (b - start_sec) / 60)
        raster.set_ylim(-1, len(motif_ids))
        raster.set_yticks(range(0, len(motif_ids), 4))
        raster.set_yticklabels([motif_ids[i] for i in range(0, len(motif_ids), 4)], fontsize=7)
        raster.set_ylabel("motif")
        raster.grid(True, alpha=0.3)
    axes[-1].set_xlabel("minutes from the start of the song")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
