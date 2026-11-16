"""Draw one theme-2 visit: whitened spectrogram and motif raster.

usage: python3 figure_theme2.py WAV PLACEMENTS_CSV START_SEC END_SEC OUT_PNG TITLE
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

ROWS = 410
ROW_HZ = spectro.RATE / spectro.FFTL
# Motifs that carry most of theme 2's probability, plus the pile.
HIGHLIGHT = {43: "#d62728", 1: "#ff7f0e", 44: "#2ca02c", 31: "#9467bd", 27: "#8c564b", 0: "#e377c2", 10: "#17becf"}


def pieces_for(wav: Path, start_sec: float, end_sec: float):
    first = int(start_sec // spectro.WINDOW_SEC)
    last = int(np.ceil(end_sec / spectro.WINDOW_SEC))
    pieces = []
    for window in range(first, last):
        audio = spectro.read_seconds(wav, window * spectro.WINDOW_SEC, spectro.WINDOW_SEC)
        white = spectro.window_spectrogram(audio, ROWS)
        pieces.append((window * spectro.WINDOW_SEC, white))
    return pieces


def main() -> None:
    wav, csv = Path(sys.argv[1]), Path(sys.argv[2])
    start_sec, end_sec = float(sys.argv[3]), float(sys.argv[4])
    out, title = Path(sys.argv[5]), sys.argv[6]
    placements = pd.read_csv(csv)
    placements = placements[(placements.start_sec >= start_sec) & (placements.start_sec < end_sec)]
    motif_ids = sorted(pd.read_csv(csv).motif.unique())
    row_of = {motif: i for i, motif in enumerate(motif_ids)}

    fig, (ax, raster) = plt.subplots(2, 1, figsize=(14, 5.2), gridspec_kw={"height_ratios": [3, 1.6]})
    pieces = pieces_for(wav, start_sec, end_sec)
    top = np.percentile(np.concatenate([p[1].ravel() for p in pieces]) ** (1 / 3), 99.9)
    for t0, white in pieces:
        t1 = t0 + white.shape[1] * spectro.HOP_SEC
        ax.imshow(
            white ** (1 / 3),
            origin="lower",
            aspect="auto",
            cmap="jet",
            vmin=0,
            vmax=top,
            extent=((t0 - start_sec) / 60, (t1 - start_sec) / 60, spectro.FIRST_BIN * ROW_HZ, (spectro.FIRST_BIN + ROWS) * ROW_HZ),
        )
    ax.set_ylabel("Hz")
    ax.set_xlim(0, (end_sec - start_sec) / 60)
    ax.set_title(title)
    ax.text(0.02, 0.95, "theme 2", transform=ax.transAxes, color="white", fontsize=12, weight="bold", va="top")

    colors = [HIGHLIGHT.get(m, "#7f7f7f") for m in placements.motif]
    raster.scatter((placements.start_sec - start_sec) / 60, [row_of[m] for m in placements.motif], s=18, marker="|", c=colors)
    raster.set_xlim(0, (end_sec - start_sec) / 60)
    raster.set_xlabel("minutes from segment start")
    raster.set_ylabel("motif")
    raster.set_yticks([row_of[m] for m in HIGHLIGHT if m in row_of])
    raster.set_yticklabels([str(m) for m in HIGHLIGHT if m in row_of])
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
