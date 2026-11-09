"""Spectrogram hint images for the labeller, one per CSV theme, from the TRAIN half.

    python -m analysis.song_hmm.exp3.reference.make_examples

The CSV theme labels may be wrong: these images are hints, not ground truth.
Writes PNGs into ``analysis/song_hmm/exp3/reference/examples/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

from analysis.song_hmm.exp1_slot_hmm.labels import load_labels

from .export_clips import CSV_PATH, N_SECONDS, TEST_START, WAV_PATH

OUT_DIR = Path(__file__).resolve().parent / "examples"
PRE_S, POST_S = 15, 45  # context before / after the theme start
OVERVIEW_SONG = 6  # index into CSV songs (train half) for the whole-song overview


def theme_runs(theme: np.ndarray, until: int) -> list[tuple[int, int, int]]:
    """Contiguous (start, end, theme) runs of labelled seconds before ``until``."""
    runs, s = [], 0
    for i in range(1, until + 1):
        if i == until or theme[i] != theme[s]:
            if theme[s] >= 0:
                runs.append((s, i, int(theme[s])))
            s = i
    return runs


def spec_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, s = spectrogram(x.astype(np.float64) - x.mean(), fs=sr, nperseg=512, noverlap=384)
    keep = f <= 1000
    db = 10 * np.log10(s[keep] + 1e-12)
    return f[keep], t, db


def draw(ax, data, sr, a, b, spans, title):
    f, t, db = spec_db(np.asarray(data[a * sr : b * sr]), sr)
    lo, hi = np.percentile(db, [20, 99.5])
    ax.pcolormesh(t + a, f, db, shading="auto", cmap="magma", vmin=lo, vmax=hi)
    for s, e, label in spans:
        s, e = max(s, a), min(e, b)
        ax.axvline(s, color="cyan", lw=1.2)
        ax.axvline(e, color="cyan", lw=1.2, ls="--")
        ax.text((s + e) / 2, 960, label, color="cyan", ha="center", va="top", fontsize=10, weight="bold")
    ax.set_xlim(a, b)
    ax.set_ylim(0, 1000)
    ax.set_ylabel("Hz")
    ax.set_xlabel("recording time (s)")
    ax.set_title(title, fontsize=10)


def main() -> None:
    sr, data = wavfile.read(WAV_PATH, mmap=True)
    labels = load_labels(CSV_PATH, N_SECONDS)
    runs = theme_runs(labels.theme, TEST_START)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    note = "CSV labels may be wrong - hint only"
    for th in range(6):
        cand = [r for r in runs if r[2] == th and r[0] >= PRE_S]
        med = np.median([e - s for s, e, _ in cand])
        s, e, _ = min(cand, key=lambda r: (abs((r[1] - r[0]) - med), r[0]))
        a, b = s - PRE_S, s + POST_S
        spans = [(rs, re, f"T{rt + 1}") for rs, re, rt in runs if re > a and rs < b]
        fig, ax = plt.subplots(figsize=(12, 3.6))
        draw(ax, data, sr, a, b, spans,
             f"Theme T{th + 1} per CSV: {s}-{e} s ({e - s} s, typical length ~{med:.0f} s). {note}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"theme_T{th + 1}.png", dpi=110)
        plt.close(fig)
        print(f"T{th + 1}: {s}-{e} s")

    a, b = (int(round(v)) for v in labels.song_bounds[OVERVIEW_SONG])
    spans = [(rs, re, f"T{rt + 1}") for rs, re, rt in runs if re > a and rs < b]
    fig, ax = plt.subplots(figsize=(16, 3.8))
    draw(ax, data, sr, a - 10, b + 10, spans, f"One whole song per CSV: {a}-{b} s (T1..T6, then T1 of next song). {note}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "song_overview.png", dpi=100)
    plt.close(fig)
    print(f"overview song: {a}-{b} s")


if __name__ == "__main__":
    main()
