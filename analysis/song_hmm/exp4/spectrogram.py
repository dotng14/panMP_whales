"""Stitch GPL windows and plot spectrograms with themes / boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from numpy.typing import NDArray

from analysis.dict_learning.code.cli import load_windows


def stitch_gpl(
    gpl_dir: Path,
    *,
    window_offset: int = 0,
    n_windows: int | None = None,
    t0: float | None = None,
    t1: float | None = None,
) -> tuple[NDArray[np.floating], float, float, float]:
    """Return (spectrogram F×T, t_start_s, t_end_s, sec_per_bin).

    If t0/t1 are given (absolute seconds from wav start), only that range is kept.
    """
    windows = load_windows(gpl_dir, n_windows, window_offset)
    if not windows:
        raise ValueError(f"no windows in {gpl_dir}")
    sec_per_bin = float(windows[0].overlap) / float(windows[0].sample_freq)
    chunks: list[NDArray[np.floating]] = []
    times0: list[float] = []
    for w in windows:
        start = (int(w.start_sample) - 1) / float(w.sample_freq)
        chunks.append(np.asarray(w.data, dtype=float))
        times0.append(start)
    # concatenate along time; windows are contiguous 75 s blocks
    spec = np.concatenate(chunks, axis=1)
    t_start = times0[0]
    t_end = times0[-1] + chunks[-1].shape[1] * sec_per_bin
    if t0 is not None or t1 is not None:
        t0 = t_start if t0 is None else t0
        t1 = t_end if t1 is None else t1
        i0 = max(0, int(round((t0 - t_start) / sec_per_bin)))
        i1 = min(spec.shape[1], int(round((t1 - t_start) / sec_per_bin)))
        spec = spec[:, i0:i1]
        t_start = t_start + i0 * sec_per_bin
        t_end = t_start + spec.shape[1] * sec_per_bin
    return spec, float(t_start), float(t_end), float(sec_per_bin)


def plot_themes_on_spectrogram(
    spec: NDArray[np.floating],
    t_start: float,
    t_end: float,
    themes: Sequence[dict],
    boundaries: Sequence[float],
    *,
    title: str = "",
    ax: plt.Axes | None = None,
    vmax_pct: float = 99.0,
    show_legend: bool = True,
) -> plt.Axes:
    """Draw whitened spectrogram with coloured theme spans and boundary lines."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3.5))
    finite = spec[np.isfinite(spec)]
    vmax = float(np.percentile(finite, vmax_pct)) if finite.size else 1.0
    vmin = float(np.percentile(finite, 1.0)) if finite.size else 0.0
    ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[t_start / 60.0, t_end / 60.0, 0, spec.shape[0]],
        cmap="magma",
        vmin=vmin,
        vmax=max(vmax, vmin + 1e-6),
        interpolation="nearest",
    )
    colors = plt.cm.tab20.colors
    label_color: dict[str, tuple] = {}
    y0, y1 = ax.get_ylim()
    for th in themes:
        lab = str(th.get("theme", "?"))
        if lab not in label_color:
            label_color[lab] = colors[len(label_color) % len(colors)]
        x0 = float(th["t0"]) / 60.0
        x1 = float(th["t1"]) / 60.0
        ax.axvspan(x0, x1, ymin=0.0, ymax=0.08, color=label_color[lab], alpha=0.95, lw=0)
        ax.axvspan(x0, x1, color=label_color[lab], alpha=0.12, lw=0)
    for b in boundaries:
        if t_start <= b <= t_end:
            ax.axvline(b / 60.0, color="cyan", lw=1.4, alpha=0.95)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("freq bin")
    ax.set_title(title)
    if show_legend and label_color:
        handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in label_color.items()]
        handles.append(plt.Line2D([0], [0], color="cyan", lw=2, label="boundary"))
        ax.legend(
            handles=handles,
            loc="upper right",
            ncol=min(8, len(handles)),
            fontsize=7,
            framealpha=0.7,
        )
    return ax


def plot_eval_slices(
    gpl_dir: Path,
    themes: Sequence[dict],
    boundaries: Sequence[float],
    *,
    window_offset: int,
    n_windows: int,
    slice_s: float = 600.0,
    title_prefix: str = "",
    out_dir: Path | None = None,
) -> list[plt.Figure]:
    """Plot the eval region in fixed-length slices for notebook readability."""
    spec, t0, t1, _ = stitch_gpl(gpl_dir, window_offset=window_offset, n_windows=n_windows)
    figs: list[plt.Figure] = []
    t = t0
    idx = 0
    while t < t1 - 1e-6:
        te = min(t + slice_s, t1)
        # column slice
        frac0 = (t - t0) / max(t1 - t0, 1e-9)
        frac1 = (te - t0) / max(t1 - t0, 1e-9)
        i0 = int(frac0 * spec.shape[1])
        i1 = max(i0 + 1, int(frac1 * spec.shape[1]))
        sub = spec[:, i0:i1]
        th = [x for x in themes if x["t1"] >= t and x["t0"] <= te]
        bd = [b for b in boundaries if t <= b <= te]
        fig, ax = plt.subplots(figsize=(14, 3.2))
        plot_themes_on_spectrogram(
            sub,
            t,
            te,
            th,
            bd,
            title=f"{title_prefix}  {t/60:.1f}–{te/60:.1f} min",
            ax=ax,
        )
        figs.append(fig)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"slice_{idx:02d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        idx += 1
        t = te
    return figs
