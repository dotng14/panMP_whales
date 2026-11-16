"""Plot motif placements on GPL spectrograms (hour / 30-min / 5-min panels)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from analysis.song_hmm.exp4.spectrogram import stitch_gpl
from analysis.song_hmm.exp5.paths import GPL_193500, WINDOWS_PER_HOUR, WORK
from analysis.song_hmm.exp6.export_csvs import Placement

# 75 s windows: 48 per hour → 24 per 30 min → 4 per 5 min
WINDOWS_PER_HALF_HOUR = WINDOWS_PER_HOUR // 2  # 24
WINDOWS_PER_FIVE_MIN = WINDOWS_PER_HOUR // 12  # 4
N_HALF_HOURS = 20  # 10 h
N_FIVE_MIN = 10 * 12  # 120 panels


def _draw_panel(
    placements: Sequence[Placement],
    *,
    window_offset: int,
    n_windows: int,
    gpl_dir: Path,
    ax: plt.Axes,
    title: str,
    vmax_pct: float = 99.0,
    span_ymax: float = 0.28,
) -> plt.Axes:
    spec, t0, t1, _ = stitch_gpl(gpl_dir, window_offset=window_offset, n_windows=n_windows)
    finite = spec[np.isfinite(spec)]
    vmax = float(np.percentile(finite, vmax_pct)) if finite.size else 1.0
    vmin = float(np.percentile(finite, 1.0)) if finite.size else 0.0
    ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[t0 / 60.0, t1 / 60.0, 0, spec.shape[0]],
        cmap="magma",
        vmin=vmin,
        vmax=max(vmax, vmin + 1e-6),
        interpolation="nearest",
    )
    colors = plt.cm.tab20.colors
    seen: dict[int, tuple] = {}
    n_vis = 0
    for p in placements:
        if p.end_sec < t0 or p.start_sec > t1:
            continue
        n_vis += 1
        if p.motif not in seen:
            seen[p.motif] = colors[len(seen) % len(colors)]
        c = seen[p.motif]
        x0 = max(p.start_sec, t0) / 60.0
        x1 = min(p.end_sec, t1) / 60.0
        ax.axvspan(x0, x1, ymin=0.0, ymax=span_ymax, color=c, alpha=0.9, lw=0)
        ax.axvline(x0, color=c, alpha=0.85, lw=0.8, ymin=0.0, ymax=span_ymax)
    ax.set_xlim(t0 / 60.0, t1 / 60.0)
    ax.set_xlabel("time (min from wav start)")
    ax.set_ylabel("freq bin")
    ax.set_title(f"{title}  ({n_vis} placements, {len(seen)} motif ids)")
    if seen:
        handles = [
            Patch(facecolor=c, edgecolor="none", label=f"M{m}")
            for m, c in sorted(seen.items())[:16]
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            ncol=min(8, len(handles)),
            fontsize=6,
            framealpha=0.7,
        )
    return ax


def plot_motifs_hour(
    placements: Sequence[Placement],
    *,
    hour: int,
    gpl_dir: Path = GPL_193500,
    ax: plt.Axes | None = None,
    title: str | None = None,
    vmax_pct: float = 99.0,
) -> plt.Axes:
    """Hour is 1-based (1..10)."""
    if not 1 <= hour <= 10:
        raise ValueError(hour)
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 2.8))
    return _draw_panel(
        placements,
        window_offset=(hour - 1) * WINDOWS_PER_HOUR,
        n_windows=WINDOWS_PER_HOUR,
        gpl_dir=gpl_dir,
        ax=ax,
        title=title or f"Hour {hour}",
        vmax_pct=vmax_pct,
    )


def plot_motifs_half_hour(
    placements: Sequence[Placement],
    *,
    panel: int,
    gpl_dir: Path = GPL_193500,
    ax: plt.Axes | None = None,
    title: str | None = None,
    vmax_pct: float = 99.0,
) -> plt.Axes:
    """``panel`` is 1-based index over 20 half-hours (0–0.5 h … 9.5–10 h)."""
    if not 1 <= panel <= N_HALF_HOURS:
        raise ValueError(panel)
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 2.6))
    offset = (panel - 1) * WINDOWS_PER_HALF_HOUR
    t0_min = (panel - 1) * 30
    t1_min = panel * 30
    return _draw_panel(
        placements,
        window_offset=offset,
        n_windows=WINDOWS_PER_HALF_HOUR,
        gpl_dir=gpl_dir,
        ax=ax,
        title=title or f"Panel {panel:02d}: {t0_min}–{t1_min} min",
        vmax_pct=vmax_pct,
    )


def plot_motifs_ten_hours(
    placements: Sequence[Placement],
    *,
    out_dir: Path | None = None,
    gpl_dir: Path = GPL_193500,
) -> list[plt.Figure]:
    out_dir = Path(out_dir) if out_dir is not None else WORK / "figures" / "motifs_by_hour"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: list[plt.Figure] = []
    for hour in range(1, 11):
        fig, ax = plt.subplots(figsize=(14, 2.8))
        plot_motifs_hour(placements, hour=hour, gpl_dir=gpl_dir, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"hour_{hour:02d}.png", dpi=120, bbox_inches="tight")
        figs.append(fig)
    return figs


def plot_motifs_twenty_half_hours(
    placements: Sequence[Placement],
    *,
    out_dir: Path | None = None,
    gpl_dir: Path = GPL_193500,
) -> list[plt.Figure]:
    """20 panels × 30 minutes covering the full 10 h wav."""
    out_dir = Path(out_dir) if out_dir is not None else WORK / "figures" / "motifs_by_30min"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: list[plt.Figure] = []
    for panel in range(1, N_HALF_HOURS + 1):
        fig, ax = plt.subplots(figsize=(14, 2.6))
        plot_motifs_half_hour(placements, panel=panel, gpl_dir=gpl_dir, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"panel_{panel:02d}.png", dpi=130, bbox_inches="tight")
        figs.append(fig)
        plt.close(fig)
    return figs


def plot_motifs_five_min(
    placements: Sequence[Placement],
    *,
    panel: int,
    gpl_dir: Path = GPL_193500,
    ax: plt.Axes | None = None,
    title: str | None = None,
    vmax_pct: float = 99.0,
) -> plt.Axes:
    """``panel`` is 1-based index over 120 five-minute slices (0–5 min … 595–600 min)."""
    if not 1 <= panel <= N_FIVE_MIN:
        raise ValueError(panel)
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3.2))
    offset = (panel - 1) * WINDOWS_PER_FIVE_MIN
    t0_min = (panel - 1) * 5
    t1_min = panel * 5
    return _draw_panel(
        placements,
        window_offset=offset,
        n_windows=WINDOWS_PER_FIVE_MIN,
        gpl_dir=gpl_dir,
        ax=ax,
        title=title or f"Panel {panel:03d}: {t0_min}–{t1_min} min",
        vmax_pct=vmax_pct,
        span_ymax=0.32,
    )


def plot_motifs_all_five_min(
    placements: Sequence[Placement],
    *,
    out_dir: Path | None = None,
    gpl_dir: Path = GPL_193500,
    dpi: int = 140,
) -> list[Path]:
    """120 panels × 5 minutes covering the full 10 h wav. Returns saved PNG paths."""
    out_dir = Path(out_dir) if out_dir is not None else WORK / "figures" / "motifs_by_5min"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for panel in range(1, N_FIVE_MIN + 1):
        fig, ax = plt.subplots(figsize=(14, 3.2))
        plot_motifs_five_min(placements, panel=panel, gpl_dir=gpl_dir, ax=ax)
        fig.tight_layout()
        path = out_dir / f"panel_{panel:03d}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        if panel % 24 == 0:
            print(f"  wrote {panel}/{N_FIVE_MIN}", flush=True)
    return paths
