"""5-minute theme panels: GPL display + motif bars + theme boundary lines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from analysis.song_hmm.exp6.export_csvs import Placement
from analysis.song_hmm.exp7.paths import GPL_WAV23, N_FIVE_MIN_PANELS, WINDOWS_PER_FIVE_MIN
from analysis.song_hmm.exp7.plot_panels import stitch_gpl_display

FREQ_LO_HZ = 10.0
FREQ_HI_HZ = 1000.0


def load_eval(eval_path: Path) -> tuple[list[dict], list[float]]:
    data = json.loads(Path(eval_path).read_text())
    themes = [t for t in (data.get("themes") or []) if float(t["t1"]) > float(t["t0"])]
    bounds = [float(b) for b in (data.get("boundaries") or [])]
    return themes, bounds


def plot_five_min_theme_panels(
    placements: Sequence[Placement],
    themes: Sequence[dict],
    boundaries: Sequence[float],
    *,
    out_dir: Path,
    gpl_dir: Path = GPL_WAV23,
    n_panels: int = N_FIVE_MIN_PANELS,
    panel_indices: Sequence[int] | None = None,
    dpi: int = 140,
    freq_lo_hz: float = FREQ_LO_HZ,
    freq_hi_hz: float = FREQ_HI_HZ,
) -> list[Path]:
    """Write 5-min panels; ``panel_indices`` 1-based (default: all)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab20.colors
    theme_colors = plt.cm.Set2.colors
    indices = list(panel_indices) if panel_indices is not None else list(range(1, n_panels + 1))
    paths: list[Path] = []

    # Stable theme → color
    theme_ids = sorted({str(t.get("theme")) for t in themes})
    theme_cmap = {tid: theme_colors[i % len(theme_colors)] for i, tid in enumerate(theme_ids)}

    for panel in indices:
        offset = (panel - 1) * WINDOWS_PER_FIVE_MIN
        t0_min = (panel - 1) * 5
        t1_min = panel * 5
        fig, (ax_s, ax_m) = plt.subplots(
            2,
            1,
            figsize=(14, 4.6),
            gridspec_kw={"height_ratios": [3.4, 1.0], "hspace": 0.08},
            sharex=True,
        )
        spec, t0, t1, f0, f1 = stitch_gpl_display(
            gpl_dir,
            window_offset=offset,
            n_windows=WINDOWS_PER_FIVE_MIN,
            freq_lo_hz=freq_lo_hz,
            freq_hi_hz=freq_hi_hz,
        )
        finite = spec[np.isfinite(spec)]
        vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        vmin = float(np.percentile(finite, 1.0)) if finite.size else 0.0
        ax_s.imshow(
            spec,
            origin="lower",
            aspect="auto",
            extent=[t0 / 60.0, t1 / 60.0, f0, f1],
            cmap="jet",
            vmin=vmin,
            vmax=max(vmax, vmin + 1e-6),
            interpolation="nearest",
        )
        ax_s.set_ylabel("freq (Hz)")
        ax_s.set_title(
            f"Panel {panel:02d}: {t0_min}–{t1_min} min  "
            f"(GPL display {freq_lo_hz:.0f}–{freq_hi_hz:.0f} Hz) + themes"
        )
        ax_s.tick_params(labelbottom=False)

        # Theme spans tinted on spectrogram
        for th in themes:
            tt0, tt1 = float(th["t0"]), float(th["t1"])
            if tt1 < t0 or tt0 > t1:
                continue
            x0 = max(tt0, t0) / 60.0
            x1 = min(tt1, t1) / 60.0
            if x1 > x0:
                c = theme_cmap.get(str(th.get("theme")), "white")
                ax_s.axvspan(x0, x1, color=c, alpha=0.18, lw=0)

        for b in boundaries:
            if t0 <= b <= t1:
                ax_s.axvline(b / 60.0, color="yellow", lw=1.6, alpha=0.95)
                ax_m.axvline(b / 60.0, color="yellow", lw=1.6, alpha=0.95)

        in_panel = [p for p in placements if p.end_sec >= t0 and p.start_sec <= t1]
        seen: dict[int, tuple] = {}
        ax_m.set_xlim(t0 / 60.0, t1 / 60.0)
        ax_m.set_ylim(0, 1)
        ax_m.set_yticks([])
        ax_m.set_xlabel("time (min from wav start)")
        ax_m.set_ylabel("motifs")
        for p in in_panel:
            if p.motif not in seen:
                seen[p.motif] = colors[len(seen) % len(colors)]
            x0 = max(p.start_sec, t0) / 60.0
            x1 = min(p.end_sec, t1) / 60.0
            if x1 > x0:
                ax_m.axvspan(x0, x1, ymin=0.15, ymax=0.85, color=seen[p.motif], alpha=0.95, lw=0)

        n_th = sum(1 for th in themes if float(th["t1"]) >= t0 and float(th["t0"]) <= t1)
        ax_m.set_title(
            f"{len(in_panel)} placements, {len(seen)} motif ids, {n_th} theme spans in panel",
            fontsize=9,
            loc="left",
        )
        handles = [
            Patch(facecolor=theme_cmap[tid], edgecolor="none", label=tid, alpha=0.5)
            for tid in theme_ids[:8]
        ]
        if handles:
            ax_s.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7, ncol=4)

        path = out_dir / f"panel_{panel:02d}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
