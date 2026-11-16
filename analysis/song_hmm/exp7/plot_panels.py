"""5-minute motif panels for wav23 using the GPL **display** spectrogram pipeline.

Uses per-window ``*_display.npy`` (cube-root transform from
``analysis.GPL_Python.gpl_outputs``), jet colormap, and Hz axis — not raw
whitened bins — so panels match the GPL viewer look.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from numpy.typing import NDArray

from analysis.GPL_Python.gpl_outputs import load_display
from analysis.song_hmm.exp6.export_csvs import Placement
from analysis.song_hmm.exp7.paths import (
    GPL_WAV23,
    N_FIVE_MIN_PANELS,
    WINDOWS_PER_FIVE_MIN,
)

_SIDECAR_RE = re.compile(r"^spwhiten_(?P<stem>.+?)_w(?P<win>\d+)_")

# Song band for display (matches Exp5 GPL 10–1000 Hz focus)
FREQ_LO_HZ = 10.0
FREQ_HI_HZ = 1000.0


def _sorted_sidecars(npy_dir: Path) -> list[Path]:
    return sorted(
        npy_dir.glob("spwhiten_*.json"),
        key=lambda p: int(_SIDECAR_RE.match(p.stem).group("win")),  # type: ignore[union-attr]
    )


def stitch_gpl_display(
    gpl_dir: Path,
    *,
    window_offset: int = 0,
    n_windows: int | None = None,
    freq_lo_hz: float = FREQ_LO_HZ,
    freq_hi_hz: float = FREQ_HI_HZ,
) -> tuple[NDArray[np.floating], float, float, float, float]:
    """Return (display F×T, t0_s, t1_s, freq_lo, freq_hi) from GPL display npys."""
    npy_dir = Path(gpl_dir) / "spectrogram_npy"
    sidecars = _sorted_sidecars(npy_dir)
    if not sidecars:
        raise FileNotFoundError(f"no sidecars in {npy_dir}")
    if n_windows is None:
        n_windows = len(sidecars) - window_offset
    chosen = sidecars[window_offset : window_offset + n_windows]
    if not chosen:
        raise ValueError(f"empty window slice offset={window_offset} n={n_windows}")

    chunks: list[NDArray[np.floating]] = []
    t0: float | None = None
    t1 = 0.0
    f0 = freq_lo_hz
    f1 = freq_hi_hz
    for sc in chosen:
        meta = json.loads(sc.read_text())
        disp_name = meta.get("display_npy") or f"{sc.stem}_display.npy"
        disp_path = npy_dir / disp_name
        if not disp_path.is_file():
            raise FileNotFoundError(disp_path)
        arr = load_display(disp_path).astype(float)
        # Map Hz → row indices on the analysis axis
        meta_f0 = float(meta.get("start_freq", 0.0))
        meta_f1 = float(meta.get("end_freq", arr.shape[0] - 1))
        n_rows = arr.shape[0]
        if n_rows <= 1 or meta_f1 <= meta_f0:
            lo_i, hi_i = 0, n_rows
        else:
            lo_i = int(max(0, np.floor((freq_lo_hz - meta_f0) / (meta_f1 - meta_f0) * (n_rows - 1))))
            hi_i = int(min(n_rows, np.ceil((freq_hi_hz - meta_f0) / (meta_f1 - meta_f0) * (n_rows - 1)) + 1))
            if hi_i <= lo_i:
                lo_i, hi_i = 0, n_rows
        arr = arr[lo_i:hi_i, :]
        f0 = freq_lo_hz
        f1 = freq_hi_hz
        start = (int(meta["start_sample"]) - 1) / float(meta["sample_freq"])
        sec_per_bin = float(meta.get("overlap", meta.get("hop", 1))) / float(meta["sample_freq"])
        # Prefer absolute time from start_sample
        if t0 is None:
            t0 = start
        t1 = start + arr.shape[1] * sec_per_bin
        chunks.append(arr)
    spec = np.concatenate(chunks, axis=1)
    assert t0 is not None
    return spec, float(t0), float(t1), float(f0), float(f1)


def plot_five_min_panels(
    placements: Sequence[Placement],
    *,
    out_dir: Path,
    gpl_dir: Path = GPL_WAV23,
    n_panels: int = N_FIVE_MIN_PANELS,
    dpi: int = 140,
    freq_lo_hz: float = FREQ_LO_HZ,
    freq_hi_hz: float = FREQ_HI_HZ,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = plt.cm.tab20.colors
    for panel in range(1, n_panels + 1):
        offset = (panel - 1) * WINDOWS_PER_FIVE_MIN
        t0_min = (panel - 1) * 5
        t1_min = panel * 5
        fig, (ax_s, ax_m) = plt.subplots(
            2,
            1,
            figsize=(14, 4.4),
            gridspec_kw={"height_ratios": [3.4, 0.9], "hspace": 0.08},
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
            f"(GPL display {freq_lo_hz:.0f}–{freq_hi_hz:.0f} Hz)"
        )
        ax_s.tick_params(labelbottom=False)

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
        ax_m.set_title(f"{len(in_panel)} placements, {len(seen)} ids", fontsize=9, loc="left")
        if seen:
            handles = [
                Patch(facecolor=c, edgecolor="none", label=f"M{m}")
                for m, c in sorted(seen.items())[:12]
            ]
            ax_m.legend(handles=handles, loc="upper right", ncol=6, fontsize=6, framealpha=0.7)
        path = out_dir / f"panel_{panel:02d}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
