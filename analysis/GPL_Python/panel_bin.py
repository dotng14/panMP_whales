"""Browser display spectrogram panel (``panel.bin``).

Owned by analysis: written by :class:`~analysis.GPL_Python.process_v4.ProcessingOrchestrator`
after whitened windows are on disk. The catalog records ``panel_path``; the
backend streams the file after a catalog lookup.
"""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmin

from .gpl_outputs import clip_intensity_percentile, display_from_spwhiten

_SIDECAR_RE = re.compile(r"^spwhiten_(?P<stem>.+?)_w(?P<win>\d+)_")

MAX_DISPLAY_COLS = int(os.environ.get("GPL_MAX_DISPLAY_COLS", "65536"))
PANEL_VERSION = 3
_MAGNITUDE_PERCENTILE = 99.9
_BOUNDARY_CUTOFF_MARGIN = 8


def _header_padding(header_len: int) -> int:
    return (4 - (4 + header_len) % 4) % 4


def read_panel_header(panel_path: Path) -> dict | None:
    """Read the JSON header from panel.bin, or None if unreadable."""
    try:
        with panel_path.open("rb") as f:
            (header_len,) = struct.unpack("<I", f.read(4))
            hdr = json.loads(f.read(header_len))
        return hdr if isinstance(hdr, dict) else None
    except Exception:
        return None


def _load_window_arrays(npy_dir: Path, sidecars: list[Path]) -> tuple[list[np.ndarray], list[int]]:
    windows: list[np.ndarray] = []
    window_ends: list[int] = []
    offset = 0
    for sc in sidecars:
        meta = json.loads(sc.read_text())
        ncols = int(meta["cols"])
        arr = np.load(npy_dir / f"{sc.stem}.npy").astype(np.float32)
        windows.append(arr[:, :ncols])
        offset += ncols
        window_ends.append(offset)
    return windows, window_ends


def _filter_boundary_cutoffs(
    cutoffs: list[int],
    window_ends: list[int],
    stride: int,
    *,
    margin: int = _BOUNDARY_CUTOFF_MARGIN,
) -> list[int]:
    if len(window_ends) <= 1:
        return cutoffs
    bounds = {((end - 1) // stride) for end in window_ends[:-1]}
    return [c for c in cutoffs if not any(abs(c - b) <= margin for b in bounds)]


def build_panel(output_dir: Path, title: str) -> tuple[dict, list[bytes]]:
    """Concatenate windows, apply global display transform, decimate, quantize."""
    npy_dir = output_dir / "spectrogram_npy"
    sidecars = sorted(
        npy_dir.glob("spwhiten_*.json"),
        key=lambda p: int(_SIDECAR_RE.match(p.stem).group("win")),  # type: ignore[union-attr]
    )
    if not sidecars:
        raise FileNotFoundError(f"No sidecar JSON files found in {npy_dir}")

    windows, window_ends = _load_window_arrays(npy_dir, sidecars)
    total_samples = 0
    sample_rate = 1
    freq_lo = 0.0
    freq_hi = 0.0
    for i, sc in enumerate(sidecars):
        meta = json.loads(sc.read_text())
        total_samples += meta["end_sample"] - meta["start_sample"]
        sample_rate = meta["sample_freq"]
        if i == 0:
            freq_lo = float(meta.get("start_freq", 0.0))
            freq_hi = float(meta.get("end_freq", 0.0))

    concat = np.concatenate(windows, axis=1)
    clipped, _, _ = clip_intensity_percentile(concat, hi_percentile=_MAGNITUDE_PERCENTILE)
    display = display_from_spwhiten(clipped)

    stride = 1
    if display.shape[1] > MAX_DISPLAY_COLS:
        stride = -(-display.shape[1] // MAX_DISPLAY_COLS)
        display = display[:, ::stride]
    freq_bins, time_cols = display.shape

    energy_raw = display.sum(axis=0).astype(np.float32)
    energy_smooth = gaussian_filter1d(energy_raw, sigma=3).astype(np.float32)
    cutoffs = argrelmin(energy_smooth, order=5)[0].tolist()
    cutoffs = _filter_boundary_cutoffs(cutoffs, window_ends, stride)

    norm_max = float(np.percentile(display, _MAGNITUDE_PERCENTILE))
    if norm_max <= 0:
        norm_max = 1.0
    magnitude_u8 = np.clip(display / norm_max * 255.0, 0, 255).astype(np.uint8)

    audio_duration_sec = total_samples / sample_rate

    header = {
        "panelVersion": PANEL_VERSION,
        "freqBins": int(freq_bins),
        "timeCols": int(time_cols),
        "cutoffs": cutoffs,
        "title": title,
        "colormap": "jet",
        "brightness": 1.0,
        "normMax": norm_max,
        "audioDurationSec": audio_duration_sec,
        "freqLo": freq_lo,
        "freqHi": freq_hi,
        "arrays": [
            {"name": "magnitude", "dtype": "uint8", "length": int(magnitude_u8.size)},
            {"name": "energyRaw", "dtype": "float32", "length": int(energy_raw.size)},
            {"name": "energySmoothed", "dtype": "float32", "length": int(energy_smooth.size)},
        ],
    }

    return header, [
        magnitude_u8.tobytes(),
        energy_raw.tobytes(),
        energy_smooth.tobytes(),
    ]


def write_panel_bin(output_dir: Path, title: str) -> Path:
    """Build the panel and write ``output_dir/panel.bin``."""
    header, blobs = build_panel(output_dir, title)
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = _header_padding(len(header_json))

    panel_path = output_dir / "panel.bin"
    with panel_path.open("wb") as f:
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        f.write(b"\x00" * pad)
        for blob in blobs:
            f.write(blob)
    return panel_path
