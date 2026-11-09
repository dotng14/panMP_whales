"""Signals the pan matrix profile runs on, as ``(rows, T)`` arrays.

- A saved ``.npy`` series, 1-D or ``(rows, T)``.
- A WAV: its RMS envelope (one row) or band energy (``analysis.bands``).
- A GPL output folder, from the whitened spectrogram: summed over frequency
  (one row), summed within frequency bands, projected onto the recording's
  top frequency patterns (``pca``), or every frequency row. With a
  ``rate`` the columns are averaged into bins of ``1 / rate`` seconds by the
  centre time of each STFT frame, as the song HMM does; without one the
  native STFT columns are kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis.bands import band_energy

from . import rms_envelope

WAV_KINDS = ("rms", "bands")
GPL_KINDS = ("energy", "bands", "pca", "full")
DEFAULT_PCA_ROWS = 8


@dataclass
class Signal:
    """A ``(rows, T)`` signal plus how it was built."""

    data: np.ndarray
    rate: float | None
    info: dict = field(default_factory=dict)


def load_series(path: Path) -> Signal:
    data = np.load(path)
    if data.ndim not in (1, 2):
        raise ValueError(f"{path} must hold a 1-D or (rows, T) array, got shape {data.shape}")
    rows = np.atleast_2d(np.asarray(data, dtype=np.float64))
    return Signal(rows, None, {"source": "series", "path": str(path)})


def load_wav(path: Path, rate: int, kind: str) -> Signal:
    from scipy.io import wavfile

    if kind not in WAV_KINDS:
        raise ValueError(f"WAV signal must be one of {WAV_KINDS}, got {kind!r}")
    sample_rate, samples = wavfile.read(path)
    if kind == "rms":
        data = rms_envelope(samples, int(sample_rate), int(rate))[np.newaxis, :]
    else:
        data = band_energy(samples, int(sample_rate), int(rate))
    info = {"source": "wav", "path": str(path), "kind": kind, "sample_rate": int(sample_rate)}
    return Signal(np.asarray(data, dtype=np.float64), float(rate), info)


def _gpl_windows(gpl_dir: Path) -> list[tuple[np.ndarray, dict]]:
    npy_dir = gpl_dir / "spectrogram_npy" if (gpl_dir / "spectrogram_npy").is_dir() else gpl_dir
    windows = []
    for path in sorted(npy_dir.glob("spwhiten_*.npy")):
        if path.name.endswith("_display.npy"):
            continue
        meta = json.loads(path.with_suffix(".json").read_text())
        windows.append((np.asarray(np.load(path), dtype=np.float64), meta))
    if not windows:
        raise ValueError(f"no whitened spectrogram windows in {npy_dir}")
    windows.sort(key=lambda item: int(item[1]["start_sample"]))
    return windows


def default_band_edges(start_freq: float, end_freq: float, width: float = 100.0) -> list[float]:
    """Edges every ``width`` Hz from ``start_freq``; the last band ends at ``end_freq``."""
    edges = list(np.arange(float(start_freq), float(end_freq), float(width)))
    return edges + [float(end_freq)]


def _band_rows(spec: np.ndarray, freqs: np.ndarray, edges: list[float]) -> np.ndarray:
    out = np.zeros((len(edges) - 1, spec.shape[1]))
    for band, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = band == len(edges) - 2
        mask = (freqs >= lo) & ((freqs <= hi) if last else (freqs < hi))
        out[band] = spec[mask].sum(axis=0)
    return out


def frequency_pca(blocks: list[np.ndarray], n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Top ``n_rows`` frequency patterns of ``(F, T_k)`` blocks taken together.

    Every column is one sample in ``F`` dimensions. Returns the orthonormal
    ``(n_rows, F)`` basis, strongest pattern first, and the ``(F,)`` mean
    column. Each pattern's sign is fixed so its largest entry is positive.
    """
    n_freq = blocks[0].shape[0]
    if not 1 <= n_rows <= n_freq:
        raise ValueError(f"PCA rows must be between 1 and {n_freq} (frequency rows), got {n_rows}")
    n_cols = sum(block.shape[1] for block in blocks)
    mean = sum(block.sum(axis=1) for block in blocks) / n_cols
    scatter = np.zeros((n_freq, n_freq))
    for block in blocks:
        centred = block - mean[:, None]
        scatter += centred @ centred.T
    values, vectors = np.linalg.eigh(scatter)
    basis = vectors[:, np.argsort(values)[::-1][:n_rows]].T
    peak = basis[np.arange(n_rows), np.argmax(np.abs(basis), axis=1)]
    basis *= np.where(peak < 0, -1.0, 1.0)[:, None]
    return basis, mean


def gpl_rows(
    blocks: list[np.ndarray],
    kind: str,
    *,
    start_freq: float,
    end_freq: float,
    band_edges: list[float] | None = None,
    pca_rows: int = DEFAULT_PCA_ROWS,
) -> tuple[list[np.ndarray], dict]:
    """Signal rows for each ``(F, T_k)`` whitened-spectrogram block.

    Row ``r`` of an ``F``-row block sits at
    ``start_freq + r * (end_freq - start_freq) / (F - 1)`` Hz. Returns one
    ``(rows, T_k)`` array per block and how they were built.
    """
    if kind not in GPL_KINDS:
        raise ValueError(f"GPL signal must be one of {GPL_KINDS}, got {kind!r}")  # no-pipeline-cover: load_gpl and argparse check the kind first
    n_freq = blocks[0].shape[0]
    if any(block.shape[0] != n_freq for block in blocks):
        raise ValueError("whitened windows have different frequency rows")
    info: dict = {"kind": kind, "start_freq": float(start_freq), "end_freq": float(end_freq), "band_edges": None}
    if kind == "energy":
        return [block.sum(axis=0, keepdims=True) for block in blocks], info
    if kind == "full":
        return list(blocks), info
    if kind == "pca":
        basis, mean = frequency_pca(blocks, int(pca_rows))
        info["pca_rows"] = int(pca_rows)
        return [basis @ (block - mean[:, None]) for block in blocks], info
    edges = list(band_edges) if band_edges else default_band_edges(start_freq, end_freq)
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError(f"band edges must increase, got {edges}")
    freqs = np.linspace(float(start_freq), float(end_freq), n_freq)
    info["band_edges"] = edges
    return [_band_rows(block, freqs, edges) for block in blocks], info


def decimate_columns(rows: np.ndarray, factor: int) -> np.ndarray:
    """Average every ``factor`` columns of ``(R, T)``; a partial last group is dropped.

    Column ``s`` of the result covers input columns ``[s * factor, (s + 1) * factor)``.
    """
    if factor < 1:
        raise ValueError(f"decimation factor must be >= 1, got {factor}")  # no-pipeline-cover: PanSeedConfig checks --decimate first
    if factor == 1:
        return np.asarray(rows, dtype=np.float64)
    n = rows.shape[1] // factor
    if n < 1:
        raise ValueError(f"decimation factor {factor} is longer than the signal ({rows.shape[1]} columns)")  # no-pipeline-cover: pan_profile needs a longer signal than any motif length allows
    return rows[:, : n * factor].reshape(rows.shape[0], n, factor).mean(axis=2)


def load_gpl(
    gpl_dir: Path,
    kind: str,
    *,
    rate: float | None = None,
    band_edges: list[float] | None = None,
    pca_rows: int = DEFAULT_PCA_ROWS,
) -> Signal:
    """Signal from the whitened spectrogram of a GPL output folder (see :func:`gpl_rows`)."""
    if kind not in GPL_KINDS:
        raise ValueError(f"GPL signal must be one of {GPL_KINDS}, got {kind!r}")
    windows = _gpl_windows(gpl_dir)
    meta0 = windows[0][1]
    rows_per_window, built = gpl_rows(
        [spec for spec, _ in windows],
        kind,
        start_freq=float(meta0.get("start_freq", 0.0)),
        end_freq=float(meta0.get("end_freq", 0.0)),
        band_edges=band_edges,
        pca_rows=pca_rows,
    )
    info = {"source": "gpl", "path": str(gpl_dir), **built}
    if rate is None:
        sample_rate = float(meta0["sample_freq"])
        info["native_rate"] = sample_rate / float(meta0["overlap"])
        return Signal(np.concatenate(rows_per_window, axis=1), None, info)
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    return Signal(_bin_by_centre_time(windows, rows_per_window, float(rate)), float(rate), info)


def _bin_by_centre_time(windows, rows_per_window, rate: float) -> np.ndarray:
    """Average STFT frames into ``1 / rate`` s bins; fill empty bins by interpolation."""
    last_meta = windows[-1][1]
    duration = int(last_meta["end_sample"]) / float(last_meta["sample_freq"])
    n_bins = max(1, int(np.ceil(duration * rate)))
    n_rows = rows_per_window[0].shape[0]
    total = np.zeros((n_rows, n_bins))
    count = np.zeros(n_bins)
    for (spec, meta), rows in zip(windows, rows_per_window):
        hop = int(meta["overlap"])
        sample_rate = float(meta["sample_freq"])
        fftl = int(meta.get("fftl", 0))
        start = int(meta["start_sample"]) - 1
        centre = (start + np.arange(spec.shape[1]) * hop + fftl / 2.0) / sample_rate
        index = np.minimum((centre * rate).astype(np.int64), n_bins - 1)
        np.add.at(total.T, index, rows.T)
        np.add.at(count, index, 1.0)
    filled = count > 0
    out = np.zeros_like(total)
    out[:, filled] = total[:, filled] / count[filled]
    if not np.all(filled):
        known = np.flatnonzero(filled)
        missing = np.flatnonzero(~filled)
        for r in range(n_rows):
            out[r, missing] = np.interp(missing, known, out[r, known])
    return out
