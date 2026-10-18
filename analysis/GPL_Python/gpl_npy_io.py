"""Load and plot GPL_Python spectrogram window exports (.npy + sidecar JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WindowPaths:
    raw_npy_path: Path
    display_npy_path: Path
    sidecar_json_path: Path
    meta: dict[str, Any]


def load_sidecar(json_path: Path) -> dict[str, Any]:
    return json.loads(json_path.read_text())


def _resolve_freq_bin_bounds(
    meta: dict[str, Any],
    shape_rows: int,
    freq_bin_min: int | None,
    freq_bin_max: int | None,
) -> tuple[int, int]:
    if freq_bin_min is None and freq_bin_max is None:
        return 0, shape_rows - 1
    lo = 0 if freq_bin_min is None else freq_bin_min
    hi = shape_rows - 1 if freq_bin_max is None else freq_bin_max
    if lo < 0:
        raise ValueError(f"freq_bin_min must be >= 0, got {lo}")
    if hi < lo:
        raise ValueError(f"freq_bin_max must be >= freq_bin_min, got {lo}..{hi}")
    if lo >= shape_rows:
        raise ValueError(f"freq_bin_min={lo} is outside spectrogram with {shape_rows} bins")
    hi = min(hi, shape_rows - 1)
    return lo, hi


def load_raw(
    path: Path,
    *,
    freq_bin_min: int | None = None,
    freq_bin_max: int | None = None,
) -> np.ndarray:
    sp = np.load(path)
    if sp.ndim != 2:
        raise ValueError(f"Expected 2-D array in {path}, got shape {sp.shape}")
    lo, hi = _resolve_freq_bin_bounds({}, sp.shape[0], freq_bin_min, freq_bin_max)
    return sp[lo : hi + 1, :]


def load_display(
    path: Path,
    *,
    freq_bin_min: int | None = None,
    freq_bin_max: int | None = None,
) -> np.ndarray:
    return load_raw(path, freq_bin_min=freq_bin_min, freq_bin_max=freq_bin_max)


def resolve_window_paths(npy_dir: Path, display_name_or_stem: str) -> WindowPaths:
    npy_dir = npy_dir.expanduser().resolve()
    name = display_name_or_stem
    if name.endswith("_display.npy"):
        display_path = npy_dir / name
        stem = name.removesuffix("_display.npy")
    elif name.endswith(".npy"):
        stem = name.removesuffix(".npy")
        display_path = npy_dir / f"{stem}_display.npy"
    else:
        stem = name
        display_path = npy_dir / f"{stem}_display.npy"
    raw_path = npy_dir / f"{stem}.npy"
    json_path = npy_dir / f"{stem}.json"
    if not display_path.is_file():
        raise FileNotFoundError(f"Missing display npy: {display_path}")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing raw npy: {raw_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing sidecar JSON: {json_path}")
    meta = load_sidecar(json_path)
    return WindowPaths(
        raw_npy_path=raw_path,
        display_npy_path=display_path,
        sidecar_json_path=json_path,
        meta=meta,
    )


def matlab_gpl31_n_display_rows(freq_lo: float, freq_hi: float) -> int:
    """Row count for GPL_v31.m ``linspace(freq_lo, freq_hi, freq_hi - freq_lo + 1)``."""
    return max(1, int(round(freq_hi - freq_lo + 1)))


def resample_spectrogram_matlab_gpl31_axis(
    sp: np.ndarray,
    freq_lo: float,
    freq_hi: float,
) -> np.ndarray:
    """Vertically resample spectrogram rows to match MATLAB GPL_v31 ``imagesc`` y-axis."""
    sp = np.asarray(sp, dtype=np.float64)
    n_out = matlab_gpl31_n_display_rows(freq_lo, freq_hi)
    n_in = int(sp.shape[0])
    if n_in == n_out:
        return sp
    if n_in == 1:
        return np.repeat(sp, n_out, axis=0)
    x_old = np.arange(n_in, dtype=np.float64)
    x_new = np.linspace(0.0, n_in - 1.0, n_out)
    cols = int(sp.shape[1])
    out = np.empty((n_out, cols), dtype=np.float64)
    for j in range(cols):
        out[:, j] = np.interp(x_new, x_old, sp[:, j])
    return out


def freq_axis_hz(meta: dict[str, Any], n_rows: int | None = None) -> np.ndarray:
    rows = int(n_rows if n_rows is not None else meta.get("rows", 0))
    if rows <= 0:
        raise ValueError("Cannot build frequency axis: rows must be positive")
    f0 = float(meta.get("start_freq", 0.0))
    f1 = float(meta.get("end_freq", rows - 1))
    if rows == 1:
        return np.array([f0], dtype=float)
    return np.linspace(f0, f1, rows, dtype=float)


def matlab_gpl31_freq_axis_hz(meta: dict[str, Any]) -> np.ndarray:
    """Frequency centers for MATLAB GPL_v31 display grid (one row per Hz step)."""
    f0 = float(meta.get("start_freq", 0.0))
    f1 = float(meta.get("end_freq", f0))
    n_rows = matlab_gpl31_n_display_rows(f0, f1)
    return np.linspace(f0, f1, n_rows, dtype=float)


def imshow_extent(meta: dict[str, Any], shape: tuple[int, ...]) -> list[float]:
    n_rows = int(shape[0])
    n_cols = int(shape[1])
    f0, f1 = float(meta.get("start_freq", 0.0)), float(meta.get("end_freq", n_rows - 1))
    window_sec = float(meta.get("window_length_sec", n_cols))
    return [0.0, window_sec, f0, f1]


def matlab_gpl31_imshow_extent(meta: dict[str, Any], n_cols: int) -> list[float]:
    """Extent for resampled GPL_v31-style display (y-axis spans freq_lo..freq_hi)."""
    f0 = float(meta.get("start_freq", 0.0))
    f1 = float(meta.get("end_freq", f0))
    window_sec = float(meta.get("window_length_sec", n_cols))
    return [0.0, window_sec, f0, f1]


def analysis_band_slice(meta: dict[str, Any]) -> slice | None:
    """Return slice into exported rows for the detection analysis band, if metadata defines it."""
    if "analysis_freq_bin_lo" not in meta:
        return None
    lo = int(meta["analysis_freq_bin_lo"])
    hi = int(meta["analysis_freq_bin_hi"])
    return slice(lo, hi + 1)


def is_legacy_band_limited_export(meta: dict[str, Any]) -> bool:
    """True when sidecar lacks full-spectrum fields (pre-refactor exports)."""
    return "analysis_freq_lo" not in meta
