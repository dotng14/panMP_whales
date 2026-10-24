"""Write and load whitened spectrograms, the display panel, and detection tables."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from .gpl_signal import DetectorParams


def matlab_datenum_from_datetime(dt: datetime) -> float:
    # MATLAB datenum epoch is 0000-01-00; Python ordinal epoch is 0001-01-01.
    return dt.toordinal() + 366 + (
        dt.hour / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0 + dt.microsecond / 86400.0 / 1e6
    )


DEFAULT_INTENSITY_CLIP_HI_PERCENTILE = 100.0


def clip_intensity_percentile(
    sp: np.ndarray,
    *,
    lo: float = 0.0,
    hi_percentile: float = DEFAULT_INTENSITY_CLIP_HI_PERCENTILE,
) -> tuple[np.ndarray, float, float]:
    """Clip spectrogram intensities to ``[lo, percentile(sp, hi_percentile)]``."""
    sp = np.asarray(sp, dtype=np.float64)
    hi = float(np.percentile(sp, hi_percentile))
    if hi < lo:
        hi = lo
    return np.clip(sp, lo, hi).astype(np.float32), lo, hi


def display_from_spwhiten(sp_whiten: np.ndarray) -> np.ndarray:
    """Cube-root display transform applied after intensity clipping."""
    return np.power(np.abs(sp_whiten).astype(np.float64), 1.0 / 3.0).astype(np.float32)


class OutputWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        intensity_clip_hi_percentile: float = DEFAULT_INTENSITY_CLIP_HI_PERCENTILE,
    ) -> None:
        self.output_dir = output_dir
        self.intensity_clip_hi_percentile = intensity_clip_hi_percentile
        self.npy_dir = output_dir / "spectrogram_npy"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.npy_dir.mkdir(parents=True, exist_ok=True)

    def write_window_spectrogram_npy(
        self,
        file_stem: str,
        window_index: int,
        sp_whiten: np.ndarray,
        parm: DetectorParams,
        start_sample: int,
        end_sample: int,
    ) -> Path:
        base = f"spwhiten_{file_stem}_w{window_index:06d}_s{start_sample:09d}_e{end_sample:09d}"
        npy_path = self.npy_dir / f"{base}.npy"
        # Keep the analysis array raw. Panel v3 applies one global clip after
        # stitching all windows; clipping each window here creates visible seams.
        sp_raw = np.asarray(sp_whiten, dtype=np.float32)
        np.save(npy_path, sp_raw)

        # Clipping is only for the offline display/BMP preview.
        sp_export, clip_lo, clip_hi = clip_intensity_percentile(
            sp_whiten, hi_percentile=self.intensity_clip_hi_percentile
        )

        bmp_path = self.npy_dir / f"{base}.bmp"
        disp_bins = display_from_spwhiten(sp_export)
        disp = disp_bins
        display_npy_path = self.npy_dir / f"{base}_display.npy"
        np.save(display_npy_path, disp)
        plt.imsave(bmp_path, disp, cmap="jet", origin="lower", format="bmp")

        meta = {
            "rows": int(sp_raw.shape[0]),
            "cols": int(sp_raw.shape[1]),
            "npy_is_clipped": False,
            "sample_freq": parm.sample_freq,
            "fftl": parm.fftl,
            "overlap": parm.fft_overlap,
            "start_freq": parm.freq_lo,
            "end_freq": parm.freq_hi,
            "window_length_sec": parm.window_length,
            "noise_ceiling": parm.noise_ceiling,
            "thresh": parm.thresh,
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "intensity_clip_lo": clip_lo,
            "intensity_clip_hi": clip_hi,
            "intensity_clip_hi_percentile": self.intensity_clip_hi_percentile,
            "preview_bmp": bmp_path.name,
            "display_npy": display_npy_path.name,
            "display_rows": int(disp.shape[0]),
            "analysis_rows": int(sp_raw.shape[0]),
            "display_y_axis": "analysis_bins",
            "display_transform": (
                f"clip[0, p{self.intensity_clip_hi_percentile:g}]; "
                "abs(sp_whiten)^(1/3); jet, origin=lower"
            ),
        }
        (self.npy_dir / f"{base}.json").write_text(json.dumps(meta, indent=2) + "\n")
        return npy_path

    def write_detection_outputs(
        self, clean_name: str, detections: list[dict[str, Any]], parm: DetectorParams
    ) -> tuple[Path, Path, Path]:
        mat_path = self.output_dir / f"Detections_{clean_name}.mat"
        tlab_path = self.output_dir / f"Labels_{clean_name}.tlab"
        csv_path = self.output_dir / f"CSV_{clean_name}.csv"

        if detections:
            times_struct = []
            labels = []
            for d in detections:
                times_struct.append(
                    {
                        "start_time": d["start_time"],
                        "end_time": d["end_time"],
                        "julian_start_time": d["julian_start_time"],
                        "julian_end_time": d["julian_end_time"],
                        "start_freq_hz": d.get("start_freq_hz", np.nan),
                        "end_freq_hz": d.get("end_freq_hz", np.nan),
                        "min_freq_hz": d.get("min_freq_hz", np.nan),
                        "max_freq_hz": d.get("max_freq_hz", np.nan),
                        "energy": d.get("energy", np.array([])),
                        "fname": d["fname"],
                    }
                )
                labels.append(int(d.get("boundary_cross_flag", 0)))
            savemat(
                mat_path,
                {"Times": np.array(times_struct, dtype=object), "Labels": np.array(labels, dtype=object), "parm": asdict(parm)},
                do_compression=True,
            )
            with tlab_path.open("w") as f:
                for d, lbl in zip(detections, labels):
                    # Keep simple: absolute MATLAB datenums in file.
                    f.write(f"{d['julian_start_time']:.10f}\t{d['julian_end_time']:.10f}\t{lbl}\n")

            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "start_time",
                        "end_time",
                        "julian_start_time",
                        "julian_end_time",
                        "start_freq_hz",
                        "end_freq_hz",
                        "min_freq_hz",
                        "max_freq_hz",
                        "boundary_cross_flag",
                        "fname",
                    ],
                )
                writer.writeheader()
                for d in detections:
                    writer.writerow(
                        {
                            "start_time": d["start_time"],
                            "end_time": d["end_time"],
                            "julian_start_time": d["julian_start_time"],
                            "julian_end_time": d["julian_end_time"],
                            "start_freq_hz": d.get("start_freq_hz", np.nan),
                            "end_freq_hz": d.get("end_freq_hz", np.nan),
                            "min_freq_hz": d.get("min_freq_hz", np.nan),
                            "max_freq_hz": d.get("max_freq_hz", np.nan),
                            "boundary_cross_flag": d.get("boundary_cross_flag", 0),
                            "fname": d["fname"],
                        }
                    )
        else:
            mat_path.write_bytes(b"")
            tlab_path.write_text("")
            csv_path.write_text("")
        return mat_path, tlab_path, csv_path

import json
import os
import re
import struct
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmin


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
