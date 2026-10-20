"""Load GPL_Python window spectrograms and detection CSVs; map samples ↔ time bins."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.objects import Window


@dataclass(frozen=True)
class Detection:
    """One GPL_Python CSV detection (file-global 1-based sample times)."""

    start_time: int
    end_time: int
    index: int
    fname: str = ""
    boundary_cross_flag: int = 0
    raw: dict[str, Any] | None = None


def load_sidecar(json_path: Path) -> dict[str, Any]:
    return json.loads(Path(json_path).read_text())


def load_window(
    npy_path: Path,
    json_path: Path | None = None,
    *,
    window_id: str | None = None,
) -> Window:
    npy_path = Path(npy_path)
    if json_path is None:
        json_path = npy_path.with_suffix(".json")
    json_path = Path(json_path)
    meta = load_sidecar(json_path)
    data = np.load(npy_path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D spectrogram in {npy_path}, got {data.shape}")
    wid = window_id or npy_path.stem
    return Window(
        window_id=wid,
        data=np.asarray(data, dtype=np.float64),
        start_sample=int(meta["start_sample"]),
        end_sample=int(meta["end_sample"]),
        overlap=int(meta["overlap"]),
        fftl=int(meta.get("fftl", 0)),
        sample_freq=float(meta.get("sample_freq", 0.0)),
        start_freq=float(meta.get("start_freq", 0.0)),
        end_freq=float(meta.get("end_freq", 0.0)),
        npy_path=npy_path,
        meta=meta,
    )


def discover_windows(npy_dir: Path) -> list[Window]:
    npy_dir = Path(npy_dir)
    windows: list[Window] = []
    for npy_path in sorted(npy_dir.glob("*.npy")):
        if npy_path.name.endswith("_display.npy"):
            continue
        json_path = npy_path.with_suffix(".json")
        if not json_path.is_file():
            continue
        windows.append(load_window(npy_path, json_path))
    return windows


def load_detections_csv(csv_path: Path) -> list[Detection]:
    csv_path = Path(csv_path)
    detections: list[Detection] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            st = int(float(row["start_time"]))
            et = int(float(row["end_time"]))
            flag = int(float(row.get("boundary_cross_flag") or 0))
            detections.append(
                Detection(
                    start_time=st,
                    end_time=et,
                    index=i,
                    fname=str(row.get("fname") or ""),
                    boundary_cross_flag=flag,
                    raw=dict(row),
                )
            )
    return detections


def find_detection_csvs(output_dir: Path) -> list[Path]:
    return sorted(Path(output_dir).glob("CSV_*.csv"))


def global_samples_to_time_bins(
    global_start: int,
    global_end: int,
    *,
    win_start_sample: int,
    overlap: int,
    n_cols: int,
) -> tuple[int, int] | None:
    """Map file-global 1-based sample interval to window-local ``[t0, t1)``."""
    if overlap <= 0:
        raise ValueError(f"overlap must be positive, got {overlap}")
    if global_end < global_start:
        return None
    local_start_1 = global_start - win_start_sample + 1
    local_end_1 = global_end - win_start_sample + 1
    t0 = max(0, (local_start_1 - 1) // overlap)
    t1 = min(n_cols, local_end_1 // overlap)
    if t1 <= t0:
        return None
    return int(t0), int(t1)


def detection_overlaps_window(det: Detection, window: Window) -> bool:
    return not (det.end_time < window.start_sample or det.start_time > window.end_sample)


def load_windows_from_paths(paths: Sequence[Path | str]) -> list[Window]:
    return [load_window(Path(p)) for p in paths]


def load_detections_from_paths(paths: Iterable[Path | str]) -> list[Detection]:
    out: list[Detection] = []
    for p in paths:
        out.extend(load_detections_csv(Path(p)))
    return out


def load_gpl_inputs(
    *,
    window_paths: Sequence[Path | str] | None = None,
    npy_dir: Path | str | None = None,
    csv_paths: Sequence[Path | str] | None = None,
    gpl_output_dir: Path | str | None = None,
) -> tuple[list[Window], list[Detection]]:
    """Load windows and detections from explicit paths or by directory discovery.

    Windows come from ``window_paths`` if given, else discovered in ``npy_dir``.
    Detections come from ``csv_paths`` if given, else from ``CSV_*.csv`` files in
    ``gpl_output_dir``; with neither, the detection list is empty.
    """
    if window_paths is not None:
        windows = load_windows_from_paths(window_paths)
    elif npy_dir is not None:
        windows = discover_windows(Path(npy_dir))
    else:
        raise ValueError("window_paths or npy_dir is required")

    if csv_paths is not None:
        detections = load_detections_from_paths(csv_paths)
    elif gpl_output_dir is not None:
        detections = load_detections_from_paths(find_detection_csvs(Path(gpl_output_dir)))
    else:
        detections = []

    return windows, detections
