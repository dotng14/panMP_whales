"""Synthetic inputs for the CLI tests: GPL output folders, WAV files and series.

A GPL folder has the layout ``GPL_Python`` writes: ``spectrogram_npy/`` with
``spwhiten_*.npy`` windows and JSON sidecars, plus a ``CSV_*.csv`` detection
table. Motif shapes are planted at known places in low noise, and the
detections sit on the planted copies, so the builders find real atoms.
"""

from __future__ import annotations

import csv
import json
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# This checkout. Subprocesses run from here with it first on PYTHONPATH, so
# `python -m analysis...` never picks up another installed checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_FREQ = 1000.0
SILENT_COLUMNS = 8
HOP = 10
FFTL = 20


@dataclass
class GplFolder:
    """Where a synthetic GPL folder was written and what was planted in it."""

    root: Path
    n_freq: int
    n_time: int
    motifs: list[np.ndarray]
    # (window index, motif index, column) for every planted copy.
    copies: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def npy_dir(self) -> Path:
        return self.root / "spectrogram_npy"


def motif_shapes(n_freq: int, width: int, n_motifs: int) -> list[np.ndarray]:
    """Distinct, roughly orthogonal time–frequency shapes."""
    shapes = []
    t = np.linspace(0.0, 1.0, width)
    for k in range(n_motifs):
        shape = np.zeros((n_freq, width))
        band = slice((k * n_freq) // n_motifs, ((k + 1) * n_freq) // n_motifs)
        shape[band, :] = np.sin(np.pi * (k + 1) * t)[None, :]
        shapes.append(shape)
    return shapes


def write_gpl_folder(
    root: Path,
    *,
    n_windows: int = 3,
    n_freq: int = 12,
    n_time: int = 240,
    n_motifs: int = 3,
    width: int = 12,
    copies_per_window: int = 4,
    noise: float = 0.05,
    detections: str = "planted",
    sidecar_sample_freq: bool = True,
    seed: int = 0,
) -> GplFolder:
    """Write a GPL output folder.

    The last ``SILENT_COLUMNS`` columns of every window are exactly zero.
    ``detections`` picks the CSV:

    - ``planted``: one detection over every planted copy, plus per window one
      over noise, one wider than the default width limit, one on the silent
      stretch, and one shorter than an STFT hop.
    - ``narrow``: detections only 2 columns wide (below any width limit).
    - ``empty``: a CSV with a header and no rows.
    - ``none``: no CSV at all.

    ``sidecar_sample_freq=False`` leaves ``sample_freq`` out of the sidecars.
    """
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    npy_dir = root / "spectrogram_npy"
    npy_dir.mkdir(exist_ok=True)
    motifs = motif_shapes(n_freq, width, n_motifs)
    folder = GplFolder(root=root, n_freq=n_freq, n_time=n_time, motifs=motifs)
    rows: list[dict[str, object]] = []
    window_samples = n_time * HOP
    body = n_time - SILENT_COLUMNS
    slot = body // copies_per_window
    for w in range(n_windows):
        start_sample = 1 + w * window_samples
        data = noise * rng.standard_normal((n_freq, n_time))
        data[:, body:] = 0.0
        for c in range(copies_per_window):
            motif = (w + c) % n_motifs
            column = c * slot + int(rng.integers(2, max(3, slot - width - 2)))
            data[:, column : column + width] += (1.0 + 0.1 * rng.standard_normal()) * motifs[motif]
            folder.copies.append((w, motif, column))
            span = width if detections != "narrow" else 2
            rows.append(_detection(start_sample, column, span))
        if detections == "planted":
            rows.append(_detection(start_sample, body - width - 1, width))
            rows.append(_detection(start_sample, 0, 2 * width + 1))
            rows.append(_detection(start_sample, body + 2, 4))
            sub_hop = start_sample + 3
            rows.append({**_detection(start_sample, 0, 1), "start_time": sub_hop, "end_time": sub_hop + 4})
        stem = f"spwhiten_synthetic_w{w + 1:06d}_s{start_sample:09d}_e{start_sample + window_samples - 1:09d}"
        np.save(npy_dir / f"{stem}.npy", data.astype(np.float32))
        sidecar = {
            "rows": n_freq,
            "cols": n_time,
            "sample_freq": SAMPLE_FREQ,
            "fftl": FFTL,
            "overlap": HOP,
            "start_sample": start_sample,
            "end_sample": start_sample + window_samples - 1,
            "start_freq": 10.0,
            "end_freq": 10.0 + 10.0 * n_freq,
        }
        if not sidecar_sample_freq:
            del sidecar["sample_freq"]
        (npy_dir / f"{stem}.json").write_text(json.dumps(sidecar))
        # Display arrays sit next to the windows in real GPL output; loaders skip them.
        np.save(npy_dir / f"{stem}_display.npy", np.zeros((2, 2), dtype=np.uint8))
    # An array without a sidecar: window loaders must skip it.
    np.save(npy_dir / "notes.npy", np.zeros(3))
    if detections != "none":
        with (root / "CSV_synthetic.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["start_time", "end_time", "boundary_cross_flag", "fname"]
            )
            writer.writeheader()
            if detections != "empty":
                writer.writerows(rows)
    return folder


def _detection(window_start_sample: int, column: int, span: int) -> dict[str, object]:
    start = window_start_sample + column * HOP
    return {
        "start_time": start,
        "end_time": start + span * HOP - 1,
        "boundary_cross_flag": 0,
        "fname": "synthetic.wav",
    }


def planted_series(
    n: int = 600,
    *,
    rows: int = 1,
    length: int = 24,
    starts: tuple[int, ...] = (40, 220, 420),
    noise: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """``(rows, n)`` noise with one shape repeated at ``starts`` (1-D when ``rows == 1``)."""
    rng = np.random.default_rng(seed)
    series = noise * rng.standard_normal((rows, n))
    t = np.linspace(0.0, 1.0, length)
    for r in range(rows):
        shape = np.sin(2.0 * np.pi * (r + 1) * t)
        for start in starts:
            series[r, start : start + length] += shape
    return series[0] if rows == 1 else series


def write_wav(
    path: Path, *, sample_rate: int = 2000, seconds: float = 6.0, channels: int = 1, seed: int = 0
) -> Path:
    """16-bit WAV: noise plus a 150 Hz burst and a 350 Hz burst, each repeated."""
    rng = np.random.default_rng(seed)
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    audio = 0.02 * rng.standard_normal(n)
    for start_s in (0.5, 2.5, 4.5):
        on = (t >= start_s) & (t < start_s + 0.4)
        audio[on] += 0.5 * np.sin(2.0 * np.pi * 150.0 * t[on])
        on = (t >= start_s + 0.6) & (t < start_s + 0.9)
        audio[on] += 0.5 * np.sin(2.0 * np.pi * 350.0 * t[on])
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    pcm = np.repeat(pcm[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def run_module(*args: str):
    """Run ``python -m <args>`` from this checkout and return the completed process."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])
    ))
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
