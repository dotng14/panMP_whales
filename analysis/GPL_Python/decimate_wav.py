#!/usr/bin/env python3
"""Decimate a WAV from 200 kHz to 10 kHz.

The ratio is an integer 20. ``scipy.signal.resample_poly`` low-passes below
the new Nyquist (5 kHz) and keeps every 20th sample.

Example:
    python -m analysis.GPL_Python.decimate_wav input.wav -o input_10kHz.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SOURCE_RATE = 200_000
TARGET_RATE = 10_000


def integer_wav_scale(dtype: np.dtype) -> float:
    """Full-scale divisor for signed integer PCM (MATLAB ``audioread``)."""
    if np.issubdtype(dtype, np.integer) and dtype.itemsize == 2:
        return 32768.0
    if np.issubdtype(dtype, np.integer):
        return float(2 ** (dtype.itemsize * 8 - 1))
    return 1.0


def read_wav(path: Path) -> tuple[np.ndarray, int, np.dtype]:
    sample_rate, data = wavfile.read(path)
    dtype = data.dtype
    if np.issubdtype(dtype, np.integer):
        data = data.astype(np.float64) / integer_wav_scale(dtype)
    else:
        data = np.asarray(data, dtype=np.float64)
    return data, int(sample_rate), dtype


def write_wav(path: Path, data: np.ndarray, sample_rate: int, dtype: np.dtype) -> None:
    if np.issubdtype(dtype, np.integer):
        scale = integer_wav_scale(dtype)
        info = np.iinfo(dtype)
        pcm = np.clip(np.rint(data * scale), info.min, info.max).astype(dtype)
        wavfile.write(path, sample_rate, pcm)
        return
    wavfile.write(path, sample_rate, np.asarray(data, dtype=np.float32))


def decimate_audio(
    data: np.ndarray,
    source_rate: int = SOURCE_RATE,
    target_rate: int = TARGET_RATE,
) -> np.ndarray:
    """Anti-alias and downsample along axis 0."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"sample rates must be positive, got {source_rate} -> {target_rate}")
    if source_rate % target_rate != 0:
        raise ValueError(
            f"sample-rate ratio must be an integer, got {source_rate} / {target_rate}"
        )
    down = source_rate // target_rate
    if down == 1:
        return np.array(data, copy=True)
    return resample_poly(data, up=1, down=down, axis=0)


def decimate_wav(
    in_path: Path,
    out_path: Path,
    *,
    source_rate: int = SOURCE_RATE,
    target_rate: int = TARGET_RATE,
) -> tuple[int, int, int]:
    """Decimate ``in_path`` and write ``out_path``. Returns (n_in, n_out, rate)."""
    data, file_rate, dtype = read_wav(in_path)
    if file_rate != source_rate:
        raise ValueError(
            f"{in_path} is {file_rate} Hz, expected {source_rate} Hz"
        )
    decimated = decimate_audio(data, source_rate, target_rate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_path, decimated, target_rate, dtype)
    return int(data.shape[0]), int(decimated.shape[0]), target_rate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decimate a WAV from 200 kHz to 10 kHz (factor 20).",
    )
    parser.add_argument("wav", type=Path, help="Input WAV, expected at 200 kHz")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output WAV (default: <input stem>_10kHz.wav beside the input)",
    )
    parser.add_argument("--source-rate", type=int, default=SOURCE_RATE)
    parser.add_argument("--target-rate", type=int, default=TARGET_RATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    in_path = args.wav.expanduser().resolve()
    out_path = args.output
    if out_path is None:
        out_path = in_path.with_name(f"{in_path.stem}_10kHz.wav")
    else:
        out_path = out_path.expanduser().resolve()
    n_in, n_out, rate = decimate_wav(
        in_path,
        out_path,
        source_rate=args.source_rate,
        target_rate=args.target_rate,
    )
    print(
        f"wrote {out_path}  {n_in} -> {n_out} samples at {rate} Hz "
        f"({n_out / rate:.2f} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
