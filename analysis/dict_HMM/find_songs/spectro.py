"""Whitened spectrogram windows from a 10 kHz Site H recording.

The dictionary was learned on 200 kHz audio with a 32768-sample FFT and a
24576-sample step: a 0.16384 s frame every 0.12288 s, 6.1035 Hz per row,
rows 2..1638 of the FFT. Resampling 10 kHz audio to 12.5 kHz and using a
2048-sample FFT with a 1536-sample step gives the same frame, step, and
row spacing. A 10 kHz file holds frequencies below 5 kHz, so only the
lower rows exist.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SOURCE_RATE = 10000
RATE = 12500
FFTL = 2048
STEP = 1536
WINDOW_SEC = 75
FIRST_BIN = 2  # FFT bin of dictionary row 0
N_TIME = 609
HOP_SEC = STEP / RATE  # 0.12288


def wav_layout(path: Path) -> tuple[int, int, int]:
    """Sample rate, byte offset of the samples, and number of samples."""
    with open(path, "rb") as handle:
        head = handle.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError(f"{path} is not a wav file")
        rate = 0
        while True:
            chunk = handle.read(8)
            if len(chunk) < 8:
                raise ValueError(f"{path} has no data chunk")
            name, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if name == b"fmt ":
                body = handle.read(size)
                audio, channels, rate = struct.unpack("<HHI", body[:8])
                bits = struct.unpack("<H", body[14:16])[0]
                if audio != 1 or channels != 1 or bits != 16:
                    raise ValueError(f"{path} is not 16-bit mono PCM")
            elif name == b"data":
                return rate, handle.tell(), size // 2
            else:
                handle.seek(size + (size & 1), 1)


def read_seconds(path: Path, start_sec: float, seconds: float) -> np.ndarray:
    rate, offset, n_samples = wav_layout(path)
    if rate != SOURCE_RATE:
        raise ValueError(f"{path} is {rate} Hz, expected {SOURCE_RATE}")
    first = int(round(start_sec * rate))
    count = min(int(round(seconds * rate)), n_samples - first)
    data = np.fromfile(path, dtype="<i2", count=count, offset=offset + 2 * first)
    return data.astype(np.float64) / 32768.0


def shortest_half_mean(rows: np.ndarray) -> np.ndarray:
    """Noise level of each row: mean of the densest half of its values.

    This is `base3x` followed by the mean, as in the detector's whitening.
    """
    ordered = np.sort(rows, axis=1)
    n = ordered.shape[1] - (ordered.shape[1] % 2)
    ordered = ordered[:, :n]
    half = n // 2
    spread = ordered[:, half:] - ordered[:, :half]
    pick = np.argmin(spread, axis=1)
    index = np.arange(rows.shape[0])
    low = ordered[index, pick][:, None]
    high = ordered[index, pick + half][:, None]
    inside = (rows >= low) & (rows <= high)
    return (rows * inside).sum(axis=1) / inside.sum(axis=1)


def whiten(magnitude: np.ndarray) -> np.ndarray:
    mu = shortest_half_mean(magnitude)
    white = np.abs((magnitude - mu[:, None]) / mu[:, None])
    out = 4.0 * white
    out[1:, :] += white[:-1, :]
    out[:-1, :] += white[1:, :]
    out[:, 1:] += white[:, :-1]
    out[:, :-1] += white[:, 1:]
    return out / 8.0


def window_spectrogram(samples_10k: np.ndarray, n_rows: int) -> np.ndarray:
    """Whitened spectrogram of one 75-second window, rows 0..n_rows-1 of the dictionary."""
    audio = resample_poly(samples_10k, 5, 4)
    n_time = (audio.size - FFTL) // STEP
    frames = np.lib.stride_tricks.sliding_window_view(audio, FFTL)[::STEP][:n_time]
    spectrum = np.abs(np.fft.rfft(frames * np.hamming(FFTL), axis=1))
    # One extra row above the kept band so the smoothing sees the same neighbor.
    magnitude = spectrum[:, FIRST_BIN : FIRST_BIN + n_rows + 1].T
    return whiten(magnitude)[:n_rows]
