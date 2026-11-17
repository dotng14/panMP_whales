#!/usr/bin/env python3
"""0.1 s whitened 50–500 Hz energy ratio for selected minutes.

Matches pre_processing/c/energy_filter.c: 75 s GPL windows, Hamming STFT,
densest-half whitening, cross-kernel smoothing. Segment energy is the sum of
squares of whitened cells. A bin ratio is band energy / overall energy.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

LO_HZ = 50
HI_HZ = 500
FREQ_LO_HZ = 10
FREQ_RES_HZ = 10
WINDOW_SEC = 75
SKIP_PERCENT = 75


def llround(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def ceil_log2_int(n: int) -> int:
    e, v = 0, 1
    while v < n:
        v <<= 1
        e += 1
    return e


def layout_for(rate: int) -> dict:
    fftl = 1 << ceil_log2_int((rate + FREQ_RES_HZ - 1) // FREQ_RES_HZ)
    if fftl < 16:
        fftl = 16
    hop = llround(fftl - (1.0 - SKIP_PERCENT * 1e-2) * fftl)
    if hop < 1:
        hop = 1
    nfreq = fftl // 2 + 1
    nyquist_bin = fftl // 2
    row0 = llround(FREQ_LO_HZ * fftl / rate)
    row0 = min(max(row0, 1), nyquist_bin)
    nrows = nyquist_bin - row0 + 1
    blo = llround(LO_HZ * fftl / rate)
    bhi = llround(HI_HZ * fftl / rate)
    blo = max(blo, row0)
    bhi = min(bhi, nyquist_bin)
    hamming = 0.54 - 0.46 * np.cos(2.0 * math.pi * np.arange(fftl) / (fftl - 1))
    return {
        "rate": rate,
        "fftl": fftl,
        "hop": hop,
        "row0": row0,
        "nrows": nrows,
        "band0": blo - row0,
        "band1": bhi - row0,
        "hamming": hamming.astype(np.float64),
    }


def open_wav(path: Path) -> tuple[int, int, int]:
    """Return sample rate, data-chunk byte offset, and sample count."""
    path = Path(path)
    with path.open("rb") as handle:
        hdr = handle.read(65536)
    if hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
        raise ValueError(f"not a wav file: {path}")
    rate = 0
    data_off = 0
    data_bytes = 0
    i = 12
    while i + 8 <= len(hdr):
        chunk = hdr[i : i + 4]
        sz = struct.unpack_from("<I", hdr, i + 4)[0]
        if chunk == b"fmt " and i + 24 <= len(hdr):
            audio, ch = struct.unpack_from("<HH", hdr, i + 8)
            r = struct.unpack_from("<I", hdr, i + 12)[0]
            bps = struct.unpack_from("<H", hdr, i + 22)[0]
            if audio != 1 or ch != 1 or bps != 16:
                raise ValueError(f"expected mono 16-bit PCM, got audio={audio} ch={ch} bps={bps}")
            rate = int(r)
        elif chunk == b"data":
            data_off = i + 8
            data_bytes = sz
            break
        step = 8 + sz + (sz & 1)
        if step <= 0:
            raise ValueError("bad wav chunk")
        i += step
    if not rate or not data_off:
        raise ValueError("wav is missing fmt or data")
    return rate, data_off, data_bytes // 2


def read_samples(path: Path, data_off: int, n_samples: int, first: int, n: int) -> np.ndarray:
    path = Path(path)
    out = np.zeros(n, dtype=np.float64)
    if first >= n_samples or n <= 0:
        return out
    lo = max(first, 0)
    hi = min(first + n, n_samples)
    if hi <= lo:
        return out
    with path.open("rb") as handle:
        handle.seek(data_off + lo * 2)
        raw = np.frombuffer(handle.read((hi - lo) * 2), dtype="<i2")
    dest = lo - first
    out[dest : dest + raw.size] = raw.astype(np.float64) / 32768.0
    return out


def shortest_half_mean(row: np.ndarray) -> float:
    n = row.size
    keys = np.sort(row)
    even = n - (n % 2)
    half = even // 2
    span = keys[half : half + half] - keys[:half]
    i = int(np.argmin(span))
    low = keys[i]
    high = keys[i + half]
    chosen = row[(row >= low) & (row <= high)]
    if chosen.size == 0:
        return 0.0
    return float(chosen.mean())


def whiten_window(mag: np.ndarray) -> np.ndarray:
    """mag is (nrows, n_frames) non-negative. Returns smoothed whitened cells."""
    white = np.empty_like(mag)
    for r in range(mag.shape[0]):
        mu = shortest_half_mean(mag[r])
        if not mu > 0.0:
            mu = 1e-20
        white[r] = np.abs((mag[r] - mu) / mu)
    smooth = 4.0 * white
    smooth[:, 1:] += white[:, :-1]
    smooth[:, :-1] += white[:, 1:]
    smooth[1:, :] += white[:-1, :]
    smooth[:-1, :] += white[1:, :]
    smooth *= 0.125
    return smooth


def window_frames(samples: np.ndarray, layout: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed spectrogram (nrows, n_frames) and frame-center sample offsets."""
    fftl = layout["fftl"]
    hop = layout["hop"]
    n = samples.size
    if n < fftl:
        return np.zeros((layout["nrows"], 0)), np.zeros(0, dtype=np.int64)
    n_frames = (n - fftl) // hop
    if n_frames <= 0:
        return np.zeros((layout["nrows"], 0)), np.zeros(0, dtype=np.int64)
    hamming = layout["hamming"]
    frames = np.empty((n_frames, fftl), dtype=np.float64)
    for t in range(n_frames):
        a = t * hop
        frames[t] = samples[a : a + fftl] * hamming
    spec = np.fft.rfft(frames, axis=1)
    row0 = layout["row0"]
    nrows = layout["nrows"]
    mag = np.abs(spec[:, row0 : row0 + nrows]).T
    smooth = whiten_window(mag)
    centers = (np.arange(n_frames, dtype=np.int64) * hop) + fftl // 2
    return smooth, centers


def ratios_over(path: Path, t0: float, t1: float, dt: float = 0.1) -> dict:
    path = Path(path)
    """Energy ratio on [t0, t1) at resolution dt, using full 75 s GPL windows."""
    rate, data_off, n_samples = open_wav(path)
    layout = layout_for(rate)
    win = WINDOW_SEC * rate
    seg = int(round(dt * rate))
    if seg < 1:
        raise ValueError("dt is shorter than one sample")
    # Windows that can place a frame center inside [t0, t1).
    first_win = int(math.floor(t0 / WINDOW_SEC))
    last_win = int(math.floor((t1 - 1e-9) / WINDOW_SEC))
    n_bins = int(math.ceil((t1 - t0) / dt))
    band = np.zeros(n_bins, dtype=np.float64)
    overall = np.zeros(n_bins, dtype=np.float64)
    b0, b1 = layout["band0"], layout["band1"]
    for w in range(first_win, last_win + 1):
        start = w * win
        if start >= n_samples:
            break
        n = min(win, n_samples - start)
        samples = read_samples(path, data_off, n_samples, start, n)
        smooth, centers = window_frames(samples, layout)
        if centers.size == 0:
            continue
        abs_center = start + centers
        t_center = abs_center / rate
        keep = (t_center >= t0) & (t_center < t1)
        if not np.any(keep):
            continue
        energy = smooth[:, keep] ** 2
        overall_e = energy.sum(axis=0)
        band_e = energy[b0 : b1 + 1].sum(axis=0)
        bins = ((abs_center[keep] - int(round(t0 * rate))) // seg).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)
        np.add.at(band, bins, band_e)
        np.add.at(overall, bins, overall_e)
    ratio = np.divide(band, overall, out=np.zeros_like(band), where=overall > 0)
    times = t0 + np.arange(n_bins) * dt
    return {
        "rate": rate,
        "fftl": layout["fftl"],
        "hop": layout["hop"],
        "t": times,
        "band": band,
        "overall": overall,
        "ratio": ratio,
        "dt": dt,
    }


def interval_ratio(band: np.ndarray, overall: np.ndarray, i0: int, i1: int) -> float:
    b = band[i0:i1].sum()
    o = overall[i0:i1].sum()
    return float(b / o) if o > 0 else 0.0


def fft_band_fraction(ratio: np.ndarray, dt: float, f_lo: float = 0.1, f_hi: float = 1.0) -> dict:
    """Fraction of Parseval energy whose frequency lies in [f_lo, f_hi]."""
    x = np.asarray(ratio, dtype=np.float64)
    n = x.size
    spec = np.fft.rfft(x)
    power = np.abs(spec) ** 2
    weight = np.ones(power.size)
    if n % 2 == 0:
        weight[1:-1] = 2.0
    else:
        weight[1:] = 2.0
    power *= weight
    freqs = np.fft.rfftfreq(n, d=dt)
    mask = (freqs >= f_lo - 1e-9) & (freqs <= f_hi + 1e-9)
    total = float(power.sum())
    band = float(power[mask].sum())
    ac = total - float(power[0])
    return {
        "fraction_of_total": band / total if total else 0.0,
        "fraction_of_ac": band / ac if ac else 0.0,
        "freqs": freqs,
        "power": power,
        "f_lo_bin": float(freqs[mask][0]) if np.any(mask) else None,
        "f_hi_bin": float(freqs[mask][-1]) if np.any(mask) else None,
    }
