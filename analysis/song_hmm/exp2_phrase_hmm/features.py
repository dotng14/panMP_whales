"""Phrase-window features: wav -> model matrix. Every step is documented in pipeline.md.

I-0 raw audio a[n]           2 kHz samples
I-1 band spectrogram S[n,b]  ln power in 64 log-spaced bands (59 non-empty), 0.05 s frames
I-2 whitened S^[n,b]         max(S - rolling 5-min median, 0), per half
I-3 phrase window P_i        240 frames (12 s) of S^, one every 40 frames (2 s), inside one half
I-4 descriptor v_i           5 statistics x 59 bands = 295 numbers
I-5 standardized z_i         (v - mu_train) / sigma_train
I-6 model input x_i          top 20 principal components (PCA fit on train)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

NFFT = 512
HOP = 100  # samples; 0.05 s at 2 kHz
N_BANDS_EDGES = 64
F_LO, F_HI = 30.0, 1000.0
FLOOR_BLOCK_S = 60
FLOOR_BLOCKS = 5  # median over +-2 blocks
WINDOW_S = 12
STEP_S = 2
N_COMPONENTS = 20
STATS = ("level", "variability", "flux", "upsweep", "downsweep")
EPS = 1e-9


def band_matrix(sample_rate: int, nfft: int = NFFT) -> tuple[np.ndarray, np.ndarray]:
    """(n_bins, B) 0/1 matrix pooling FFT bins into non-empty log bands, and the kept band edges (B, 2)."""
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    edges = np.geomspace(F_LO, F_HI, N_BANDS_EDGES + 1)
    idx = np.searchsorted(edges, freqs, side="right") - 1
    ok = (idx >= 0) & (idx < N_BANDS_EDGES) & (freqs <= F_HI)
    m = np.zeros((freqs.size, N_BANDS_EDGES), np.float32)
    m[np.flatnonzero(ok), idx[ok]] = 1.0
    keep = m.sum(axis=0) > 0
    return m[:, keep], np.column_stack([edges[:-1], edges[1:]])[keep]


def band_spectrogram(
    audio: np.ndarray, sample_rate: int, n_frames: int, chunk: int = 36000, nfft: int = NFFT, hop: int = HOP
) -> np.ndarray:
    """I-1. S[n, b] = ln(sum over band b of |FFT(hann * a[hop n : hop n + nfft])|^2 + 1e-6)."""
    m, _ = band_matrix(sample_rate, nfft)
    window = np.hanning(nfft).astype(np.float32)
    out = np.empty((n_frames, m.shape[1]), np.float32)
    for f0 in range(0, n_frames, chunk):
        f1 = min(n_frames, f0 + chunk)
        need = (f1 - f0 - 1) * hop + nfft
        seg = np.asarray(audio[f0 * hop : f0 * hop + need], np.float32)
        if seg.size < need:
            seg = np.concatenate([seg, np.zeros(need - seg.size, np.float32)])
        frames = np.lib.stride_tricks.sliding_window_view(seg, nfft)[::hop][: f1 - f0]
        power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
        out[f0:f1] = np.log(power @ m + 1e-6)
    return out


def whiten(s: np.ndarray, fps: int, pct: float = 50.0, n_blocks_window: int = FLOOR_BLOCKS, clip: bool = True) -> np.ndarray:
    """I-2. Subtract a rolling per-band floor, then clip at 0.

    Default: 1-min block medians, then the median over 5 blocks (+-2 min). ``pct`` < 50
    uses that percentile for the block statistic instead (e.g. 10th percentile).
    """
    block = FLOOR_BLOCK_S * fps
    n_blocks = int(np.ceil(len(s) / block))
    if pct == 50.0:
        med = np.array([np.median(s[j * block : (j + 1) * block], axis=0) for j in range(n_blocks)])
    else:
        med = np.array([np.percentile(s[j * block : (j + 1) * block], pct, axis=0) for j in range(n_blocks)])
    half = n_blocks_window // 2
    out = np.empty_like(s)
    for j in range(n_blocks):
        floor = np.median(med[max(0, j - half) : j + half + 1], axis=0)
        d = s[j * block : (j + 1) * block] - floor
        out[j * block : (j + 1) * block] = np.maximum(d, 0.0) if clip else d
    return out


def _window_sums(series: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    c = np.concatenate([np.zeros((1, series.shape[1])), np.cumsum(series, axis=0, dtype=np.float64)])
    return c[starts + length] - c[starts]


def descriptors(sw: np.ndarray, fps: int, window_s: float = WINDOW_S, step_s: float = STEP_S) -> tuple[np.ndarray, np.ndarray]:
    """I-3 and I-4 for one half. Returns (window start frames, v) with v of shape (n_windows, 5 * B).

    Statistics per band b over the window's frames n (pairs n, n+1 for the last three):
      level       mean P[n,b]
      variability std  P[n,b]
      flux        mean |P[n+1,b] - P[n,b]|
      upsweep     sum P[n,b] P[n+1,b+1] / sum P[n,b] (P[n+1,b-1] + P[n+1,b] + P[n+1,b+1])
      downsweep   same with P[n+1,b-1] on top
    """
    w, step = int(window_s * fps), int(step_s * fps)
    starts = np.arange(0, len(sw) - w + 1, step)
    x = sw.astype(np.float64)
    nxt = x[1:]
    cur = x[:-1]
    up_n = np.zeros_like(cur)
    up_n[:, :-1] = cur[:, :-1] * nxt[:, 1:]  # energy at b continuing at b+1
    down_n = np.zeros_like(cur)
    down_n[:, 1:] = cur[:, 1:] * nxt[:, :-1]  # energy at b continuing at b-1
    same_n = cur * nxt

    mean = _window_sums(x, starts, w) / w
    sq = _window_sums(x * x, starts, w) / w
    std = np.sqrt(np.maximum(sq - mean**2, 0.0))
    flux = _window_sums(np.abs(nxt - cur), starts, w - 1) / (w - 1)
    up = _window_sums(up_n, starts, w - 1)
    down = _window_sums(down_n, starts, w - 1)
    same = _window_sums(same_n, starts, w - 1)
    total = up + down + same
    up_frac = np.where(total > EPS, up / np.maximum(total, EPS), 0.0)
    down_frac = np.where(total > EPS, down / np.maximum(total, EPS), 0.0)
    return starts, np.hstack([mean, std, flux, up_frac, down_frac])


def build(wav_path: Path, split_s: int) -> dict[str, np.ndarray]:
    sample_rate, audio = wavfile.read(wav_path, mmap=True)  # I-0
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sample_rate != HOP * 20:
        raise ValueError(f"expected 2 kHz audio (hop {HOP} = 0.05 s), got {sample_rate} Hz")
    fps = sample_rate // HOP
    n_seconds = int(audio.shape[0] // sample_rate)
    if not 0 < split_s < n_seconds:
        raise ValueError(f"split {split_s} s outside a {n_seconds} s recording")
    s = band_spectrogram(audio, sample_rate, n_seconds * fps)
    _, edges = band_matrix(sample_rate)

    halves = {}
    for name, f0, f1 in (("train", 0, split_s * fps), ("test", split_s * fps, n_seconds * fps)):
        sw = whiten(s[f0:f1], fps)
        starts, v = descriptors(sw, fps)
        halves[name] = (f0 // fps + starts / fps, v)

    v_tr = halves["train"][1]
    mu, sigma = v_tr.mean(axis=0), np.maximum(v_tr.std(axis=0), 1e-6)  # I-5
    z = {k: (v - mu) / sigma for k, (_, v) in halves.items()}
    _, sv, vt = np.linalg.svd(z["train"] - z["train"].mean(axis=0), full_matrices=False)  # I-6
    basis = vt[:N_COMPONENTS].T
    explained = (sv**2 / np.sum(sv**2))[:N_COMPONENTS]
    n_bands = edges.shape[0]
    out = {
        "split_s": np.int64(split_s),
        "n_seconds": np.int64(n_seconds),
        "fps": np.int64(fps),
        "band_edges": edges,
        "stats": np.array(STATS),
        "mu": mu,
        "sigma": sigma,
        "basis": basis,
        "explained": explained,
        "spectrogram_shape": np.array(s.shape),
    }
    for k, (t0, v) in halves.items():
        out[f"t_{k}"] = t0  # window start, seconds from recording start
        out[f"X_{k}"] = z[k] @ basis
        out[f"z_{k}"] = z[k].astype(np.float32)
        out[f"level_{k}"] = v[:, :n_bands].mean(axis=1)  # mean level above floor, for background seeding
    return out


def cache_key(wav_path: Path, split_s: int) -> str:
    return "|".join(
        map(str, (Path(wav_path).resolve(), split_s, NFFT, HOP, N_BANDS_EDGES, F_LO, F_HI, FLOOR_BLOCK_S, FLOOR_BLOCKS, WINDOW_S, STEP_S, N_COMPONENTS, "v1"))
    )


def load_or_build(wav_path: Path, split_s: int, cache: Path) -> dict[str, np.ndarray]:
    key = cache_key(wav_path, split_s)
    if cache.exists():
        with np.load(cache) as data:
            if str(data["key"]) == key:
                return {k: data[k] for k in data.files if k != "key"}
    feats = build(wav_path, split_s)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, key=np.array(key), **feats)
    return feats
