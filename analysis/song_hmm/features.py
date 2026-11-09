"""Per-second observations computed directly from the wav.

For each second: 16 log band levels (30-1000 Hz) above a rolling noise floor,
the broadband level above its floor, and the within-second modulation of the
broadband level. The noise floor is the 10th percentile of frame levels in a
10-minute window, computed inside each half on its own. Columns are z-scored
with the train half only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

NFFT = 512
HOP_S = 0.1
FRAMES_PER_S = 10
N_BANDS = 16
F_LO, F_HI = 30.0, 1000.0
FLOOR_WINDOW_S = 600
FLOOR_STEP_S = 60
FLOOR_PCT = 10.0
EPS = 1e-12
FEATURE_NAMES = tuple(f"band{i + 1}" for i in range(N_BANDS)) + ("level", "modulation")
N_FEATURES = len(FEATURE_NAMES)


def band_edges() -> np.ndarray:
    return np.geomspace(F_LO, F_HI, N_BANDS + 1)


def frame_levels(audio: np.ndarray, sample_rate: int, n_seconds: int, chunk_s: int = 1800) -> np.ndarray:
    """Log power per frame: 16 bands then broadband. Shape (n_seconds * 10, 17).

    Frame f starts at sample f * hop, so frames 10 s .. 10 s + 9 belong to second s.
    """
    hop = int(round(HOP_S * sample_rate))
    freqs = np.fft.rfftfreq(NFFT, 1.0 / sample_rate)
    edges = band_edges()
    band_of_bin = np.searchsorted(edges, freqs, side="right") - 1
    in_band = (band_of_bin >= 0) & (band_of_bin < N_BANDS) & (freqs <= F_HI)
    if np.bincount(band_of_bin[in_band], minlength=N_BANDS).min() == 0:
        raise ValueError("a band has no FFT bin; check the sample rate")
    weights = np.zeros((freqs.size, N_BANDS + 1))
    weights[np.flatnonzero(in_band), band_of_bin[in_band]] = 1.0
    weights[in_band, N_BANDS] = 1.0
    window = np.hanning(NFFT).astype(np.float32)

    n_frames = n_seconds * FRAMES_PER_S
    need = (n_frames - 1) * hop + NFFT
    audio = np.asarray(audio)
    if audio.size < need:
        audio = np.concatenate([audio, np.zeros(need - audio.size, dtype=audio.dtype)])
    out = np.empty((n_frames, N_BANDS + 1))
    step = chunk_s * FRAMES_PER_S
    for f0 in range(0, n_frames, step):
        f1 = min(n_frames, f0 + step)
        seg = audio[f0 * hop : (f1 - 1) * hop + NFFT].astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(seg, NFFT)[::hop][: f1 - f0]
        power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
        out[f0:f1] = np.log(power @ weights + EPS)
    return out


def noise_floor(levels: np.ndarray, n_seconds: int) -> np.ndarray:
    """10th percentile of frame levels in a 10-minute window, per second. Shape (n_seconds, C)."""
    centers = np.arange(FLOOR_STEP_S / 2, n_seconds, FLOOR_STEP_S)
    half = FLOOR_WINDOW_S / 2 * FRAMES_PER_S
    knots = np.empty((centers.size, levels.shape[1]))
    for i, c in enumerate(centers):
        lo = int(max(0, c * FRAMES_PER_S - half))
        hi = int(min(levels.shape[0], c * FRAMES_PER_S + half))
        knots[i] = np.percentile(levels[lo:hi], FLOOR_PCT, axis=0)
    seconds = np.arange(n_seconds) + 0.5
    return np.column_stack([np.interp(seconds, centers, knots[:, j]) for j in range(levels.shape[1])])


def second_features(levels: np.ndarray, n_seconds: int) -> np.ndarray:
    """Raw per-second features for one stretch of frames that starts on a second."""
    floor = noise_floor(levels, n_seconds)
    per_s = levels[: n_seconds * FRAMES_PER_S].reshape(n_seconds, FRAMES_PER_S, -1)
    above = per_s.mean(axis=1) - floor
    modulation = per_s[:, :, N_BANDS].std(axis=1)
    return np.column_stack([above, modulation])


def build_features(wav_path: Path, split_s: int) -> dict[str, np.ndarray]:
    """Raw and z-scored features for the whole recording. Stats come from [0, split_s)."""
    sample_rate, audio = wavfile.read(wav_path, mmap=True)
    if audio.ndim > 1:
        audio = audio[:, 0]
    n_seconds = int(audio.shape[0] // sample_rate)
    if not 0 < split_s < n_seconds:
        raise ValueError(f"split {split_s} s outside a {n_seconds} s recording")
    levels = frame_levels(audio, sample_rate, n_seconds)
    raw = np.vstack(
        [
            second_features(levels[: split_s * FRAMES_PER_S], split_s),
            second_features(levels[split_s * FRAMES_PER_S :], n_seconds - split_s),
        ]
    )
    mean, std = raw[:split_s].mean(axis=0), raw[:split_s].std(axis=0)
    std = np.maximum(std, 1e-6)
    return {
        "x": (raw - mean) / std,
        "raw": raw,
        "mean": mean,
        "std": std,
        "split_s": np.int64(split_s),
        "sample_rate": np.int64(sample_rate),
        "band_edges": band_edges(),
        "feature_names": np.array(FEATURE_NAMES),
    }


def load_or_build(wav_path: Path, split_s: int, cache: Path) -> dict[str, np.ndarray]:
    key = f"{Path(wav_path).resolve()}|{split_s}|{NFFT}|{HOP_S}|{N_BANDS}|{F_LO}|{F_HI}|{FLOOR_WINDOW_S}|{FLOOR_PCT}"
    if cache.exists():
        with np.load(cache) as data:
            if str(data["key"]) == key:
                return {k: data[k] for k in data.files if k != "key"}
    feats = build_features(wav_path, split_s)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, key=np.array(key), **feats)
    return feats
