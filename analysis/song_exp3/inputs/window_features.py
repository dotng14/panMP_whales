"""Window inputs. One window = one phrase-length stretch, every 2 s.

X1 window statistics: per band, level / variability / flux / up-sweep / down-sweep
   (the Experiment 2 descriptor, on any S x N spectrogram).
X2 2D-Fourier patch: the window is averaged to 0.1 s columns, and the magnitude of its
   2D Fourier transform is kept for slow time modulations (0-3 Hz) and the lowest
   frequency modulations (both signs, so rising and falling patterns differ). The
   magnitude does not change when the phrase shifts in time inside the window, but
   keeps the rhythm and the pattern of units. The mean level per band is appended so
   absolute pitch is kept too.
Both are z-scored and reduced with PCA, all statistics from the train half.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.song_hmm_phrase import features as F

STEP_S = 2.0
X2_COL_S = 0.1  # time resolution of the patch before the 2D FFT
X2_MAX_HZ = 3.0  # keep time modulations up to this rate
X2_FREQ_MODES = 15  # keep frequency modulations -15..15 (cycles across the band axis)


@dataclass
class WindowInput:
    starts_s: np.ndarray  # window start times, seconds from the start of the half
    raw: np.ndarray  # (windows, dims) before standardisation


def x1(sw: np.ndarray, fps: int, window_s: float) -> WindowInput:
    starts, v = F.descriptors(sw, fps, window_s=window_s, step_s=STEP_S)
    return WindowInput(starts / fps, v)


def x2(sw: np.ndarray, fps: int, window_s: float, chunk: int = 512) -> WindowInput:
    col = int(round(X2_COL_S * fps))
    n_col = len(sw) // col
    cols = sw[: n_col * col].reshape(n_col, col, -1).mean(1)  # (columns, bands)
    w = int(round(window_s / X2_COL_S))
    step = int(round(STEP_S / X2_COL_S))
    starts = np.arange(0, n_col - w + 1, step)
    n_t = int(X2_MAX_HZ * window_s) + 1  # rfft bins 0..X2_MAX_HZ
    keep_f = np.r_[0 : X2_FREQ_MODES + 1, -X2_FREQ_MODES:0]
    out = []
    view = np.lib.stride_tricks.sliding_window_view(cols, w, axis=0)  # (n_col-w+1, bands, w)
    for c0 in range(0, len(starts), chunk):
        patch = view[starts[c0 : c0 + chunk]].astype(np.float64)  # (n, bands, w)
        patch = patch - patch.mean(axis=(1, 2), keepdims=True)
        spec = np.abs(np.fft.fft(np.fft.rfft(patch, axis=2)[:, :, :n_t], axis=1))  # (n, bands-modes, n_t)
        spec = spec[:, keep_f, :]
        prof = view[starts[c0 : c0 + chunk]].mean(axis=2)
        out.append(np.hstack([np.log1p(spec.reshape(len(patch), -1)), prof]))
    return WindowInput(starts * X2_COL_S, np.vstack(out))


@dataclass
class Reduced:
    train: np.ndarray
    test: np.ndarray
    explained: np.ndarray  # cumulative explained variance


def standardise_pca(tr: np.ndarray, te: np.ndarray, n_max: int = 40) -> Reduced:
    mu, sd = tr.mean(0), np.maximum(tr.std(0), 1e-6)
    ztr, zte = (tr - mu) / sd, (te - mu) / sd
    _, sv, vt = np.linalg.svd(ztr - ztr.mean(0), full_matrices=False)
    basis = vt[:n_max].T
    return Reduced(ztr @ basis, zte @ basis, np.cumsum(sv**2) / np.sum(sv**2))
