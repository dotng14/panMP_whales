"""Shift and scale a whitened spectrogram column by column.

The ten-motif search does not use this. Specificity shifts and scales the
dot-product sequence instead, as described in ``spectrogram_time_normalize.md``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import fftconvolve

from analysis.dict_learning.code.core import Window


def time_normalize_windows(
    windows: list[Window],
    minutes: float = 1.0,
    *,
    divide_by_std: bool = True,
) -> list[Window]:
    """Return windows whose data is the time-normalized spectrogram.

    ``minutes`` is the standard deviation of the time-smoothing Gaussian.
    The default is one minute. ``divide_by_std`` scales each column by the
    standard deviation of its smoothed values. When it is false, the column
    is only shifted by the smoothed mean.
    """
    if minutes <= 0.0:
        raise ValueError(f"minutes must be positive, got {minutes}")
    ordered = sorted(windows, key=lambda window: (window.start_sample, window.window_id))
    if not ordered:
        return []
    original = np.concatenate([np.asarray(window.data, dtype=np.float64) for window in ordered], axis=1)
    sec_per_bin = float(ordered[0].overlap) / float(ordered[0].sample_freq)
    sigma = (float(minutes) * 60.0) / sec_per_bin
    radius = max(1, int(np.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    smoothed = fftconvolve(original, kernel[None, :], mode="same", axes=1)
    coverage = fftconvolve(np.ones(original.shape[1], dtype=np.float64), kernel, mode="same")
    smoothed = smoothed / coverage
    mean = smoothed.mean(axis=0)
    std = smoothed.std(axis=0)
    normalized = np.array(original, copy=True)
    normalized -= mean
    if divide_by_std:
        good = std > 0.0
        normalized[:, good] = normalized[:, good] / std[good]
        normalized[:, ~good] = original[:, ~good]
    pieces: list[Window] = []
    column = 0
    for window in ordered:
        width = int(np.asarray(window.data).shape[1])
        pieces.append(replace(window, data=normalized[:, column : column + width].copy()))
        column += width
    return pieces
