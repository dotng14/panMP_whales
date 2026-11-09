"""Per-channel energy normalisation (Wang et al. 2017; Lostanlen et al. 2019).

    M[t] = (1 - b) M[t-1] + b E[t]                    smoothed energy per band
    PCEN = (E / (eps + M)^alpha + delta)^r - delta^r

M follows the slowly changing background of each band, so dividing by it acts
as an automatic gain control: steady noise is pushed down, and a sound that
rises above its recent background stands out, without clipping anything.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

T_SMOOTH_S = 2.0  # background time constant: longer than a unit (~1 s), shorter than a theme
ALPHA = 0.98
DELTA = 2.0
R = 0.5
EPS = 1e-6


def smoothing_coef(fps: float, t: float = T_SMOOTH_S) -> float:
    return (np.sqrt(1 + 4 * (t * fps) ** 2) - 1) / (2 * (t * fps) ** 2)


def pcen(energy: np.ndarray, fps: float, t: float = T_SMOOTH_S) -> np.ndarray:
    """``energy``: linear band energy (frames, bands). Returns PCEN values of the same shape."""
    b = smoothing_coef(fps, t)
    zi = (1 - b) * energy[:1]  # start the smoother at the first frame's level
    m, _ = lfilter([b], [1, b - 1], energy, axis=0, zi=zi)
    return (energy / (EPS + m) ** ALPHA + DELTA) ** R - DELTA**R
