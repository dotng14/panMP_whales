"""Matrix profile via cross-correlation (pan-matrix-profile helper).

Port of mpxForPan.m. Depends on accurate sum / squared-norm helpers
from Ogita et al. (Accurate Sum and Dot Product).
"""

from __future__ import annotations

import numpy as np


def mpxForPan(a, minlag, w):
    """Compute a matrix profile and profile index.

    Parameters
    ----------
    a : array-like
        Input time series (1-D).
    minlag : int
        Minimum lag (exclusion zone / spacing) between subsequences.
    w : int
        Subsequence / window length.

    Returns
    -------
    mp : ndarray
        Matrix profile (Euclidean distances).
    mpi : ndarray
        Matrix profile index (0-based indices into the time series).
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError("non vector input")

    if w < 4 or minlag < 4:
        raise ValueError("bad parameters")

    n = len(a)
    mu, sig = muinvn(a, w)

    # Differentials have 0 as their first entry. This simplifies index
    # calculations and avoids special "first line" handling.
    df = np.concatenate(
        [[0.0], 0.5 * (a[w:n] - a[0 : n - w])]
    )
    dg = np.concatenate(
        [
            [0.0],
            (a[w:n] - mu[1 : n - w + 1]) + (a[0 : n - w] - mu[0 : n - w]),
        ]
    )

    diagmax = n - w + 1
    mp = np.full(n - w + 1, -1.0)
    mpi = np.full(n - w + 1, np.nan)

    for diag in range(minlag, diagmax):
        c = np.sum((a[diag : diag + w] - mu[diag]) * (a[0:w] - mu[0]))
        for offset in range(0, n - w - diag + 1):
            c = c + df[offset] * dg[offset + diag] + df[offset + diag] * dg[offset]
            c_cmp = c * (sig[offset] * sig[offset + diag])
            if c_cmp > mp[offset]:
                mp[offset] = c_cmp
                mpi[offset] = offset + diag
            if c_cmp > mp[offset + diag]:
                mp[offset + diag] = c_cmp
                mpi[offset + diag] = offset

    mp = np.sqrt(2 * w * (1 - np.minimum(1.0, mp)))
    return mp, mpi


def muinvn(a, w):
    """Moving average and stable inverse centered norm."""
    mu = sum2s(a, w) / w
    sig = np.zeros(len(a) - w + 1)
    for i in range(len(mu)):
        sig[i] = sq2s(a[i : i + w] - mu[i])
    sig = 1.0 / np.sqrt(sig)
    return mu, sig


def sq2s(a):
    """Accurate sum of squares (Ogita et al.)."""
    a = np.asarray(a, dtype=float)
    h = a * a
    c = ((2**27) + 1) * a
    a1 = c - (c - a)
    a2 = a - a1
    a3 = a1 * a2
    r = a2 * a2 - (((h - a1 * a1) - a3) - a3)
    p = h[0]
    s = r[0]
    for i in range(1, len(a)):
        x = p + h[i]
        z = x - p
        s = s + (((p - (x - z)) + (h[i] - z)) + r[i])
        p = x
    return p + s


def sum2s(a, w):
    """Accurate sliding window sum (Ogita et al.)."""
    a = np.asarray(a, dtype=float)
    res = np.zeros(len(a) - w + 1)
    p = a[0]
    s = 0.0
    for i in range(1, w):
        x = p + a[i]
        z = x - p
        s = s + ((p - (x - z)) + (a[i] - z))
        p = x
    res[0] = p + s
    for i in range(w, len(a)):
        x = p - a[i - w]
        z = x - p
        s = s + ((p - (x - z)) - (a[i - w] + z))
        p = x

        x = p + a[i]
        z = x - p
        s = s + ((p - (x - z)) + (a[i] - z))
        p = x

        res[i - w + 1] = p + s
    return res
