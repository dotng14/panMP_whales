"""Numba matrix profile with the same updates as mpxForPan.

The pure-Python port is the reference. This kernel runs the same diagonal
recurrence so a 10-hour, 10 Hz feature series is practical. Thread-local
profiles are reduced with a strict ``>`` so an earlier diagonal wins ties,
matching the MATLAB loop.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from mpxForPan import mpxForPan


@njit(cache=True)
def _sliding_mean_invnorm(a, w):
    n = len(a)
    nprof = n - w + 1
    mu = np.empty(nprof)
    sig = np.empty(nprof)
    acc = 0.0
    acc2 = 0.0
    for k in range(w):
        acc += a[k]
        acc2 += a[k] * a[k]
    mu[0] = acc / w
    var = acc2 - acc * acc / w
    if var < 0.0:
        var = 0.0
    sig[0] = 1.0 / np.sqrt(var)
    for i in range(w, n):
        x_out = a[i - w]
        x_in = a[i]
        acc += x_in - x_out
        acc2 += x_in * x_in - x_out * x_out
        j = i - w + 1
        mu[j] = acc / w
        var = acc2 - acc * acc / w
        if var < 0.0:
            var = 0.0
        sig[j] = 1.0 / np.sqrt(var)
    return mu, sig


@njit(parallel=True, cache=True)
def _mpx_parallel(a, mu, sig, df, dg, minlag, w, nthreads):
    """One contiguous band of diagonals per worker, so workers never share a row."""
    n = len(a)
    nprof = n - w + 1
    local_mp = np.empty((nthreads, nprof))
    local_mpi = np.empty((nthreads, nprof))
    for t in range(nthreads):
        for i in range(nprof):
            local_mp[t, i] = -1.0
            local_mpi[t, i] = -1.0

    diagmax = nprof
    span = diagmax - minlag
    for t in prange(nthreads):
        d0 = minlag + span * t // nthreads
        d1 = minlag + span * (t + 1) // nthreads
        for diag in range(d0, d1):
            c = 0.0
            mudiag = mu[diag]
            mu0 = mu[0]
            for k in range(w):
                c += (a[diag + k] - mudiag) * (a[k] - mu0)
            last = n - w - diag + 1
            for offset in range(last):
                c += df[offset] * dg[offset + diag] + df[offset + diag] * dg[offset]
                c_cmp = c * (sig[offset] * sig[offset + diag])
                if c_cmp > local_mp[t, offset]:
                    local_mp[t, offset] = c_cmp
                    local_mpi[t, offset] = offset + diag
                j = offset + diag
                if c_cmp > local_mp[t, j]:
                    local_mp[t, j] = c_cmp
                    local_mpi[t, j] = offset

    mp = np.empty(nprof)
    mpi = np.empty(nprof)
    for i in range(nprof):
        best = -1.0
        best_j = -1.0
        best_sep = 1.0e300
        for t in range(nthreads):
            val = local_mp[t, i]
            if val < 0.0:
                continue
            sep = local_mpi[t, i] - i
            if sep < 0.0:
                sep = -sep
            if val > best or (val == best and sep < best_sep):
                best = val
                best_j = local_mpi[t, i]
                best_sep = sep
        mp[i] = best
        mpi[i] = best_j
    for i in range(nprof):
        v = mp[i]
        if v > 1.0:
            v = 1.0
        if v < -1.0:
            mp[i] = np.nan
            mpi[i] = np.nan
        else:
            mp[i] = np.sqrt(2.0 * w * (1.0 - v))
    return mp, mpi


def mpx_fast(a, minlag, w, nthreads=1):
    """Same contract as ``mpxForPan``: distance profile and 0-based indices."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("non vector input")
    if w < 4 or minlag < 4:
        raise ValueError("bad parameters")
    n = len(a)
    mu, sig = _sliding_mean_invnorm(a, w)
    df = np.empty(n - w + 1, dtype=np.float64)
    dg = np.empty(n - w + 1, dtype=np.float64)
    df[0] = 0.0
    dg[0] = 0.0
    if n - w > 0:
        df[1:] = 0.5 * (a[w:n] - a[0 : n - w])
        dg[1:] = (a[w:n] - mu[1 : n - w + 1]) + (a[0 : n - w] - mu[0 : n - w])
    mp, mpi = _mpx_parallel(a, mu, sig, df, dg, int(minlag), int(w), int(nthreads))
    return mp, mpi


def assert_matches_reference(n=800, w=32, minlag=32, seed=0, atol=1e-8):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    ref_mp, ref_mpi = mpxForPan(a, minlag, w)
    mp, mpi = mpx_fast(a, minlag, w)
    if not np.allclose(mp, ref_mp, rtol=0.0, atol=atol, equal_nan=True):
        err = np.nanmax(np.abs(mp - ref_mp))
        raise AssertionError(f"distance profile mismatch, max abs {err}")
    # Indices can differ only where the two profiles tie. Compare where unique.
    disagree = mpi != ref_mpi
    if np.any(disagree):
        # A tie is acceptable when both neighbors have the same distance.
        raise AssertionError(f"index mismatch at {int(np.sum(disagree))} positions")
    return True
