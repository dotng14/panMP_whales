"""Matrix profile and the non-overlapping motif pairs it finds.

One series, or one row per frequency. Distance on a row is z-normalized
Euclidean distance. Across rows those distances add.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        raise ValueError("non vector input")  # no-pipeline-cover: length_profile passes one row

    if w < 4 or minlag < 4:
        raise ValueError("bad parameters")  # no-pipeline-cover: pan_lengths keeps lengths >= 4

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

import numpy as np
from numba import njit, prange


@njit(cache=True)  # no-pipeline-cover: numba-compiled, coverage cannot trace it; unit tests compare it with mpxForPan
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
    if var <= 0.0:
        sig[0] = 0.0
    else:
        sig[0] = 1.0 / np.sqrt(var)
    for i in range(w, n):
        x_out = a[i - w]
        x_in = a[i]
        acc += x_in - x_out
        acc2 += x_in * x_in - x_out * x_out
        j = i - w + 1
        mu[j] = acc / w
        var = acc2 - acc * acc / w
        if var <= 0.0:
            sig[j] = 0.0
        else:
            sig[j] = 1.0 / np.sqrt(var)
    return mu, sig


@njit(parallel=True, cache=True)  # no-pipeline-cover: numba-compiled, coverage cannot trace it; unit tests compare it with mpxForPan
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
        raise ValueError("non vector input")  # no-pipeline-cover: length_profile passes one row
    if w < 4 or minlag < 4:
        raise ValueError("bad parameters")  # no-pipeline-cover: pan_lengths keeps lengths >= 4
    n = len(a)
    mu, sig = _sliding_mean_invnorm(a, w)
    df = np.empty(n - w + 1, dtype=np.float64)
    dg = np.empty(n - w + 1, dtype=np.float64)
    df[0] = 0.0
    dg[0] = 0.0
    if n - w > 0:  # no-pipeline-branch: pan_profile rejects signals shorter than a pair of windows
        df[1:] = 0.5 * (a[w:n] - a[0 : n - w])
        dg[1:] = (a[w:n] - mu[1 : n - w + 1]) + (a[0 : n - w] - mu[0 : n - w])
    mp, mpi = _mpx_parallel(a, mu, sig, df, dg, int(minlag), int(w), int(nthreads))
    return mp, mpi


def _channel_stats(series: np.ndarray, window: int):
    """Per-band mean, inverse norm, and the diagonal recurrence terms.

    A flat window has inverse-norm 0. The distance kernel treats two flat
    windows as a perfect match and a flat window against a varying one as
    correlation 0.
    """
    n_bands, n = series.shape
    nprof = n - window + 1
    mu = np.empty((n_bands, nprof), dtype=np.float64)
    sig = np.empty((n_bands, nprof), dtype=np.float64)
    df = np.zeros((n_bands, nprof), dtype=np.float64)
    dg = np.zeros((n_bands, nprof), dtype=np.float64)
    for band in range(n_bands):
        mu[band], sig[band] = _sliding_mean_invnorm(series[band], window)
        if n > window:  # no-pipeline-branch: pan_profile rejects signals shorter than a pair of windows
            df[band, 1:] = 0.5 * (series[band, window:n] - series[band, 0 : n - window])
            dg[band, 1:] = (series[band, window:n] - mu[band, 1:nprof]) + (
                series[band, 0 : n - window] - mu[band, 0 : nprof - 1]
            )
    sig[~np.isfinite(sig)] = 0.0
    return mu, sig, df, dg


@njit(parallel=True, cache=True)  # no-pipeline-cover: numba-compiled, coverage cannot trace it; unit tests compare it with mpxForPan
def _mpx_channels_parallel(series, mu, sig, df, dg, minlag, w, nthreads):
    """Nearest neighbor under the sum of per-band z-normalized distances."""
    n_bands = series.shape[0]
    n = series.shape[1]
    nprof = n - w + 1
    local_mp = np.empty((nthreads, nprof))
    local_mpi = np.empty((nthreads, nprof))
    for t in range(nthreads):
        for i in range(nprof):
            local_mp[t, i] = 1.0e300
            local_mpi[t, i] = -1.0

    diagmax = nprof
    span = diagmax - minlag
    for t in prange(nthreads):
        dot = np.zeros(n_bands)
        d0 = minlag + span * t // nthreads
        d1 = minlag + span * (t + 1) // nthreads
        for diag in range(d0, d1):
            for b in range(n_bands):
                acc = 0.0
                mudiag = mu[b, diag]
                mu0 = mu[b, 0]
                for k in range(w):
                    acc += (series[b, diag + k] - mudiag) * (series[b, k] - mu0)
                dot[b] = acc
            last = n - w - diag + 1
            for offset in range(last):
                total = 0.0
                valid = True
                for b in range(n_bands):
                    dot[b] += (
                        df[b, offset] * dg[b, offset + diag]
                        + df[b, offset + diag] * dg[b, offset]
                    )
                    left = sig[b, offset]
                    right = sig[b, offset + diag]
                    if left == 0.0 and right == 0.0:
                        corr = 1.0
                    elif left == 0.0 or right == 0.0:
                        corr = 0.0
                    else:
                        corr = dot[b] * (left * right)
                    if corr > 1.0:
                        corr = 1.0
                    if corr < -1.0:
                        valid = False
                    else:
                        total += np.sqrt(2.0 * w * (1.0 - corr))
                if valid:
                    if total < local_mp[t, offset]:
                        local_mp[t, offset] = total
                        local_mpi[t, offset] = offset + diag
                    j = offset + diag
                    if total < local_mp[t, j]:
                        local_mp[t, j] = total
                        local_mpi[t, j] = offset

    mp = np.empty(nprof)
    mpi = np.empty(nprof)
    for i in range(nprof):
        best = 1.0e300
        best_j = -1.0
        best_sep = 1.0e300
        for t in range(nthreads):
            val = local_mp[t, i]
            if val >= 1.0e299:
                continue
            sep = local_mpi[t, i] - i
            if sep < 0.0:
                sep = -sep
            if val < best or (val == best and sep < best_sep):
                best = val
                best_j = local_mpi[t, i]
                best_sep = sep
        if best >= 1.0e299:
            mp[i] = np.nan
            mpi[i] = np.nan
        else:
            mp[i] = best
            mpi[i] = best_j
    return mp, mpi


def mpx_channels(series, minlag, w, nthreads=1):
    """Matrix profile of a ``(bands, time)`` series.

    The distance is the sum, over bands, of the z-normalized Euclidean
    distance used by ``mpx_fast``. The reported neighbor is the one with the
    smallest sum, at least ``minlag`` samples away.
    """
    series = np.ascontiguousarray(series, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError("expected shape (bands, time)")  # no-pipeline-cover: length_profile passes (rows, T)
    if w < 4 or minlag < 4:
        raise ValueError("bad parameters")  # no-pipeline-cover: pan_lengths keeps lengths >= 4
    if series.shape[1] < w + minlag:
        raise ValueError("series is shorter than one pair of windows")  # no-pipeline-cover: pan_profile checks the length first
    mu, sig, df, dg = _channel_stats(series, int(w))
    return _mpx_channels_parallel(
        series, mu, sig, df, dg, int(minlag), int(w), int(nthreads)
    )


def _zscore(x):
    """MATLAB-compatible z-score (sample std, ddof=1)."""
    x = np.asarray(x, dtype=float)
    std = np.std(x, ddof=1)
    if std == 0:
        return np.zeros_like(x)  # no-pipeline-cover: a constant distance profile
    return (x - np.mean(x)) / std


ENGINES = ("fast", "reference")


def pan_minlag(length: int) -> int:
    """Exclusion zone used by panMP: ``length - 2``, at least 4."""
    return max(int(length) - 2, 4)


def length_profile(signal, length: int, *, engine: str = "fast", nthreads: int = 1):
    """Matrix profile ``(distance, neighbor)`` of a ``(rows, T)`` signal at one length.

    One row uses ``mpx_fast`` (``engine="fast"``) or the MATLAB-parity
    ``mpxForPan`` (``engine="reference"``, slow). Several rows use
    ``mpx_channels``, the sum of per-row z-normalized distances; the reference
    engine is one row only. The exclusion zone is :func:`pan_minlag`.
    """
    rows = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")  # no-pipeline-cover: argparse limits --engine
    minlag = pan_minlag(length)
    if rows.shape[0] > 1:
        if engine == "reference":
            raise ValueError("the reference engine takes a single row")
        return mpx_channels(rows, minlag, int(length), nthreads=nthreads)
    if engine == "reference":
        return mpxForPan(rows[0], minlag, int(length))
    return mpx_fast(rows[0], minlag, int(length), nthreads=nthreads)


def pan_lengths(min_len: int, max_len: int, step: int) -> list[int]:
    """Motif lengths ``min_len, min_len + step, …`` up to ``max_len``."""
    if min_len < 4:
        raise ValueError("MinMotifLen must be >= 4")
    if max_len < min_len:
        raise ValueError("MaxMotifLen must be >= MinMotifLen")
    if step < 1:
        raise ValueError("StepSize must be >= 1")
    return list(range(int(min_len), int(max_len) + 1, int(step)))


def panMP(  # no-pipeline-cover: MATLAB-shaped surface for notebooks and the parity tests; the CLI uses pan_profile
    TS,
    plot=True,
    MinMotifLen=32,
    MaxMotifLen=512,
    StepSize=64,
    MotifPlotMode=True,
    *,
    engine="reference",
    nthreads=1,
    plot_path=None,
    verbose=True,
):
    """Compute a pan matrix profile surface for time series TS.

    Parameters
    ----------
    TS : array-like
        Input time series (1-D, or a row or column vector).
    plot : bool, optional
        If True (default), draw a jet colormap surface plot.
    MinMotifLen, MaxMotifLen, StepSize : int, optional
        Motif-length sweep (MATLAB defaults: 32, 512, 64).
    MotifPlotMode : bool, optional
        If True, flip the surface so motifs are peaks.
    engine : {"reference", "fast"}, optional
        ``reference`` is the MATLAB port ``mpxForPan``; ``fast`` is the numba
        ``mpx_fast`` (same distances to ~1e-8).
    nthreads : int, optional
        Threads for the fast engine.
    plot_path : path, optional
        Save the plot here instead of showing it in a window.
    verbose : bool, optional
        Print each motif length as it runs.

    Returns
    -------
    pan : ndarray
        Pan matrix profile surface (rows = time, cols = motif length).
    """
    TS = np.asarray(TS, dtype=float)
    if TS.ndim > 2 or (TS.ndim == 2 and min(TS.shape) > 1):
        raise ValueError(f"panMP takes one series, got shape {TS.shape}; use pan_profile")
    TS = TS.ravel()

    if MinMotifLen < 4:
        raise ValueError("MinMotifLen must be >= 4")
    if MaxMotifLen < MinMotifLen:
        raise ValueError("MaxMotifLen must be >= MinMotifLen")
    if StepSize < 1:
        raise ValueError("StepSize must be >= 1")
    if len(TS) < MaxMotifLen + MinMotifLen:
        raise ValueError(
            f"TS length {len(TS)} too short for MaxMotifLen={MaxMotifLen}, "
            f"MinMotifLen={MinMotifLen}"
        )

    # MATLAB grows pan on assignment, so its width ends at the last motif
    # length the loop actually visits, not at MaxMotifLen.
    last_len = MinMotifLen + ((MaxMotifLen - MinMotifLen) // StepSize) * StepSize
    n_rows = len(TS) - MinMotifLen + 1
    n_cols = last_len + StepSize - 1
    pan = np.zeros((n_rows, n_cols))

    for i in range(MinMotifLen, MaxMotifLen + 1, StepSize):
        if verbose:
            print(f"Motif length {i} of {MaxMotifLen}")
        matrixProfile, _matrixProfileIdx = length_profile(
            TS, i, engine=engine, nthreads=nthreads
        )

        # Correct for length (sqrt(i)) and normalize
        LengthNormalizedMP = _zscore(matrixProfile / np.sqrt(i))

        # If StepSize > 1, duplicate the MP StepSize times along columns
        LengthNormalizedMP = np.tile(
            LengthNormalizedMP.reshape(-1, 1), (1, StepSize)
        )

        # MATLAB columns i:i+StepSize-1 -> Python i-1:i-1+StepSize
        pan[0 : len(matrixProfile), i - 1 : i - 1 + StepSize] = LengthNormalizedMP

    if MotifPlotMode:
        pan = pan - np.nanmax(pan)
        pan = pan * -1

    # Trim spurious data from end of surface plot
    # MATLAB: pan(end-MaxMotifLen:end,:) = []  removes MaxMotifLen+1 rows
    pan = pan[: len(pan) - MaxMotifLen - 1, :]

    pan = np.flipud(pan)
    pan[:, 0 : MinMotifLen - 1] = np.nan  # invisible / unused short lengths

    pan = pan / np.nanmax(pan)

    if plot:
        import matplotlib

        if plot_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        rows, cols = pan.shape
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        surf = ax.plot_surface(
            X, Y, np.ma.masked_invalid(pan),
            cmap="jet",
            linewidth=0,
            antialiased=True,
            rcount=min(rows, 200),
            ccount=min(cols, 200),
        )
        fig.colorbar(surf, ax=ax, shrink=0.5)
        if plot_path is not None:
            fig.savefig(plot_path, dpi=100)
            plt.close(fig)
        else:  # no-pipeline-cover: interactive window for notebooks
            plt.show()

    return pan


@dataclass
class PanProfile:
    """Matrix profiles of one signal over a sweep of motif lengths.

    ``surface`` is ``(n_lengths, T)``: per length, the distance divided by
    ``sqrt(length)``, z-scored, then flipped and scaled so the strongest motif
    in the sweep is 1 and the weakest 0 (panMP's ``MotifPlotMode``). Positions
    past the last window of a length are NaN.
    """

    lengths: list[int]
    distances: list[np.ndarray]
    neighbors: list[np.ndarray]
    surface: np.ndarray


def pan_profile(signal, lengths, *, engine: str = "fast", nthreads: int = 1) -> PanProfile:
    """Pan matrix profile of a 1-D or ``(rows, T)`` signal."""
    rows = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    if rows.ndim != 2:
        raise ValueError(f"signal must be 1-D or (rows, T), got shape {rows.shape}")  # no-pipeline-cover: load_series rejects other shapes first
    lengths = [int(length) for length in lengths]
    if not lengths:  # no-pipeline-cover: pan_lengths returns at least one length
        raise ValueError("need at least one motif length")
    n = rows.shape[1]
    if n < max(lengths) + pan_minlag(max(lengths)):
        raise ValueError(
            f"signal has {n} samples, too short for motif length {max(lengths)}"
        )
    distances = []
    neighbors = []
    surface = np.full((len(lengths), n), np.nan)
    for k, length in enumerate(lengths):
        distance, neighbor = length_profile(rows, length, engine=engine, nthreads=nthreads)
        distances.append(np.asarray(distance, dtype=np.float64))
        neighbors.append(np.asarray(neighbor, dtype=np.float64))
        surface[k, : len(distance)] = _zscore(np.asarray(distance) / np.sqrt(length))
    surface = np.nanmax(surface) - surface
    peak = np.nanmax(surface)
    if peak > 0:  # no-pipeline-branch: a perfectly flat surface
        surface = surface / peak
    return PanProfile(lengths=lengths, distances=distances, neighbors=neighbors, surface=surface)


def pan_top_pairs(profile: PanProfile, k: int) -> list[dict]:
    """Most unusual pairs across lengths (``top_pairs_by_zscore``).

    Each new pair starts outside every copy already chosen, at every length.
    Its nearest neighbour is not checked, so it can repeat an earlier copy.
    """
    scored = {
        length: (-distance, np.nan_to_num(neighbor, nan=-1.0), length)
        for length, distance, neighbor in zip(
            profile.lengths, profile.distances, profile.neighbors, strict=True
        )
    }
    return [
        {
            "length": int(length),
            "start_a": int(i),
            "start_b": int(j),
            "distance": float(-score),
            "zscore": float(unusual),
        }
        for length, i, j, score, unusual in top_pairs_by_zscore(scored, k)
    ]


def rms_envelope(samples: np.ndarray, sample_rate: int, envelope_rate: int) -> np.ndarray:
    """RMS of non-overlapping frames. One sample of the result is one frame."""
    if envelope_rate < 1 or sample_rate % envelope_rate != 0:
        raise ValueError("envelope_rate must divide sample_rate")
    audio = np.asarray(samples)
    if audio.ndim == 2 and 1 in audio.shape:
        audio = audio.reshape(-1)  # no-pipeline-cover: WAV readers return 1-D mono
    if audio.ndim != 1:
        raise ValueError("expected a mono recording")
    frame = sample_rate // envelope_rate
    usable = (len(audio) // frame) * frame
    blocks = audio[:usable].astype(np.float64, copy=False).reshape(-1, frame)
    return np.sqrt(np.mean(blocks * blocks, axis=1))


def distance_to_correlation(distance: np.ndarray, window: int) -> np.ndarray:  # no-pipeline-cover: used by learn_from_motifs and the notebooks, not the pan CLI
    """Invert the matrix-profile distance ``sqrt(2 * w * (1 - corr))``."""
    return 1.0 - np.asarray(distance, dtype=np.float64) ** 2 / (2.0 * window)


def motif_profile(series: np.ndarray, window: int, nthreads: int = 1):  # no-pipeline-cover: used by learn_from_motifs and the notebooks, not the pan CLI
    """Correlation profile and neighbor index. The two starts are at least ``window`` apart."""

    distance, neighbor = mpx_fast(series, window, window, nthreads=nthreads)
    return distance_to_correlation(distance, window), np.asarray(neighbor)


def motif_rectangle(bands: np.ndarray, start: int, width: int) -> np.ndarray:  # no-pipeline-cover: used by learn_from_motifs and the notebooks, not the pan CLI
    """One motif as a ``(bands, width)`` rectangle."""
    start = int(start)
    width = int(width)
    if start < 0 or start + width > bands.shape[1]:
        raise ValueError("motif rectangle does not fit in the profile")
    return np.array(bands[:, start : start + width], dtype=np.float64, copy=True)


def band_motif_profile(bands: np.ndarray, window: int, nthreads: int = 1):  # no-pipeline-cover: used by learn_from_motifs and the notebooks, not the pan CLI
    """Summed band-distance profile. The two starts are at least ``window`` apart."""

    distance, neighbor = mpx_channels(bands, window, window, nthreads=nthreads)
    return distance, np.asarray(neighbor)


def closest_pairs(distance: np.ndarray, neighbor: np.ndarray, window: int, k: int):  # no-pipeline-cover: used by learn_from_motifs and the notebooks, not the pan CLI
    """Smallest-distance pairs whose copies do not overlap a pair already chosen.

    Each item is ``(start_a, start_b, distance)``.
    """
    chosen = top_pairs(-np.asarray(distance, dtype=np.float64), neighbor, window, k)
    return [(i, j, -score) for i, j, score in chosen]


def order_pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i <= j else (j, i)


def top_pairs(correlation: np.ndarray, neighbor: np.ndarray, window: int, k: int):  # no-pipeline-cover: used by learn_from_motifs and the notebooks, not the pan CLI
    """Highest-correlation pairs whose copies do not overlap a pair already chosen."""
    remaining = np.array(correlation, dtype=np.float64, copy=True)
    chosen = []
    intervals = []
    n = len(remaining)

    def overlaps(start: int) -> bool:
        a0, a1 = start, start + window
        return any(a0 < b1 and b0 < a1 for b0, b1 in intervals)

    while len(chosen) < k:
        i = int(np.nanargmax(remaining))
        if not np.isfinite(remaining[i]):
            break
        j = int(neighbor[i])
        if overlaps(i) or overlaps(j):
            remaining[i] = -np.inf
            continue
        score = float(remaining[i])
        i, j = order_pair(i, j)
        chosen.append((i, j, score))
        intervals.append((i, i + window))
        intervals.append((j, j + window))
        for start in (i, j):
            lo = max(0, start - window + 1)
            hi = min(n, start + window)
            remaining[lo:hi] = -np.inf
    return chosen


def top_pairs_by_zscore(profiles: dict, k: int):
    """Most unusual pairs for their length.

    ``profiles`` maps a length key to ``(correlation, neighbor, window)``.
    A chosen window blocks overlaps at every length. Ranking uses the
    within-length z-score of correlation, so a short window is not preferred
    just because short windows correlate more easily.
    """
    scored = {}
    for length, (corr, neighbor, window) in profiles.items():
        finite = np.isfinite(corr)
        center = float(np.mean(corr[finite]))
        scale = float(np.std(corr[finite], ddof=1))
        unusual = (corr - center) / scale
        scored[length] = (unusual, neighbor, window, corr)
    blocked = {
        length: np.zeros(len(corr), dtype=bool)
        for length, (_, _, _, corr) in scored.items()
    }
    chosen = []
    for _ in range(k):
        best = None
        for length, (unusual, neighbor, window, corr) in scored.items():
            score = np.array(unusual, copy=True)
            score[blocked[length]] = -np.inf
            i = int(np.nanargmax(score))
            if not np.isfinite(score[i]):
                continue
            if best is None or score[i] > best[0]:
                best = (
                    float(score[i]),
                    length,
                    i,
                    int(neighbor[i]),
                    window,
                    float(corr[i]),
                )
        if best is None:
            break
        unusual, length, i, j, window, corr = best
        i, j = order_pair(i, j)
        chosen.append((length, i, j, corr, unusual))
        intervals = ((i, i + window), (j, j + window))
        for length2, (_, _, window2, _) in scored.items():
            mask = blocked[length2]
            for start, end in intervals:
                lo = max(0, start - window2 + 1)
                hi = min(len(mask), end)
                mask[lo:hi] = True
    return chosen


def format_clock(seconds: float) -> str:  # no-pipeline-cover: used by learn_from_motifs and the notebooks, not the pan CLI
    """``m:ss.s`` from seconds on the recording clock."""
    minutes, secs = divmod(float(seconds), 60.0)
    return f"{int(minutes)}:{secs:04.1f}"
