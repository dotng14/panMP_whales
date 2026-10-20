"""Pan matrix profile surface for a time series.

Port of panMP.m. Builds a length-normalized pan-MP surface over a range
of motif lengths and optionally plots it.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from mpxForPan import mpxForPan


def _zscore(x):
    """MATLAB-compatible z-score (sample std, ddof=1)."""
    x = np.asarray(x, dtype=float)
    std = np.std(x, ddof=1)
    if std == 0:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def panMP(
    TS,
    plot=True,
    MinMotifLen=32,
    MaxMotifLen=512,
    StepSize=64,
    MotifPlotMode=True,
):
    """Compute a pan matrix profile surface for time series TS.

    Parameters
    ----------
    TS : array-like
        Input time series (1-D).
    plot : bool, optional
        If True (default), display a jet colormap surface plot.
    MinMotifLen, MaxMotifLen, StepSize : int, optional
        Motif-length sweep (MATLAB defaults: 32, 512, 64).
    MotifPlotMode : bool, optional
        If True, flip the surface so motifs are peaks.

    Returns
    -------
    pan : ndarray
        Pan matrix profile surface (rows = time, cols = motif length).
    """
    TS = np.asarray(TS, dtype=float).ravel()

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

    n_rows = len(TS) - MinMotifLen + 1
    n_cols = MaxMotifLen + StepSize - 1
    pan = np.zeros((n_rows, n_cols))

    for i in range(MinMotifLen, MaxMotifLen + 1, StepSize):
        print(f"Motif length {i} of {MaxMotifLen}")
        # minlag must be >= 4 for mpxForPan; i-2 is fine for i >= 6
        matrixProfile, _matrixProfileIdx = mpxForPan(TS, max(i - 2, 4), i)

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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        rows, cols = pan.shape
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        surf = ax.plot_surface(
            X, Y, np.nan_to_num(pan, nan=0.0),
            cmap="jet",
            linewidth=0,
            antialiased=True,
            rcount=min(rows, 200),
            ccount=min(cols, 200),
        )
        fig.colorbar(surf, ax=ax, shrink=0.5)
        plt.show()

    return pan


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    rng = np.random.default_rng(0)
    ts = rng.standard_normal(2000)
    panMP(ts)
