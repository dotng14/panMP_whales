"""X3 frame vectors: the sound at 0.25 s resolution, for M4.

For each 0.25 s frame (10 short frames at 0.025 s, or 5 at 0.05 s):
  band levels      mean of the normalised spectrogram in each band
  flux             mean absolute change between short frames (all bands)
  up / down sweep  share of energy moving one band up / down between short frames
then z-scored and PCA to 20 with train-half statistics. A unit (~1 s) spans ~4 frames,
so a sequence of frames follows the units of a phrase in order.
"""

from __future__ import annotations

import numpy as np

FRAME_S = 0.25


def frame_raw(sw: np.ndarray, fps: int) -> np.ndarray:
    k = int(round(FRAME_S * fps))
    n = len(sw) // k
    x = sw[: n * k].astype(np.float64)
    e = np.maximum(x - x.min(), 0)  # non-negative energy-like values for the sweep shares
    cur, nxt = e[:-1], e[1:]
    up = np.r_[(cur[:, :-1] * nxt[:, 1:]).sum(1), 0]
    down = np.r_[(cur[:, 1:] * nxt[:, :-1]).sum(1), 0]
    same = np.r_[(cur * nxt).sum(1), 0]
    tot = up + down + same + 1e-9
    flux = np.r_[np.abs(np.diff(x, axis=0)).mean(1), 0]
    per = lambda v: v[: n * k].reshape(n, k).mean(1)  # noqa: E731
    bands = x.reshape(n, k, -1).mean(1)
    return np.column_stack([bands, per(flux), per(up / tot), per(down / tot)])


def frames(sw_tr: np.ndarray, sw_te: np.ndarray, fps: int, n_pc: int = 20) -> dict:
    tr, te = frame_raw(sw_tr, fps), frame_raw(sw_te, fps)
    mu, sd = tr.mean(0), np.maximum(tr.std(0), 1e-6)
    ztr, zte = (tr - mu) / sd, (te - mu) / sd
    _, sv, vt = np.linalg.svd(ztr - ztr.mean(0), full_matrices=False)
    b = vt[:n_pc].T
    return {"train": ztr @ b, "test": zte @ b, "explained": float(np.sum(sv[:n_pc] ** 2) / np.sum(sv**2))}
