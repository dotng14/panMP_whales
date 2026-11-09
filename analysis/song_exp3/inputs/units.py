"""X4 unit events: find each unit as a blob in time and frequency and measure it.

1. Robust per-band score z = (value - band median) / (1.4826 * band MAD).
2. A cell is 'on' when z > Z_ON. Gaps up to CLOSE_S in time (pulses inside one unit,
   which Payne & McVay call subunits) are closed, and one band of frequency gap is bridged.
3. Connected cells (8-neighbour, time x frequency) form a blob; blobs shorter than
   MIN_DUR_S or smaller than MIN_CELLS are dropped. Blobs that overlap in time by
   at least OVERLAP of the shorter one are one sound (e.g. stacked harmonics) and merged.
4. Each unit gets the measurements below (energy / height / width / extras).
5. Unit types: a Gaussian mixture on the standardised shape measurements of the
   train-half units, K chosen by BIC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

Z_ON = 3.0
CLOSE_S = 0.1
MIN_DUR_S = 0.05
MIN_CELLS = 4
OVERLAP = 0.7
FIELDS = (
    "start_s", "end_s", "duration_s",           # width (time)
    "energy", "peak_z",                         # energy
    "peak_hz", "start_hz", "end_hz", "slope_st_per_s",  # height
    "bandwidth_st",                             # width (frequency)
    "pulse_hz", "n_harmonics", "gap_next_s",    # extras
)  # fmt: skip
SHAPE_FIELDS = ("duration_s", "peak_hz", "start_hz", "end_hz", "slope_st_per_s", "bandwidth_st", "pulse_hz", "n_harmonics")


def robust_z(v: np.ndarray) -> np.ndarray:
    """Per-band score: distance above the band median in units of its spread.

    The spread is the MAD, but never less than half the standard deviation: after a
    clipped floor (N1, N3) half of every band is exactly 0 and its MAD is 0.
    """
    med = np.median(v, axis=0)
    mad = np.median(np.abs(v - med), axis=0) * 1.4826
    scale = np.maximum(np.maximum(mad, 0.5 * v.std(axis=0)), 1e-6)
    return ((v - med) / scale).astype(np.float32)


def _semitone(hz):
    return 12 * np.log2(np.asarray(hz) / 100.0)


def detect(v: np.ndarray, fps: int, edges: np.ndarray, t0: float = 0.0, z_on: float = Z_ON) -> np.ndarray:
    """Units in a normalised spectrogram ``v`` (frames, bands). Returns a structured array with FIELDS."""
    z = robust_z(v)
    on = z > z_on
    close_t = max(1, int(round(CLOSE_S * fps)))
    on = ndimage.binary_closing(on, structure=np.ones((close_t + 1, 2), bool)) | on
    lab, n = ndimage.label(on, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.zeros(0, dtype=[(f, "f8") for f in FIELDS])
    slices = ndimage.find_objects(lab)
    centre_hz = np.sqrt(edges[:, 0] * edges[:, 1])
    st = _semitone(centre_hz)
    blobs = []
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        a, b = sl[0].start, sl[0].stop
        mask = lab[sl] == i
        if (b - a) / fps < MIN_DUR_S or mask.sum() < MIN_CELLS:
            continue
        blobs.append([a, b, sl[1].start, sl[1].stop, i])
    blobs.sort()
    # merge blobs that overlap mostly in time (one sound)
    merged: list[list] = []
    for bl in blobs:
        if merged:
            m = merged[-1]
            ov = min(m[1], bl[1]) - max(m[0], bl[0])
            if ov > 0 and ov >= OVERLAP * min(m[1] - m[0], bl[1] - bl[0]):
                m[0], m[1] = min(m[0], bl[0]), max(m[1], bl[1])
                m[2], m[3] = min(m[2], bl[2]), max(m[3], bl[3])
                m[4].append(bl[4])
                continue
        merged.append([bl[0], bl[1], bl[2], bl[3], [bl[4]]])
    out = np.zeros(len(merged), dtype=[(f, "f8") for f in FIELDS])
    for k, (a, b, f0, f1, ids) in enumerate(merged):
        m = np.isin(lab[a:b, f0:f1], ids)
        w = np.where(m, np.maximum(z[a:b, f0:f1], 0), 0.0)
        frame_e = w.sum(1)
        cent = (w * st[f0:f1]).sum(1) / np.maximum(frame_e, 1e-9)
        ok = frame_e > 0
        tt = np.arange(b - a)[ok] / fps
        if ok.sum() >= 3:  # start / end pitch from the fitted contour: the first and last frames are smeared by the window
            slope, icpt = (float(v_) for v_ in np.polyfit(tt, cent[ok], 1))
            st0, st1 = icpt, icpt + slope * (b - a - 1) / fps
        else:
            slope = 0.0
            st0 = st1 = float(cent[ok][0]) if ok.any() else 0.0
        span = np.array([np.ptp(np.flatnonzero(r)) + 1 if r.any() else 0 for r in m]) * 12 * np.log2(edges[0, 1] / edges[0, 0]) if f1 > f0 else np.zeros(1)
        prof = w.sum(0)
        pk = int(np.argmax(prof))
        peaks = np.flatnonzero((prof > 0.3 * prof.max()) & (prof >= np.r_[0, prof[:-1]]) & (prof >= np.r_[prof[1:], 0]))
        pulse = 0.0
        k_ma = max(3, int(0.25 * fps))
        if b - a >= max(int(0.3 * fps), 2 * k_ma + 4):
            core = frame_e[2:-2]  # drop the onset / offset ramps
            e = core - np.convolve(core, np.ones(k_ma) / k_ma, mode="same")  # keep only fast changes
            spec = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
            fr = np.fft.rfftfreq(len(e), 1 / fps)
            band = fr >= 3.0
            if band.any() and spec[band].max() > 0.3 * spec[band].sum() and np.std(core) > 0.1 * np.mean(core):
                pulse = float(fr[band][np.argmax(spec[band])])
        out[k] = (
            t0 + a / fps, t0 + b / fps, (b - a) / fps,
            float(w.sum()), float(z[a:b, f0:f1][m].max()),
            float(centre_hz[f0 + pk]), float(2 ** (st0 / 12) * 100), float(2 ** (st1 / 12) * 100), slope,
            float(np.mean(span[span > 0])) if np.any(span > 0) else 0.0,
            pulse, float(len(peaks)), 0.0,
        )  # fmt: skip
    if len(out) > 1:
        out["gap_next_s"][:-1] = np.maximum(out["start_s"][1:] - out["end_s"][:-1], 0)
    return out


def shape_matrix(u: np.ndarray) -> np.ndarray:
    """Measurements used for unit types (log scales where the quantity is multiplicative)."""
    return np.column_stack([
        np.log(u["duration_s"]), _semitone(np.maximum(u["peak_hz"], 1)), _semitone(np.maximum(u["start_hz"], 1)),
        _semitone(np.maximum(u["end_hz"], 1)), np.clip(u["slope_st_per_s"], -200, 200) / 20, u["bandwidth_st"],
        u["pulse_hz"] > 0, u["n_harmonics"],
    ]).astype(np.float64)  # fmt: skip


@dataclass
class UnitTypes:
    mean: np.ndarray
    std: np.ndarray
    gmm: object
    k: int

    def predict(self, u: np.ndarray) -> np.ndarray:
        return self.gmm.predict((shape_matrix(u) - self.mean) / self.std)


def fit_types(u_train: np.ndarray, ks=range(6, 21, 2), seed: int = 0, max_units: int = 20000) -> UnitTypes:
    from sklearn.mixture import GaussianMixture

    x = shape_matrix(u_train)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-6)
    xs = (x - mean) / std
    rng = np.random.default_rng(seed)
    sub = xs[rng.choice(len(xs), min(max_units, len(xs)), replace=False)]
    best, best_bic = None, np.inf
    for k in ks:
        g = GaussianMixture(k, covariance_type="full", random_state=seed, reg_covar=1e-3).fit(sub)
        bic = g.bic(sub)
        if bic < best_bic:
            best, best_bic = g, bic
    return UnitTypes(mean, std, best, best.n_components)


def token_windows(u: np.ndarray, types: np.ndarray, k: int, centres: np.ndarray, window_s: float = 20.0) -> np.ndarray:
    """Bag of unit types and type bigrams in each window: a vector per window (for screening)."""
    out = np.zeros((len(centres), k + k * k), np.float32)
    mid = 0.5 * (u["start_s"] + u["end_s"])
    order = np.argsort(mid)
    mid, types = mid[order], types[order]
    lo = np.searchsorted(mid, centres - window_s / 2)
    hi = np.searchsorted(mid, centres + window_s / 2)
    for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
        t = types[a:b]
        if len(t):
            np.add.at(out[i], t, 1.0)
        if len(t) > 1:
            np.add.at(out[i], k + t[:-1] * k + t[1:], 1.0)
    return np.sqrt(out)  # damp counts
