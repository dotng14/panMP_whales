from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve2d

from .gpl_models import DetectorParams


def integer_wav_scale(dtype: np.dtype) -> float:
    """MATLAB ``audioread`` full-scale divisor for signed integer PCM."""
    if np.issubdtype(dtype, np.integer) and dtype.itemsize == 2:
        return 32768.0
    if np.issubdtype(dtype, np.integer):
        return float(2 ** (dtype.itemsize * 8 - 1))
    return 1.0


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    sample_rate, data = wavfile.read(path)
    if data.ndim == 2:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / integer_wav_scale(data.dtype)
    else:
        data = data.astype(np.float64)
    return data, int(sample_rate)


def base3x(baseline: np.ndarray) -> np.ndarray:
    """Match ``GPL_v4/base3x.m`` (column-major ``reshape`` pairing)."""
    flat = np.asarray(baseline, dtype=float).ravel()
    bs = np.sort(flat)
    n = len(bs)
    z = np.ones(n, dtype=float)
    bs1 = n
    if bs1 % 2 == 1:
        bs = bs[:-1]
        bs1 -= 1
    pairs = bs.reshape(bs1 // 2, 2, order="F")
    idx = int(np.argmin(pairs[:, 1] - pairs[:, 0]))
    low, high = pairs[idx, 0], pairs[idx, 1]
    z[flat < low] = 0.0
    z[flat > high] = 0.0
    return np.where(z)[0]


def matlab_csaps(x: np.ndarray, y: np.ndarray, p: float) -> np.ndarray:
    """Approximate MATLAB ``csaps(x, y, p)`` (Curve Fitting Toolbox)."""
    from scipy.interpolate import UnivariateSpline

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) < 2:
        return y.copy()
    if p >= 1.0:
        return UnivariateSpline(x, y, k=3, s=0.0)(x)
    if p <= 0.0:
        return np.full_like(y, np.mean(y))
    # MATLAB csaps balances fit error vs roughness with smoothness p in (0, 1).
    smooth = ((1.0 - p) / p) * len(x) * float(np.var(y))
    return UnivariateSpline(x, y, k=3, s=max(smooth, 1e-12))(x)


def gpl_spline_after(baseline0: np.ndarray) -> np.ndarray:
    """``GPL_v31`` final spline pass on the quiet baseline (before cropping)."""
    baseline0 = np.asarray(baseline0, dtype=np.float64).ravel()
    t = np.arange(1, len(baseline0) + 1, dtype=np.float64)
    log_spline = matlab_csaps(t, np.log(baseline0), 0.3)
    baseline_spline = np.exp(log_spline)
    b_spline, _ = whiten_vec(baseline_spline)
    b_spline = np.asarray(b_spline, dtype=np.float64)
    neg = b_spline <= 0
    b_spline[neg] = baseline_spline[neg]
    return b_spline


def whiten_matrix(sp: np.ndarray, fac: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp = np.asarray(sp, dtype=float)
    sz1, sz2 = sp.shape
    qs = np.full((sz1, sz2), np.nan, dtype=float)
    for i in range(sz1):
        ks = base3x(sp[i, :])
        if ks.size:
            qs[i, : len(ks)] = sp[i, ks]
    sqs = np.sum(qs, axis=0)
    qs = qs[:, np.isfinite(sqs)]
    if qs.size == 0:
        mu = np.zeros(sz1, dtype=float)
        return sp.copy(), qs, mu
    mu = np.mean(qs, axis=1)
    sp_whiten = sp - fac * mu[:, None]
    qs_out = qs - fac * mu[:, None]
    return sp_whiten, qs_out, mu


def whiten_vec(sp: np.ndarray) -> tuple[np.ndarray, float]:
    flat = np.asarray(sp, dtype=float).ravel()
    ks = base3x(flat)
    if ks.size == 0:
        mu = float(np.mean(flat)) if flat.size else 0.0
    else:
        mu = float(np.mean(flat[ks]))
    return flat - mu, mu


class STFTProcessor:
    def compute(self, data: np.ndarray, parm: DetectorParams) -> np.ndarray:
        x = np.asarray(data, dtype=float).ravel()
        if parm.num_time_bins <= 0:
            return np.zeros((parm.fftl, 0), dtype=float)
        # Match GPL_v4/GPL_fft.m: full abs(FFT) rows, hamming(fftl).
        win = np.hamming(parm.fftl)
        sp = np.zeros((parm.fftl, parm.num_time_bins), dtype=float)
        for j in range(parm.num_time_bins):
            start = j * parm.fft_overlap
            finish = start + parm.fftl
            if finish > len(x):
                break
            q = np.fft.fft(x[start:finish] * win)
            sp[:, j] = np.abs(q)
        return sp


class WhiteningProcessor:
    def apply(self, sp: np.ndarray, parm: DetectorParams) -> np.ndarray:
        spc, _, mu = whiten_matrix(sp, parm.white_x)
        sp_whiten = np.abs(spc / (mu[:, None]))
        if parm.whiten == 2:
            sp2, _, mu2 = whiten_matrix(sp_whiten.T)
            sp_whiten = np.abs(sp2.T / mu2[None, :])
        cross = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=float) / 8.0
        return convolve2d(sp_whiten, cross, mode="same")


def gpl_shuffle(nc: int, deck: int) -> np.ndarray:
    cr = np.argsort(np.random.rand(deck)) + 1  # match MATLAB 1-based behavior conceptually
    return cr[:nc] - 1


class QuietEstimator:
    def estimate(
        self, sp: np.ndarray, sp_whiten: np.ndarray, parm: DetectorParams
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        low_off = 0
        high_off = parm.num_freq_bins
        sum_t = np.maximum(np.sum(sp_whiten**2, axis=0, keepdims=True) ** 0.5, 1e-12)
        sum_f = np.maximum(np.sum(sp_whiten.T**2, axis=0, keepdims=True).T ** 0.5, 1e-12)
        u = sp_whiten / sum_t
        y = sp_whiten / sum_f
        bas = np.abs(u) ** parm.xp1 * np.abs(y) ** parm.xp2
        baseline0 = np.sum(bas[low_off:high_off, :] ** 2, axis=0)

        qs = sp_whiten.copy()
        quiet_fft = sp.copy()
        lks = 0
        for _ in range(10):
            if lks >= parm.num_time_bins or qs.shape[1] == 0:
                break
            sum_t2 = np.maximum(np.sum(qs**2, axis=0, keepdims=True) ** 0.5, 1e-12)
            sum_f2 = np.maximum(np.sum(qs.T**2, axis=0, keepdims=True).T ** 0.5, 1e-12)
            u2 = qs / sum_t2
            y2 = qs / sum_f2
            bas2 = np.abs(u2) ** parm.xp1 * np.abs(y2) ** parm.xp2
            baseline1 = np.sum(bas2**2, axis=0)
            ks = np.where(baseline1 <= parm.noise_ceiling)[0]
            if ks.size < 2:
                return (
                    np.zeros_like(sp_whiten),
                    np.zeros_like(sp),
                    np.ones(parm.num_time_bins, dtype=int),
                    baseline0,
                )
            qs = qs[:, ks]
            quiet_fft = quiet_fft[:, ks]
            lks = len(ks)
            xtnd = int(np.ceil(parm.num_time_bins / len(ks)))
            dex = np.concatenate([gpl_shuffle(lks, lks) for _ in range(xtnd)])[: parm.num_time_bins]
            qs = qs[:, dex]
            quiet_fft = quiet_fft[:, dex]

        ks0 = np.where(baseline0 <= parm.noise_ceiling)[0]
        if ks0.size == 0:
            return np.zeros_like(sp_whiten), np.zeros_like(sp), np.ones(parm.num_time_bins, dtype=int), baseline0
        q0 = sp_whiten[:, ks0]
        rep = int(np.ceil(parm.num_time_bins / q0.shape[1]))
        q0 = np.tile(q0, rep)[:, : parm.num_time_bins]
        sum_t3 = np.maximum(np.sum(q0**2, axis=0, keepdims=True) ** 0.5, 1e-12)
        sum_f3 = np.maximum(np.sum(q0.T**2, axis=0, keepdims=True).T ** 0.5, 1e-12)
        bas3 = np.abs(q0 / sum_t3) ** parm.xp1 * np.abs(q0 / sum_f3) ** parm.xp2
        baseline1 = np.sum(bas3[: parm.num_freq_bins, :] ** 2, axis=0)
        _, _ = whiten_vec(baseline1)
        ks_refine = np.where(baseline1[: len(ks0)] > parm.thresh)[0]
        kept = np.delete(ks0, ks_refine)
        blocked = np.zeros(parm.num_time_bins, dtype=int)
        blocked[kept] = 1
        qs2 = sp_whiten[:, kept] if kept.size else np.zeros((sp_whiten.shape[0], 1))
        rep2 = int(np.ceil(parm.num_time_bins / qs2.shape[1]))
        quiet_whiten = np.tile(qs2, rep2)[:, : parm.num_time_bins]
        return quiet_whiten, quiet_fft[:, : parm.num_time_bins], blocked, baseline0

