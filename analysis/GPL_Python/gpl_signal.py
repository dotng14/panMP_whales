"""Read a WAV, take its STFT, and whiten the noise floor."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def matlab_crop_freq_bin_indices(
    fftl: int,
    sample_freq: float,
    freq_lo: float,
    freq_hi: float,
) -> tuple[int, int]:
    """1-based inclusive FreqBinLo/FreqBinHi on the GPL_fft spectrogram grid.

    Matches ``GPL_parameter_Humpback.m`` (``linspace`` crop axis, rounded).
    """
    freq_range = np.linspace(0.0, sample_freq / 2.0, fftl // 2)
    lo = int(np.argmin(np.abs(freq_range - freq_lo))) + 1
    hi = int(np.argmin(np.abs(freq_range - freq_hi))) + 1
    lo = int(max(1, round(lo)))
    hi = int(min(len(freq_range), round(hi)))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def matlab_noise_level_k(
    fftl: int,
    sample_freq: float,
    freq_lo: float,
    freq_hi: float,
) -> int:
    """Frequency-bin count ``K`` for noise-level estimation (MATLAB arange grid)."""
    freq = np.arange(fftl // 2 + 1, dtype=float) / fftl * sample_freq
    lo = int(np.argmin(np.abs(freq - freq_lo))) + 1
    hi = int(np.argmin(np.abs(freq - freq_hi))) + 1
    if lo > hi:
        lo, hi = hi, lo
    return hi - lo + 1


def slice_gpl_freq_band(sp_fft: np.ndarray, freq_bin_lo: int, freq_bin_hi: int) -> np.ndarray:
    """Crop a full GPL_fft spectrogram using 1-based inclusive MATLAB indices."""
    return sp_fft[freq_bin_lo - 1 : freq_bin_hi, :]


@dataclass
class FilterValues:
    adjacent_call_bin_num: int = 5


@dataclass
class FilterParams:
    values: FilterValues = field(default_factory=FilterValues)


CONTOURS = ("cm", "cm_max", "cm_max2")


@dataclass
class MeasureSwitches:
    """``parm.measure.*`` of ``GPL_parameter_Humpback.m``: which contour measurements to take."""

    cm_freq_limits: int = 1
    cm_max_freq_limits: int = 0
    cm_max2_freq_limits: int = 0
    cm_freq_bandwidth: int = 1
    cm_max_freq_bandwidth: int = 0
    cm_max2_freq_bandwidth: int = 0
    cm_duration: int = 1
    cm_max_duration: int = 0
    cm_max2_duration: int = 0
    cm_slope: int = 1
    cm_max_slope: int = 0
    cm_max2_slope: int = 0
    spec_noise: int = 0
    rl_min_size: int = 5
    spec_rl: int = 0

    def on(self, contour: str, measure: str) -> bool:
        return bool(getattr(self, f"{contour}_{measure}"))


@dataclass
class ContourLimits:
    """Acceptance limits for one contour (``parm.filter_parm.values.<contour>_*``).

    ``None`` takes the MATLAB default: ``min_duration_s`` = ``min_call_duration``,
    the start / min frequencies = ``freq_lo``, the end / max frequencies = ``freq_hi``.
    """

    min_duration_s: float | None = None
    min_freq_bandwidth_hz: float = 50.0
    start_freq_hz: float | None = None
    end_freq_hz: float | None = None
    min_freq_hz: float | None = None
    max_freq_hz: float | None = None
    slope_lower: float = -200.0
    slope_upper: float = 2000.0


@dataclass
class ContourParams:
    """Contour extraction, measurement and filter switches (``GPL_parameter_Humpback.m`` defaults)."""

    template_on: int = 1  # extract contours (0: start/end times only)
    cm_on: int = 1  # all islands
    cm_max_on: int = 0  # the strongest island
    cm_max2_on: int = 0  # the second strongest island
    measurements_on: int = 1
    filter_on: int = 1  # pass/fail flags against ``limits``
    contour_cutoff: float = 6.0  # ContourCutoff: whitened level (over the call's noise) a pixel must pass
    min_island_size: int = 10  # MinIslandSize: pixels needed for a contour
    smaller_island_cutoff: float = 0.05  # islands in cm must be this fraction of the largest island's area
    island_radius: int = 1  # only 1 (direct neighbours) is supported
    measure: MeasureSwitches = field(default_factory=MeasureSwitches)
    limits: dict[str, ContourLimits] = field(default_factory=lambda: {c: ContourLimits() for c in CONTOURS})

    def contour_on(self, contour: str) -> bool:
        return bool(getattr(self, f"{contour}_on"))


@dataclass
class DetectorParams:
    sample_freq: float
    total_samples: int
    nwav: int = 1
    window_length: float = 75.0
    deck_test_adjustment: int = 0
    future_sample_ext: int = 1
    fifo_removal: int = 0
    spline_before: int = 0
    spline_loop: int = 0
    spline_after: int = 1
    xp1: float = 1.0
    xp2: float = 2.0
    whiten: int = 1
    white_x: float = 1.0
    num_loops: int = 6
    freq_lo: float = 10.0
    freq_hi: float = 10000.0
    min_call_duration: float = 0.2
    max_call_duration: float = 10.0
    freq_resolution_hz: float = 10.0
    skip_percent: float = 75.0
    plot: int = 0
    save_csv: int = 1
    output_size_separation: int = 1
    filter_parm: FilterParams = field(default_factory=FilterParams)
    contour: ContourParams = field(default_factory=ContourParams)
    noise_ceiling: float = 0.0
    thresh: float = 0.0
    noise_level: float = 0.0
    fftl: int = 0
    fft_overlap: int = 0
    nrec: int = 0
    freq_bin_lo: int = 0
    freq_bin_hi: int = 0
    sum_freq_bin_lo: int = 0
    sum_freq_bin_hi: int = 0
    num_freq_bins: int = 0
    num_time_bins: int = 0

    def crop_freq_slice(self) -> slice:
        """0-based slice for ``sp_fft[slice]`` using 1-based ``freq_bin_lo/hi``."""
        return slice(self.freq_bin_lo - 1, self.freq_bin_hi)

    @classmethod
    def for_audio(
        cls,
        sample_freq: float,
        total_samples: int,
        nwav: int = 1,
        *,
        freq_lo: float = 10.0,
        freq_hi: float = 10000.0,
    ) -> "DetectorParams":
        if freq_lo < 0:
            raise ValueError(f"freq_lo must be >= 0, got {freq_lo}")
        if freq_hi <= freq_lo:
            raise ValueError(f"freq_hi must be > freq_lo, got {freq_lo}..{freq_hi}")
        nyquist = sample_freq / 2.0
        if freq_hi > nyquist:
            raise ValueError(
                f"freq_hi ({freq_hi} Hz) exceeds Nyquist ({nyquist} Hz) "
                f"for sample_freq={sample_freq}"
            )

        p = cls(
            sample_freq=sample_freq,
            total_samples=total_samples,
            nwav=nwav,
            freq_lo=freq_lo,
            freq_hi=freq_hi,
        )
        p.nrec = int(round(p.window_length * p.sample_freq)) if nwav == 1 else int(total_samples / nwav)
        p.fftl = int(2 ** np.ceil(np.log2(max(16.0, p.sample_freq / p.freq_resolution_hz))))
        p.fft_overlap = int(round(p.fftl - (1 - p.skip_percent * 1e-2) * p.fftl))
        p.num_time_bins = int(np.floor((p.nrec - p.fftl) / p.fft_overlap))
        p.freq_bin_lo, p.freq_bin_hi = matlab_crop_freq_bin_indices(
            p.fftl, p.sample_freq, p.freq_lo, p.freq_hi
        )
        p.sum_freq_bin_lo = p.freq_bin_lo
        p.sum_freq_bin_hi = p.freq_bin_hi
        p.num_freq_bins = p.freq_bin_hi - p.freq_bin_lo + 1

        # MATLAB noise_level block from GPL_parameter_Humpback.m
        pp = np.arange(3, 18, dtype=float)
        mm = np.array(
            [
                0.161023825523259,
                0.149449174795745,
                0.166356950686883,
                0.215366733811165,
                0.314728324708629,
                0.506543111049952,
                0.881888744912427,
                1.640324687324695,
                3.231593355705235,
                6.701372610649511,
                14.55798970370740,
                33.00422800929706,
                77.83704351512681,
                190.4458954305009,
                482.2813101318087,
            ],
            dtype=float,
        )
        p_exp = 2 * (p.xp1 + p.xp2)
        mu_p = float(np.interp(p_exp, pp, mm))
        mu_2 = 1 - np.pi / 4
        k = matlab_noise_level_k(p.fftl, p.sample_freq, p.freq_lo, p.freq_hi)
        j = max(1, int(np.floor((p.nrec - p.fftl) / p.fft_overlap) + 1))
        if p.whiten == 1:
            p.noise_level = 0.444 * mu_p ** (3 / 4) / (j ** p.xp2 * k ** (p.xp1 - 1) * mu_2 ** (p_exp / 2))
        else:
            p.noise_level = np.exp(-3.5) * p_exp**2 * mu_p ** (3 / 4) / (
                j ** p.xp2 * k ** (p.xp1 - 1) * mu_2 ** (p_exp / 2)
            )
        p.noise_ceiling = 0.5 * p.noise_level
        p.thresh = 1.5 * p.noise_level
        return p

    def with_window_samples(self, window_samples: int) -> "DetectorParams":
        out = copy.deepcopy(self)
        out.nrec = int(window_samples)
        out.num_time_bins = int(np.floor((out.nrec - out.fftl) / out.fft_overlap))
        return out

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["filter_parm"] = FilterParams(values=FilterValues(self.filter_parm.values.adjacent_call_bin_num))
        d["contour"] = copy.deepcopy(self.contour)
        return d

    def with_overrides(self, overrides: dict[str, Any]) -> "DetectorParams":
        """A copy with values from a parameters file (``load_params_file``) applied.

        Nested sections (``contour``, ``contour.measure``, ``contour.limits.cm`` ...)
        merge key by key. Values derived from the sample rate and band
        (``fftl``, bins, thresholds) and the band itself (``--freq-lo`` /
        ``--freq-hi``) cannot be set here.
        """
        out = copy.deepcopy(self)
        bad = sorted(set(overrides) & DERIVED_PARAMS)
        if bad:
            raise ValueError(f"parameters file cannot set {', '.join(bad)} (derived, or set on the command line)")
        _apply(out, overrides, "")
        return out


# Set by ``for_audio`` from the sample rate and band, not by a parameters file.
DERIVED_PARAMS = frozenset(
    {
        "sample_freq", "total_samples", "nwav", "freq_lo", "freq_hi", "noise_ceiling", "thresh",
        "noise_level", "fftl", "fft_overlap", "nrec", "freq_bin_lo", "freq_bin_hi",
        "sum_freq_bin_lo", "sum_freq_bin_hi", "num_freq_bins", "num_time_bins",
    }
)


def _apply(target: Any, values: dict[str, Any], where: str) -> None:
    for key, value in values.items():
        name = f"{where}{key}"
        if isinstance(target, dict):
            if key not in target:
                raise ValueError(f"unknown parameter {name!r}")
            current = target[key]
        else:
            known = {f.name for f in fields(target)}
            if key not in known:
                raise ValueError(f"unknown parameter {name!r}")
            current = getattr(target, key)
        if is_dataclass(current) or isinstance(current, dict):
            if not isinstance(value, dict):
                raise ValueError(f"parameter {name!r} is a section; give an object")
            _apply(current, value, f"{name}.")
        elif isinstance(target, dict):
            target[key] = value
        else:
            setattr(target, key, value)


def load_params_file(path: Path | str) -> dict[str, Any]:
    """Read a JSON parameters file (see ``gpl_params.example.json``); keys starting with ``_`` are comments."""
    def strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if not k.startswith("_")}
        return obj

    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a parameters file holds one JSON object")
    return strip(data)

import os
import numpy as np

try:
    import cupy as cp  # type: ignore[import]
    _CUPY_AVAILABLE = True
except ModuleNotFoundError:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False

_BACKEND = os.environ.get("GPL_COMPUTE", "cpu").lower()

if _CUPY_AVAILABLE and _BACKEND == "cuda" and cp is not None:
    # Reuse freed VRAM blocks instead of calling cudaMalloc each time.
    cp.get_default_memory_pool().set_limit(size=2 * 1024**3)  # 2 GB per GPU


def get_array_module(gpu_id: int = 0):
    """Return cupy (with the given device active) or numpy."""
    if _BACKEND == "cuda" and _CUPY_AVAILABLE and cp is not None:
        cp.cuda.Device(gpu_id).use()
        return cp
    return np


def numpy_from(xp, arr) -> np.ndarray:
    """Convert a cupy or numpy array to a plain numpy array."""
    if xp is np:
        return np.asarray(arr)
    return cp.asnumpy(arr)

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve2d



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
