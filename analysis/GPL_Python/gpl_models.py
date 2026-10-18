from __future__ import annotations

from dataclasses import dataclass, field
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
        out = DetectorParams(**self.to_dict())
        out.nrec = int(window_samples)
        out.num_time_bins = int(np.floor((out.nrec - out.fftl) / out.fft_overlap))
        return out

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["filter_parm"] = FilterParams(values=FilterValues(self.filter_parm.values.adjacent_call_bin_num))
        return d

