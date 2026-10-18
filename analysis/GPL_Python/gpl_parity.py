"""Python GPL parity checkpoints mirroring ``GPL_v4/gpl_parity_export.m``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .gpl_detector import GPLDetector
from .gpl_models import DetectorParams, slice_gpl_freq_band
from .gpl_signal import QuietEstimator, STFTProcessor, WhiteningProcessor, read_wav_mono, whiten_matrix


def compute_python_parity(
    wav_path: Path,
    *,
    freq_lo: float = 10.0,
    freq_hi: float = 10_000.0,
    window_sec: float = 75.0,
    start_sample: int = 1,
    fifo_removal: int = 0,
) -> dict[str, Any]:
    """Compute checkpoint arrays on one GPL window (Python GPL_Python path)."""
    wav_path = wav_path.expanduser().resolve()
    audio, sample_freq = read_wav_mono(wav_path)

    nrec = int(round(window_sec * sample_freq))
    start_idx = max(0, int(start_sample) - 1)
    finish_idx = start_idx + nrec
    if finish_idx > len(audio):
        raise ValueError(
            f"Window exceeds audio length: need {finish_idx} samples, have {len(audio)}"
        )
    sub_data = audio[start_idx:finish_idx]

    parm = DetectorParams.for_audio(
        sample_freq,
        len(audio),
        nwav=1,
        freq_lo=freq_lo,
        freq_hi=freq_hi,
    )
    local = parm.with_window_samples(len(sub_data))
    if fifo_removal:
        raise NotImplementedError("Python parity export does not implement fifo_removal yet")

    sp_fft_full = STFTProcessor().compute(sub_data, local)
    sp_band = slice_gpl_freq_band(sp_fft_full, local.freq_bin_lo, local.freq_bin_hi)

    whitener = WhiteningProcessor()
    sp_whiten = whitener.apply(sp_band, local)
    spc, _, mu = whiten_matrix(sp_band, local.white_x)
    sp_whiten_norm = np.abs(spc / (mu[:, None]))

    quiet_est = QuietEstimator()
    _, _, blocked, baseline0 = quiet_est.estimate(sp_band, sp_whiten, local)
    base_save = baseline0

    sp_loop = sp_whiten.copy()
    sum_t = np.maximum(np.sum(sp_loop**2, axis=0, keepdims=True) ** 0.5, 1e-12)
    sum_f = np.maximum(np.sum(sp_loop.T**2, axis=0, keepdims=True).T ** 0.5, 1e-12)
    norm_v = sp_loop / sum_t
    norm_h = sp_loop / sum_f
    sp_w_v, _, _ = whiten_matrix(norm_v.T, local.white_x)
    norm_v = sp_w_v.T
    sp_w_h, _, _ = whiten_matrix(norm_h, local.white_x)
    norm_h = sp_w_h
    bas = np.abs(norm_v) ** local.xp1 * np.abs(norm_h) ** local.xp2
    base_in_iter0 = np.sum(bas**2, axis=0)

    detector = GPLDetector()
    result = detector.detect_window(sub_data, local)
    det_start = [int(d["start_time"]) for d in result.detections]
    det_finish = [int(d["end_time"]) for d in result.detections]

    return {
        "engine": "GPL_Python",
        "wav_path": str(wav_path),
        "freq_lo": freq_lo,
        "freq_hi": freq_hi,
        "sample_freq": sample_freq,
        "start_sample": start_sample,
        "nrec": nrec,
        "fftl": local.fftl,
        "fft_overlap": local.fft_overlap,
        "num_time_bins": local.num_time_bins,
        "freq_bin_lo": local.freq_bin_lo,
        "freq_bin_hi": local.freq_bin_hi,
        "num_freq_bins": local.num_freq_bins,
        "noise_ceiling": local.noise_ceiling,
        "thresh": local.thresh,
        "sp_fft_pos": sp_fft_full[: local.fftl // 2 + 1, :],
        "sp_band": sp_band,
        "mu": mu,
        "sp_whiten_norm": sp_whiten_norm,
        "sp_whiten": sp_whiten,
        "baseline0": baseline0,
        "base_save": base_save,
        "base_in_iter0": base_in_iter0,
        "blocked": blocked,
        "n_detections": len(result.detections),
        "det_start_bin": np.asarray(det_start, dtype=np.int64),
        "det_finish_bin": np.asarray(det_finish, dtype=np.int64),
    }
