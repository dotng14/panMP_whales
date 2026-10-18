from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gpl_models import DetectorParams, slice_gpl_freq_band
from .gpl_signal import (
    QuietEstimator,
    STFTProcessor,
    WhiteningProcessor,
    gpl_spline_after,
    whiten_matrix,
    whiten_vec,
)


def gpl_cropping(base_in: np.ndarray, noise_ceiling: float, thresh: float) -> tuple[np.ndarray, np.ndarray]:
    base_out = np.asarray(base_in, dtype=float).copy()
    calls: list[tuple[int, int]] = []
    while True:
        k2 = int(np.argmax(base_out))
        k1 = float(base_out[k2])
        if k1 < thresh:
            break
        js = np.where(base_out < noise_ceiling)[0]
        left = js[js < k2]
        right = js[js > k2]
        js1 = 0 if left.size == 0 else int(left[-1])
        js2 = len(base_out) - 1 if right.size == 0 else int(right[0])
        calls.append((js1, js2))
        base_out[js1 : js2 + 1] = 0.0
    if not calls:
        return base_out, np.zeros((0, 2), dtype=int)
    # convert to MATLAB-like 1-based bins for downstream parity logic
    arr = np.array(calls, dtype=int) + 1
    return base_out, arr


def merge_calls(calls: np.ndarray, adjacent_call_bin_num: int, max_duration_bins: int) -> np.ndarray:
    if calls.size == 0:
        return calls.reshape(0, 2)
    calls = calls[np.argsort(calls[:, 0])]
    merged = [calls[0].copy()]
    for i in range(1, len(calls)):
        candidate_end = max(merged[-1][1], calls[i][1])
        gap = calls[i][0] - merged[-1][1]
        candidate_duration = candidate_end - merged[-1][0]
        if gap <= adjacent_call_bin_num and candidate_duration <= max_duration_bins:
            merged[-1][1] = candidate_end
        else:
            merged.append(calls[i].copy())
    return np.asarray(merged, dtype=int)


def call_duration_filter(mcalls: np.ndarray, parm: DetectorParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mcalls.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.zeros((0, 2), dtype=int)
    dt = parm.fft_overlap / parm.sample_freq
    cutoff_long = int(np.ceil(parm.max_call_duration / dt))
    calls = merge_calls(mcalls, parm.filter_parm.values.adjacent_call_bin_num, cutoff_long)
    start = calls[:, 0]
    finish = calls[:, 1]
    cutoff_short = int(np.ceil(parm.min_call_duration / dt))
    mask = (finish - start >= cutoff_short) & (finish - start <= cutoff_long)
    start = start[mask]
    finish = finish[mask]
    times = np.column_stack(((start - 1) * parm.fft_overlap + 1, finish * parm.fft_overlap))
    return start, finish, times


@dataclass
class DetectorResult:
    detections: list[dict]
    sp_whiten: np.ndarray
    baseline_raw: np.ndarray
    baseline_loop: np.ndarray


class GPLDetector:
    def __init__(self) -> None:
        self.stft = STFTProcessor()
        self.whitener = WhiteningProcessor()
        self.quiet = QuietEstimator()

    def detect_window(self, sub_data: np.ndarray, parm: DetectorParams) -> DetectorResult:
        sp_full = self.stft.compute(sub_data, parm)
        sp = slice_gpl_freq_band(sp_full, parm.freq_bin_lo, parm.freq_bin_hi)
        sp_whiten = self.whitener.apply(sp, parm)
        sp_loop = sp_whiten.copy()
        quiet_whiten, _, blocked, baseline0 = self.quiet.estimate(sp, sp_whiten, parm)
        base_save = baseline0.copy()

        mcalls = np.zeros((0, 2), dtype=int)
        iteration_count = 0
        num_calls = 2
        dt = parm.fft_overlap / parm.sample_freq
        max_duration_bins = int(np.ceil(parm.max_call_duration / dt))

        while iteration_count < parm.num_loops and num_calls > 1:
            norm_v = sp_loop / np.maximum(np.sum(sp_loop**2, axis=0, keepdims=True) ** 0.5, 1e-12)
            norm_h = sp_loop / np.maximum(np.sum(sp_loop.T**2, axis=0, keepdims=True).T ** 0.5, 1e-12)
            norm_v, _, _ = whiten_matrix(norm_v.T)
            norm_v = norm_v.T
            norm_h, _, _ = whiten_matrix(norm_h)
            bas = np.abs(norm_v) ** parm.xp1 * np.abs(norm_h) ** parm.xp2
            base_in = np.sum(bas[:, :] ** 2, axis=0)
            b0, _ = whiten_vec(base_in)
            base_in = b0

            base_out, calls = gpl_cropping(base_in, parm.noise_ceiling, parm.thresh)
            if calls.size:
                calls = calls[np.argsort(calls[:, 0])]
                # Merge touching calls exactly as MATLAB pre-pass
                merged: list[list[int]] = []
                i = 0
                while i < len(calls):
                    s, e = int(calls[i, 0]), int(calls[i, 1])
                    while i + 1 < len(calls) and int(calls[i, 1]) == int(calls[i + 1, 0]):
                        e = int(calls[i + 1, 1])
                        i += 1
                    merged.append([s, e])
                    i += 1
                calls = np.asarray(merged, dtype=int)
            num_calls = 0 if calls.size == 0 else len(calls)
            unblocked = np.zeros(parm.num_time_bins, dtype=int)
            for j in range(num_calls):
                st = max(1, int(calls[j, 0]))
                fn = min(parm.num_time_bins, int(calls[j, 1]))
                unblocked[st - 1 : fn] = 1
            tst = unblocked - blocked
            tst[tst < 1] = 0

            st = np.where(np.diff(tst) > 0)[0] + 2
            if tst[0] == 1:
                st = np.concatenate(([1], st))
            fn = np.where(np.diff(tst) < 0)[0] + 1
            if tst[-1] == 1:
                fn = np.concatenate((fn, [parm.num_time_bins]))
            calls = np.column_stack((st, fn)) if st.size and fn.size else np.zeros((0, 2), dtype=int)
            if calls.size:
                calls = merge_calls(calls, parm.filter_parm.values.adjacent_call_bin_num, max_duration_bins)
                # MATLAB equivalent: dur = diff(calls');
                # In MATLAB this returns 1xN (row-wise duration per call).
                # NumPy default diff axis differs, so force axis=0 on calls.T.
                dur = np.diff(calls.T, axis=0).ravel()
                calls = calls[dur != 0] if calls.size else calls
                mcalls = np.vstack((mcalls, calls)) if mcalls.size else calls
            iteration_count += 1
            num_calls = len(calls)
            blocked = np.clip(blocked + unblocked, 0, 1)
            ksel = np.where(base_out == 0)[0]
            if ksel.size:
                sp_loop[:, ksel] = quiet_whiten[:, ksel]

        if parm.spline_after:
            baseline_spline = gpl_spline_after(baseline0)
            _, calls_spline = gpl_cropping(baseline_spline, parm.noise_ceiling, parm.thresh)
            if calls_spline.size:
                mcalls = (
                    merge_calls(
                        np.vstack((mcalls, calls_spline)) if mcalls.size else calls_spline,
                        parm.filter_parm.values.adjacent_call_bin_num,
                        max_duration_bins,
                    )
                )

        start, finish, times = call_duration_filter(mcalls, parm)
        boundary_start = int(len(mcalls) > 0 and np.min(mcalls[:, 0]) == 1)
        boundary_end = int(len(mcalls) > 0 and np.max(mcalls[:, 1]) == parm.num_time_bins)
        detections: list[dict] = []
        baseline_loop = base_in if "base_in" in locals() else baseline0
        for idx in range(len(times)):
            bflag = 0
            if idx == 0 and boundary_start:
                bflag = 1
            if idx == len(times) - 1 and boundary_end:
                bflag = 1
            sbin = max(1, int(start[idx]))
            fbin = min(len(base_save), int(finish[idx]))
            detections.append(
                {
                    "start_time": int(times[idx, 0]),
                    "end_time": int(times[idx, 1]),
                    "boundary_cross_flag": int(bflag),
                    "energy": base_save[sbin - 1 : fbin].copy(),
                    "energy2": baseline_loop[sbin - 1 : fbin].copy() if len(baseline_loop) else np.array([]),
                    "sp": sp_whiten[:, sbin - 1 : fbin].copy(),
                }
            )

        return DetectorResult(
            detections=detections,
            sp_whiten=sp_whiten,
            baseline_raw=base_save,
            baseline_loop=baseline_loop,
        )

