"""Call contours and measurements (gpl_contour), checked against the MATLAB formulas they port."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from scipy.io import wavfile

from analysis.GPL_Python.gpl_outputs import load_detections_npy

from analysis.GPL_Python import gpl_contour as gc
from analysis.GPL_Python.gpl_signal import DetectorParams, load_params_file
from analysis.GPL_Python.process_v4 import ProcessingOrchestrator

SR = 10_000


def _parm(**contour) -> DetectorParams:
    parm = DetectorParams.for_audio(SR, 80_000, freq_lo=10.0, freq_hi=4_000.0)
    return parm.with_overrides({"contour": contour}) if contour else parm


def _ridges(cols: range, start: int, step: int) -> set[tuple[int, int]]:
    """A ridge two rows thick, so its pixels touch directly (island1x does not join diagonals)."""
    return {(start + step * (c - cols.start) + d, c) for c in cols for d in (0, 1)}


STRONG = _ridges(range(5, 35), 10, 1)  # upsweep, rows 10 -> 40
WEAK = _ridges(range(5, 15), 52, -1)  # short downsweep, far from the first


def _call(rows=60, cols=40, seed=0) -> np.ndarray:
    """Noise near 1 with a strong upsweep and a weaker short ridge."""
    rng = np.random.default_rng(seed)
    call = 1.0 + 0.1 * rng.standard_normal((rows, cols))
    for r, c in STRONG:
        call[r, c] = 100.0
    for r, c in WEAK:
        call[r, c] = 40.0
    return call


def test_contours_pick_all_islands_the_strongest_and_the_second():
    found = gc.find_contours(_call(), _parm())
    strong, weak = STRONG, WEAK
    as_set = lambda m: set(zip(*np.nonzero(m)))  # noqa: E731
    assert strong <= as_set(found["cm_max"]) and not weak & as_set(found["cm_max"])
    assert weak <= as_set(found["cm_max2"]) and not strong & as_set(found["cm_max2"])
    assert as_set(found["cm"]) == as_set(found["cm_max"]) | as_set(found["cm_max2"])
    assert found["cm"].shape == (60, 40)


def test_no_contour_below_the_island_size():
    call = np.ones((30, 30))
    call[5, 5:10] = 100.0  # 5 pixels <= MinIslandSize 10
    assert not any(c.any() for c in gc.find_contours(call, _parm()).values())


def test_islands_join_across_columns_like_linear_indices():
    """island1x joins index k with k+1: the last row of one column and the first of the next."""
    mask = np.zeros((4, 3), dtype=bool)
    mask[3, 0] = mask[0, 1] = True
    mask[2, 2] = True
    labels, order = gc._islands(mask)
    assert len(order) == 2 and labels[3, 0] == labels[0, 1] != labels[2, 2]


def test_sparse_round_trip_drops_negative_pixels():
    c = np.zeros((3, 4))
    c[0, 1], c[2, 3], c[1, 2] = 2.0, 1.0, -0.5
    s = gc.to_sparse(c)
    assert s["index"].dtype == np.uint32 and list(s["index"]) == [4, 8, 12]  # 1-based, column-major
    assert s["scale"] == 2.0 and list(s["values"]) == [65535, 0, 32768]
    back = gc.from_sparse(s)
    assert back[1, 2] == 0 and back[0, 1] == 2.0 and back[2, 3] == pytest.approx(1.0, abs=1e-4)


def test_measurements_follow_gpl_measurements_m():
    parm = _parm()
    c = np.zeros((20, 10))
    c[[2, 4], 1] = 1.0  # first column: rows 3 and 5 (1-based), mean 4
    c[5, 2] = c[6, 3] = 1.0
    c[9, 4] = 1.0  # last column: row 10
    m = gc.measure_contour(c, "cm", parm)
    df, dt, lo = SR / parm.fftl, parm.fft_overlap / SR, parm.freq_bin_lo
    assert m["duration_bin"] == 4 and m["duration_sec"] == pytest.approx(4 * dt)
    assert m["freq_bandwidth_bin"] == 5 and m["freq_bandwidth_hz"] == pytest.approx(5 * df)
    assert m["start_freq_hz"] == pytest.approx((4 + lo) * df - 2)
    assert m["end_freq_hz"] == pytest.approx((10 + lo) * df - 2)
    assert m["min_freq_hz"] == pytest.approx((3 + lo - 2) * df)
    assert m["max_freq_hz"] == pytest.approx((10 + lo - 2) * df)
    assert m["abs_bandwidth_hz"] == pytest.approx(m["max_freq_hz"] - m["min_freq_hz"] + 1)
    # Equal weights: the weighted fit is ordinary least squares of row on column.
    rows, cols = np.nonzero(c)
    p1, p0 = np.polyfit(cols + 1.0, rows + 1.0, 1)
    assert m["slope"] == pytest.approx(p1 * df / dt)
    assert m["slope_y_int"] == pytest.approx(p0 + lo - 1)
    assert m["slope_ci"] > 0


def test_slope_needs_three_columns():
    c = np.zeros((5, 5))
    c[1, 1] = c[2, 2] = 1.0
    m = gc.measure_contour(c, "cm", _parm())
    assert np.isnan(m["slope"]) and np.isnan(m["slope_ci"])


def test_filter_flags_and_nan():
    parm = _parm()
    flags = gc.filter_flags(
        {"duration_sec": 0.1, "freq_bandwidth_hz": 60.0, "start_freq_hz": float("nan"), "slope": float("nan")},
        "cm", parm,
    )
    # duration 0.1 < min_call_duration 0.2 fails; NaN passes a < / > test but not the slope range.
    assert flags == {"duration_sec": 0, "freq_bandwidth_hz": 1, "start_freq_hz": 1, "slope": 0}


def test_rl_is_the_contour_above_the_nearest_noise():
    sp = np.full((10, 10), 2.0)
    sp[4, 2:8] = 5.0
    cm = np.zeros_like(sp)
    cm[4, 2:8] = 1.0
    # Each contour pixel is 3 over its nearest noise: one row, six columns of 3.
    assert gc.estimate_rl(cm, sp) == pytest.approx(np.sqrt(np.mean([9.0] * 6 + [0.0] * 4)))


def test_switches_choose_contours_columns_and_rl():
    parm = _parm(cm_max_on=1, measure={"cm_max_slope": 1, "spec_rl": 1, "spec_noise": 1})
    call = gc.describe_call(_call(), _call() + 1.0, 0.5, parm)
    assert set(call) == {"cm", "cm_max", "spec_noise", "spec_rl", "filter_v3"}
    assert "slope" in call["cm_max"] and "start_freq_hz" not in call["cm_max"]
    cols = gc.table_columns(parm)
    assert "cm_max_slope" in cols and "cm_max_start_freq_hz" not in cols and "filter_cm_max_slope" in cols
    row = gc.table_row(call, cols)
    assert row["cm_max_slope"] == call["cm_max"]["slope"] and row["spec_noise"] == 0.5
    assert gc.describe_call(_call(), _call(), 0.0, _parm(template_on=0)) == {}


def test_end_to_end_writes_contour_columns_and_npy_calls(tmp_path):
    t = np.arange(SR * 8) / SR
    sig = 0.01 * np.random.default_rng(1).standard_normal(t.size)
    on = (t >= 3.0) & (t < 4.6)  # one 1.6 s upsweep, 500 -> 1500 Hz (625 Hz/s)
    tau = t[on] - 3.0
    sig[on] += 0.5 * np.sin(2 * np.pi * (500 * tau + 312.5 * tau**2))
    wav = tmp_path / "demo_20240101_000000.wav"
    wavfile.write(wav, SR, sig.astype(np.float32))
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"contour": {"cm_max_on": 1, "measure": {"cm_max_slope": 1}}}))

    out = tmp_path / "out"
    ProcessingOrchestrator(out, freq_lo=10.0, freq_hi=4_000.0, params=load_params_file(params)).process(wav)

    rows = list(csv.DictReader((out / "CSV_demo_20240101_000000.csv").open()))
    assert rows, "no detections"
    assert {"cm_slope", "cm_duration_sec", "cm_max_slope", "filter_cm_slope"} <= set(rows[0])
    sweep = max(rows, key=lambda r: float(r["cm_duration_sec"]))
    assert float(sweep["cm_duration_sec"]) > 1.0
    assert 450 < float(sweep["cm_slope"]) < 800
    assert 350 < float(sweep["start_freq_hz"]) < 650 and 1350 < float(sweep["end_freq_hz"]) < 1650
    det = load_detections_npy(out / "Detections_demo_20240101_000000.npy")
    calls = det["calls"]
    assert len(calls) == len(rows) and {"values", "index", "scale", "size", "slope"} <= set(calls[0]["cm"])
    assert not (out / "Detections_demo_20240101_000000.mat").exists()
