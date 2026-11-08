"""Check the 193500 song report against the published MATLAB PDF values.

Uses Audio Toolbox MFCC when ``coeffs_matlab_10hz.npz`` is present (see
``run_mfcc_full.m``). Matrix-profile kernel parity is checked separately in
``test_matlab_equivalence.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.matrix_profile import mpxForPan, mpx_fast
from analysis.matrix_profile.replicate_report.report_193500 import (
    MATLAB_COEFFS_NPZ,
    build_report,
)


pytestmark = pytest.mark.skipif(
    not MATLAB_COEFFS_NPZ.exists(),
    reason="MATLAB Audio Toolbox MFCC cache missing (coeffs_matlab_10hz.npz)",
)


@pytest.fixture(scope="module")
def report():
    return build_report()


def test_fast_profile_matches_translated_mpx():
    rng = np.random.default_rng(1)
    series = rng.standard_normal(900)
    ref_mp, ref_idx = mpxForPan(series, 48, 48)
    mp, idx = mpx_fast(series, 48, 48)
    np.testing.assert_allclose(mp, ref_mp, rtol=0.0, atol=1e-8, equal_nan=True)
    np.testing.assert_array_equal(idx, ref_idx)


def test_best_match_correlation_scales_from_7_to_20_minutes(report):
    assert round(report["best_r"]["7"], 2) == 0.63
    assert round(report["best_r"]["20"], 2) == 0.44
    assert report["best_r"]["20"] < report["best_r"]["7"]
    expected = {
        7: 0.63,
        8: 0.60,
        9: 0.57,
        10: 0.55,
        11: 0.52,
        12: 0.50,
        13: 0.50,
        14: 0.51,
        15: 0.50,
        16: 0.50,
        17: 0.49,
        18: 0.47,
        19: 0.46,
        20: 0.44,
    }
    for minutes, value in expected.items():
        assert round(report["best_r"][str(minutes)], 2) == value


def test_coarse_scan_pairs(report):
    expected = {
        "2 vs 4": (125, 66, 0.62, 4.5),
        "4 vs 8": (66, 51, 0.36, 2.6),
        "8 vs 16": (51, 31, 0.90, 7.4),
    }
    by_pair = {p["pair"]: p for p in report["coarse_pairs"]}
    for name, (n_a, n_b, ks, w_min) in expected.items():
        p = by_pair[name]
        assert (p["n_a"], p["n_b"]) == (n_a, n_b)
        assert round(p["ks"], 2) == ks
        assert round(p["w_min"], 1) == pytest.approx(w_min, abs=0.1)


def test_fine_scan_length_table(report):
    expected = {
        6: (54, 0.15, 11.5, (8.0, 13.7), 20, 53),
        7: (53, 0.13, 11.8, (8.5, 13.7), 18, 52),
        8: (51, 0.11, 12.2, (9.7, 13.8), 14, 50),
        9: (48, 0.11, 12.8, (11.1, 13.9), 7, 47),
        10: (47, 0.09, 12.8, (11.4, 14.0), 5, 46),
    }
    by_min = {int(r["minutes"]): r for r in report["fine_lengths"]}
    for minutes, (n, thr, med, iqr, under, n_gaps) in expected.items():
        row = by_min[minutes]
        assert row["occurrences"] == n
        assert round(row["threshold"], 2) == thr
        assert round(row["median_interval_min"], 1) == med
        assert (round(row["iqr"][0], 1), round(row["iqr"][1], 1)) == iqr
        assert (row["under_10"], row["n_gaps"]) == (under, n_gaps)


def test_fine_scan_neighbor_pairs(report):
    expected = {
        "6 vs 7": (2, 0.06, 0.2, 51, 54),
        "7 vs 8": (4, 0.13, 0.5, 47, 53),
        "8 vs 9": (6, 0.18, 0.8, 44, 51),
        "9 vs 10": (2, 0.11, 0.3, 44, 48),
    }
    by_pair = {p["pair"]: p for p in report["fine_pairs"]}
    for name, (pct, ks, w_min, shared, n_a) in expected.items():
        p = by_pair[name]
        assert round(p["count_diff_pct"]) == pct
        assert round(p["ks"], 2) == ks
        assert round(p["w_min"], 1) == pytest.approx(w_min, abs=0.1)
        assert (p["shared"], p["n_a"]) == (shared, n_a)


def test_song_markers(report):
    assert report["n_song_markers"] == 53
    assert report["n_songs"] == 51
    assert round(report["song_length_median_min"], 1) == 12.0
    assert round(report["song_length_min_min"], 1) == 7.1
    assert round(report["song_length_max_min"], 1) == 15.7


def test_phrase_groups_structure(report):
    assert report["n_phrase_groups"] >= 29
    assert report["n_kept"] == 7
    g1 = report["groups"][0]
    assert g1["occurrences"] == 30
    assert round(g1["seed_r"], 2) == 0.78
    assert round(g1["concentration"], 2) == 0.62
    g7 = report["groups"][6]
    assert g7["occurrences"] == 12
    assert round(g7["circular_mean"], 2) == 0.33
    assert round(g7["concentration"], 2) == 0.63


def test_energy_minima(report):
    minima = report["minima"]
    assert minima["n_minima"] in (28, 29)
    assert minima["n_inside"] >= 28
    assert round(minima["circular_mean"], 2) == 0.35
    assert minima["rayleigh_p"] == pytest.approx(3.6e-5, rel=0.5)
    assert minima["n_inside_marker"] >= 24
    assert round(minima["median_offset_min"], 1) == -1.9
