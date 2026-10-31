"""Check the 193500 song report against the published MATLAB values.

The report quotes correlations and KS distances to two decimals, Wasserstein
shifts and the minima interval to one decimal, and the Rayleigh p-value as
about 3.6e-5 (the PDF rounds that p-value to 4e-5). Assertions use that
published precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "matrix_profile" / "replicate_report"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.matrix_profile import mpxForPan  # noqa: E402
from analysis.matrix_profile import mpx_fast  # noqa: E402
from report_193500 import build_report  # noqa: E402


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


def test_coarse_scan_2_vs_4_minutes(report):
    pair = report["coarse_2_vs_4"]
    assert (pair["n_a"], pair["n_b"]) == (125, 66)
    assert round(pair["ks"], 2) == 0.62
    assert round(pair["w_min"], 1) == 4.5


def test_fine_scan_6_vs_7_minutes(report):
    pair = report["fine_6_vs_7"]
    assert (pair["n_a"], pair["n_b"]) == (54, 53)
    assert round(pair["ks"], 2) == 0.06
    assert round(pair["w_min"], 1) == 0.2
    assert (pair["shared"], pair["n_a"]) == (51, 54)


def test_phrase_groups_g1_and_g7(report):
    assert report["n_phrase_groups"] == 29
    assert report["n_kept"] == 7
    labels = [g["label"] for g in report["groups"]]
    assert labels == ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    g1, g7 = report["groups"][0], report["groups"][6]
    assert g1["occurrences"] == 30
    assert round(g1["circular_mean"], 2) == 0.73
    assert round(g1["concentration"], 2) == 0.62
    assert g7["occurrences"] == 12
    assert round(g7["circular_mean"], 2) == 0.33


def test_consensus_linear_order(report):
    assert report["consensus_order"] == ["G7", "G6", "G3", "G1", "G5", "G2", "G4"]
    assert (report["songs_agree"], report["songs_compared"]) == (27, 42)


def test_energy_minima(report):
    minima = report["minima"]
    assert minima["n_minima"] == 29
    assert minima["n_inside"] == 28
    assert round(minima["median_interval_min"], 1) == 16.9
    assert round(minima["circular_mean"], 2) == 0.35
    assert minima["rayleigh_p"] == pytest.approx(3.6e-5, rel=0.25)
