"""Tests for mpxForPan and its accurate-sum helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mpxForPan import mpxForPan, muinvn, sq2s, sum2s


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def series(rng):
    return rng.standard_normal(400)


class TestSum2s:
    def test_matches_sliding_window_sum(self, series):
        w = 16
        got = sum2s(series, w)
        expected = np.convolve(series, np.ones(w), mode="valid")
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_length(self, series):
        w = 25
        assert sum2s(series, w).shape == (len(series) - w + 1,)


class TestSq2s:
    def test_matches_sum_of_squares(self, series):
        x = series[:50] - series[:50].mean()
        assert sq2s(x) == pytest.approx(np.sum(x * x), rel=1e-12, abs=1e-12)

    def test_single_element(self):
        assert sq2s(np.array([3.0])) == pytest.approx(9.0)


class TestMuinvn:
    def test_shapes(self, series):
        w = 32
        mu, sig = muinvn(series, w)
        assert mu.shape == (len(series) - w + 1,)
        assert sig.shape == mu.shape

    def test_mu_matches_moving_average(self, series):
        w = 20
        mu, _ = muinvn(series, w)
        expected = np.convolve(series, np.ones(w), mode="valid") / w
        np.testing.assert_allclose(mu, expected, rtol=1e-12, atol=1e-12)

    def test_sig_positive(self, series):
        _, sig = muinvn(series, 32)
        assert np.all(sig > 0)
        assert np.all(np.isfinite(sig))


class TestMpxForPan:
    def test_rejects_non_vector(self):
        with pytest.raises(ValueError, match="non vector"):
            mpxForPan(np.ones((10, 2)), 4, 8)

    def test_rejects_bad_parameters(self, series):
        with pytest.raises(ValueError, match="bad parameters"):
            mpxForPan(series, 3, 8)
        with pytest.raises(ValueError, match="bad parameters"):
            mpxForPan(series, 4, 3)

    def test_output_shapes_and_finiteness(self, series):
        w, minlag = 32, 10
        mp, mpi = mpxForPan(series, minlag, w)
        expected_len = len(series) - w + 1
        assert mp.shape == (expected_len,)
        assert mpi.shape == (expected_len,)
        assert np.all(np.isfinite(mp))
        assert np.all(mp >= 0)
        assert np.all((mpi >= 0) & (mpi < expected_len) | np.isnan(mpi))

    def test_profile_index_respects_minlag(self, series):
        w, minlag = 32, 20
        mp, mpi = mpxForPan(series, minlag, w)
        for i, j in enumerate(mpi):
            if np.isnan(j):
                continue
            j = int(j)
            assert abs(i - j) >= minlag

    def test_identical_motifs_have_near_zero_distance(self, rng):
        """Two identical subsequences far apart should yield a tiny MP value."""
        w = 32
        motif = rng.standard_normal(w)
        n = 300
        a = rng.standard_normal(n) * 0.01  # low noise background
        i0, i1 = 20, 150
        a[i0 : i0 + w] = motif
        a[i1 : i1 + w] = motif

        mp, mpi = mpxForPan(a, minlag=w, w=w)

        assert mp[i0] == pytest.approx(0.0, abs=1e-6)
        assert mp[i1] == pytest.approx(0.0, abs=1e-6)
        assert int(mpi[i0]) == i1
        assert int(mpi[i1]) == i0

    def test_self_join_symmetry_of_best_pair(self, rng):
        """Nearest-neighbor pair should be mutual for a clear planted motif."""
        w = 24
        motif = np.sin(np.linspace(0, 4 * np.pi, w))
        a = rng.standard_normal(200) * 0.05
        a[10:10 + w] = motif
        a[100:100 + w] = motif.copy()
        mp, mpi = mpxForPan(a, minlag=w, w=w)
        assert int(mpi[10]) == 100
        assert int(mpi[100]) == 10
        assert mp[10] < 1e-5
        assert mp[100] < 1e-5
