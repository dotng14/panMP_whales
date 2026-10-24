"""Tests for panMP surface construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.matrix_profile import _zscore, panMP  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestZscore:
    def test_mean_zero_unit_sample_std(self, rng):
        x = rng.standard_normal(100) * 3 + 5
        z = _zscore(x)
        assert np.mean(z) == pytest.approx(0.0, abs=1e-12)
        assert np.std(z, ddof=1) == pytest.approx(1.0, abs=1e-12)

    def test_constant_series(self):
        z = _zscore(np.ones(10))
        np.testing.assert_array_equal(z, np.zeros(10))


class TestPanMP:
    def test_rejects_too_short_series(self, rng):
        with pytest.raises(ValueError, match="too short"):
            panMP(rng.standard_normal(20), plot=False, MinMotifLen=8, MaxMotifLen=16)

    def test_rejects_bad_motif_bounds(self, rng):
        ts = rng.standard_normal(100)
        with pytest.raises(ValueError, match="MinMotifLen"):
            panMP(ts, plot=False, MinMotifLen=2, MaxMotifLen=16)
        with pytest.raises(ValueError, match="MaxMotifLen"):
            panMP(ts, plot=False, MinMotifLen=16, MaxMotifLen=8)

    def test_output_shape(self, rng):
        # MaxMotifLen is not on the step grid (like MATLAB's 32:64:512), so the
        # last computed length is 24 and MATLAB's pan ends at column 24+8-1.
        MinMotifLen, MaxMotifLen, StepSize = 8, 28, 8
        n = MaxMotifLen + MinMotifLen + 40  # enough headroom after trim
        ts = rng.standard_normal(n)
        pan = panMP(
            ts,
            plot=False,
            MinMotifLen=MinMotifLen,
            MaxMotifLen=MaxMotifLen,
            StepSize=StepSize,
        )
        expected_rows = (n - MinMotifLen + 1) - MaxMotifLen - 1
        expected_cols = 24 + StepSize - 1
        assert pan.shape == (expected_rows, expected_cols)

    def test_short_motif_columns_are_nan(self, rng):
        MinMotifLen, MaxMotifLen, StepSize = 8, 24, 8
        n = MaxMotifLen + MinMotifLen + 40
        pan = panMP(
            rng.standard_normal(n),
            plot=False,
            MinMotifLen=MinMotifLen,
            MaxMotifLen=MaxMotifLen,
            StepSize=StepSize,
        )
        assert np.all(np.isnan(pan[:, : MinMotifLen - 1]))

    def test_normalized_peak_is_one(self, rng):
        MinMotifLen, MaxMotifLen, StepSize = 8, 24, 8
        n = MaxMotifLen + MinMotifLen + 40
        pan = panMP(
            rng.standard_normal(n),
            plot=False,
            MinMotifLen=MinMotifLen,
            MaxMotifLen=MaxMotifLen,
            StepSize=StepSize,
        )
        assert np.nanmax(pan) == pytest.approx(1.0)

    def test_filled_motif_bands_are_finite(self, rng):
        MinMotifLen, MaxMotifLen, StepSize = 8, 24, 8
        n = MaxMotifLen + MinMotifLen + 40
        pan = panMP(
            rng.standard_normal(n),
            plot=False,
            MinMotifLen=MinMotifLen,
            MaxMotifLen=MaxMotifLen,
            StepSize=StepSize,
        )
        # Each computed motif length i fills columns [i, i+StepSize)
        for i in range(MinMotifLen, MaxMotifLen + 1, StepSize):
            band = pan[:, i - 1 : i - 1 + StepSize]
            assert np.any(np.isfinite(band)), f"no finite values for motif length {i}"

    def test_planted_motif_produces_high_pan_values(self, rng):
        """Repeated motifs should produce elevated pan-MP peaks after MotifPlotMode."""
        MinMotifLen, MaxMotifLen, StepSize = 8, 16, 8
        w = MinMotifLen
        motif = np.sin(np.linspace(0, 4 * np.pi, w))
        n = MaxMotifLen + MinMotifLen + 80
        ts = rng.standard_normal(n) * 0.05
        ts[30 : 30 + w] = motif
        ts[90 : 90 + w] = motif.copy()

        pan = panMP(
            ts,
            plot=False,
            MinMotifLen=MinMotifLen,
            MaxMotifLen=MaxMotifLen,
            StepSize=StepSize,
            MotifPlotMode=True,
        )
        # Motif band columns should contain strong peaks (near 1 after normalize)
        band = pan[:, MinMotifLen - 1 : MinMotifLen - 1 + StepSize]
        assert np.nanmax(band) == pytest.approx(1.0, abs=1e-6)

    def test_accepts_row_or_column_vector(self, rng):
        MinMotifLen, MaxMotifLen, StepSize = 8, 16, 8
        n = MaxMotifLen + MinMotifLen + 40
        row = rng.standard_normal(n)
        col = row.reshape(-1, 1)
        a = panMP(
            row, plot=False,
            MinMotifLen=MinMotifLen, MaxMotifLen=MaxMotifLen, StepSize=StepSize,
        )
        b = panMP(
            col, plot=False,
            MinMotifLen=MinMotifLen, MaxMotifLen=MaxMotifLen, StepSize=StepSize,
        )
        np.testing.assert_allclose(a, b, equal_nan=True)
