"""``frequency_pca``, ``gpl_rows`` and ``decimate_columns`` on hand-built spectrograms."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.matrix_profile.signals import decimate_columns, frequency_pca, gpl_rows


def _blocks(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    direction = np.zeros(6)
    direction[[1, 2]] = [-0.6, -0.8]  # largest entry negative: the sign must flip
    return [np.outer(direction, 5.0 * rng.standard_normal(n)) + 0.1 * rng.standard_normal((6, n))
            for n in (40, 60)]


def test_frequency_pca_finds_the_dominant_pattern_with_a_fixed_sign():
    basis, mean = frequency_pca(_blocks(), 3)
    assert basis.shape == (3, 6) and mean.shape == (6,)
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(basis[0], [0, 0.6, 0.8, 0, 0, 0], atol=0.02)
    assert all(row[np.argmax(np.abs(row))] > 0 for row in basis)


@pytest.mark.parametrize("n_rows", [0, 7])
def test_frequency_pca_rejects_row_counts(n_rows):
    with pytest.raises(ValueError, match="between 1 and 6"):
        frequency_pca(_blocks(), n_rows)


def test_gpl_rows_kinds():
    blocks = _blocks()
    kw = {"start_freq": 10.0, "end_freq": 60.0}
    energy, info = gpl_rows(blocks, "energy", **kw)
    np.testing.assert_allclose(energy[1], blocks[1].sum(axis=0, keepdims=True))
    assert info["band_edges"] is None
    full, _ = gpl_rows(blocks, "full", **kw)
    assert full[0] is blocks[0]
    pca, info = gpl_rows(blocks, "pca", pca_rows=2, **kw)
    assert [p.shape for p in pca] == [(2, 40), (2, 60)] and info["pca_rows"] == 2
    bands, info = gpl_rows(blocks, "bands", band_edges=[10, 30, 60], **kw)
    np.testing.assert_allclose(bands[0][0], blocks[0][:2].sum(axis=0))  # rows at 10 and 20 Hz
    assert info["band_edges"] == [10, 30, 60]
    with pytest.raises(ValueError, match="GPL signal must be one of"):
        gpl_rows(blocks, "rms", **kw)
    with pytest.raises(ValueError, match="different frequency rows"):
        gpl_rows([blocks[0], blocks[1][:5]], "energy", **kw)


def test_decimate_columns_averages_whole_groups():
    rows = np.arange(14, dtype=float).reshape(2, 7)
    np.testing.assert_allclose(decimate_columns(rows, 3), [[1, 4], [8, 11]])
    assert decimate_columns(rows, 1) is not None and decimate_columns(rows, 1).shape == (2, 7)
    with pytest.raises(ValueError, match=">= 1"):
        decimate_columns(rows, 0)
    with pytest.raises(ValueError, match="longer than the signal"):
        decimate_columns(rows, 8)
