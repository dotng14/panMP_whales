"""The repeats viewer's average panel for rank-1 and subspace atoms."""

from __future__ import annotations

import numpy as np

from analysis.dict_learning.code.catalog.repeats_from_catalog import _average_raster, chunk_distance_norm


def test_subspace_atom_shows_its_first_component() -> None:
    rng = np.random.default_rng(0)
    basis = rng.random((5, 6, 4))
    raster = _average_raster(basis)
    assert (raster["rows"], raster["cols"]) == (6, 4)
    assert raster == _average_raster(basis[0])


def test_subspace_distance_uses_every_component() -> None:
    basis = np.zeros((2, 3, 2))
    basis[0, 0, 0] = 1.0
    basis[1, 1, 1] = 1.0
    patch = basis[0] + basis[1]
    assert chunk_distance_norm(patch, basis) == 0.0
    assert chunk_distance_norm(patch, basis[:1]) == 0.5
