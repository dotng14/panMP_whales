"""Tests for response-correlation template similarity (fixed matching, no MP)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.core.objects import (  # noqa: E402
    Dictionary,
    Matching,
    Template,
)
from analysis.dict_learning.code.learn.calc_template_similarity import (  # noqa: E402
    calc_template_similarity,
)


def _unit(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    return arr / float(np.linalg.norm(arr))


def _dictionary(*arrays: np.ndarray, scale: float = 1.0) -> Dictionary:
    d = Dictionary()
    for tid, data in enumerate(arrays):
        d.add(Template(template_id=tid, data=scale * np.asarray(data, dtype=np.float64)))
    return d


def test_identical_atoms_similarity_one():
    pattern = np.zeros((2, 4), dtype=np.float64)
    pattern[0, :] = 1.0
    dictionary = _dictionary(pattern, pattern.copy())
    x = np.zeros((2, 40), dtype=np.float64)
    for t in (2, 12, 22, 30):
        x[0, t : t + 4] = float(t)
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=t, amplitude=1.0)
        for t in (2, 12, 22, 30)
    ]
    result = calc_template_similarity({"w0": x}, dictionary, matchings)
    assert result.similarity.shape == (2, 2)
    assert result.similarity[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert result.similarity[0, 0] == pytest.approx(1.0, abs=1e-9)


def test_orthogonal_atoms_near_zero_correlation():
    a = np.zeros((2, 4), dtype=np.float64)
    a[0, :] = 1.0
    b = np.zeros((2, 4), dtype=np.float64)
    b[1, :] = 1.0
    dictionary = _dictionary(a, b)
    rng = np.random.default_rng(4)
    x = np.zeros((2, 80), dtype=np.float64)
    starts = list(range(2, 70, 5))
    for t in starts:
        # Independent amplitudes on the two frequency rows → nearly uncorrelated
        # responses for orthogonal unit templates.
        x[0, t : t + 4] = rng.normal()
        x[1, t : t + 4] = rng.normal()
    matchings = [
        Matching(template_id=99, window_id="w0", t_start=t, amplitude=0.0)
        for t in starts
    ]
    result = calc_template_similarity({"w0": x}, dictionary, matchings)
    assert abs(result.similarity[0, 1]) < 0.45


def test_symmetry_and_unit_diagonal():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 5))
    b = rng.normal(size=(3, 5))
    dictionary = _dictionary(a, b)
    x = rng.normal(size=(3, 60))
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=t, amplitude=1.0)
        for t in range(0, 50, 5)
    ]
    result = calc_template_similarity({"w0": x}, dictionary, matchings)
    s = result.similarity
    assert s == pytest.approx(s.T, abs=1e-12)
    assert s[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert s[1, 1] == pytest.approx(1.0, abs=1e-9)


def test_template_scale_invariant():
    pattern = np.zeros((2, 4), dtype=np.float64)
    pattern[0, :] = 1.0
    other = np.zeros((2, 4), dtype=np.float64)
    other[1, :] = 1.0
    d1 = _dictionary(pattern, other, scale=1.0)
    d2 = _dictionary(pattern, other, scale=7.5)
    x = np.zeros((2, 40), dtype=np.float64)
    for i, t in enumerate((1, 8, 16, 24, 32)):
        x[0, t : t + 4] += 1.0 + 0.4 * i
        x[1, t : t + 4] += 0.5 * (i + 1)
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=t, amplitude=1.0)
        for t in (1, 8, 16, 24, 32)
    ]
    r1 = calc_template_similarity({"w0": x}, d1, matchings)
    r2 = calc_template_similarity({"w0": x}, d2, matchings)
    assert np.all(np.isfinite(r1.similarity))
    assert r1.similarity == pytest.approx(r2.similarity, abs=1e-9)


def test_multi_width_pad():
    short = np.zeros((2, 3), dtype=np.float64)
    short[0, :] = 1.0
    wide = np.zeros((2, 6), dtype=np.float64)
    wide[0, 1:4] = 1.0  # same energy mass, broader canvas
    dictionary = _dictionary(short, wide)
    x = np.zeros((2, 40), dtype=np.float64)
    for t in (2, 12, 22, 30):
        x[0, t : t + 6] = 1.0 + 0.1 * t
    matchings = [
        Matching(template_id=1, window_id="w0", t_start=t, amplitude=1.0)
        for t in (2, 12, 22, 30)
    ]
    result = calc_template_similarity({"w0": x}, dictionary, matchings)
    assert result.params["common_width"] == 6
    assert result.similarity[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert result.similarity[1, 1] == pytest.approx(1.0, abs=1e-9)
    assert np.isfinite(result.similarity[0, 1])


def test_matching_template_id_ignored():
    a = np.zeros((2, 4), dtype=np.float64)
    a[0, :] = 1.0
    b = np.zeros((2, 4), dtype=np.float64)
    b[1, :] = 1.0
    dictionary = _dictionary(a, b)
    x = np.zeros((2, 40), dtype=np.float64)
    for i, t in enumerate((2, 10, 18, 26)):
        x[0, t : t + 4] = 1.0 + i
        x[1, t : t + 4] = 2.0 - 0.2 * i
    m_a = [
        Matching(template_id=0, window_id="w0", t_start=t, amplitude=1.0)
        for t in (2, 10, 18, 26)
    ]
    m_b = [
        Matching(template_id=1, window_id="w0", t_start=t, amplitude=9.0)
        for t in (2, 10, 18, 26)
    ]
    r_a = calc_template_similarity({"w0": x}, dictionary, m_a)
    r_b = calc_template_similarity({"w0": x}, dictionary, m_b)
    assert r_a.similarity == pytest.approx(r_b.similarity, abs=1e-12)
    assert r_a.alphas == pytest.approx(r_b.alphas, abs=1e-12)


def test_empty_matchings_raises():
    d = _dictionary(np.ones((1, 2)))
    with pytest.raises(ValueError, match="matchings"):
        calc_template_similarity({"w0": np.ones((1, 10))}, d, [])


def test_subspace_raises():
    basis = np.zeros((2, 2, 3), dtype=np.float64)
    basis[0, 0, :] = 0.5
    basis[1, 1, :] = 0.5
    d = Dictionary()
    d.add(Template(template_id=0, data=basis))
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=0, amplitude=1.0)
        for _ in range(3)
    ]
    with pytest.raises(ValueError, match="rank-1"):
        calc_template_similarity(
            {"w0": np.ones((2, 20))}, d, matchings
        )


def test_alphas_shape_and_n_patches():
    pattern = np.ones((2, 3), dtype=np.float64)
    d = _dictionary(pattern, pattern * 0 + np.array([[1, 0, 0], [0, 1, 0.5]]))
    # Make second template differ
    d = Dictionary()
    d.add(Template(template_id=0, data=_unit(np.ones((2, 3)))))
    other = np.zeros((2, 3))
    other[1, :] = 1.0
    d.add(Template(template_id=1, data=_unit(other)))
    x = np.random.default_rng(1).normal(size=(2, 30))
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=t, amplitude=1.0)
        for t in (0, 5, 10, 15, 20)
    ]
    result = calc_template_similarity({"w0": x}, d, matchings)
    assert result.n_patches == 5
    assert result.alphas.shape == (2, 5)
