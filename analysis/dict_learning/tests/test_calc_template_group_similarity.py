"""Tests for group-mean template similarity (labeled matching, no MP)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.core import (  # noqa: E402
    Dictionary,
    Matching,
    Template,
)
from analysis.dict_learning.code.learn.similarity import (  # noqa: E402
    calc_template_group_similarity,
)


def _unit(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    return arr / float(np.linalg.norm(arr))


def _dictionary(*arrays: np.ndarray) -> Dictionary:
    d = Dictionary()
    for tid, data in enumerate(arrays):
        d.add(Template(template_id=tid, data=_unit(data)))
    return d


def test_identical_atoms_cross_similarity_one():
    pattern = np.zeros((2, 4), dtype=np.float64)
    pattern[0, :] = 1.0
    dictionary = _dictionary(pattern, pattern.copy())
    x = np.zeros((2, 40), dtype=np.float64)
    for t in (2, 10, 18, 26):
        x[0, t : t + 4] = 2.0 + 0.1 * t
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=2, amplitude=1.0),
        Matching(template_id=0, window_id="w0", t_start=10, amplitude=1.0),
        Matching(template_id=1, window_id="w0", t_start=18, amplitude=1.0),
        Matching(template_id=1, window_id="w0", t_start=26, amplitude=1.0),
    ]
    result = calc_template_group_similarity({"w0": x}, dictionary, matchings)
    assert result.similarity[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert result.similarity[1, 0] == pytest.approx(1.0, abs=1e-9)
    assert result.similarity[0, 0] == pytest.approx(1.0, abs=1e-9)


def test_disjoint_support_cross_near_zero():
    a = np.zeros((2, 4), dtype=np.float64)
    a[0, :] = 1.0
    b = np.zeros((2, 4), dtype=np.float64)
    b[1, :] = 1.0
    dictionary = _dictionary(a, b)
    x = np.zeros((2, 40), dtype=np.float64)
    for t in (2, 10):
        x[0, t : t + 4] = 3.0  # pure A pattern for G_A
    for t in (18, 26):
        x[1, t : t + 4] = 4.0  # pure B pattern for G_B
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=2, amplitude=1.0),
        Matching(template_id=0, window_id="w0", t_start=10, amplitude=1.0),
        Matching(template_id=1, window_id="w0", t_start=18, amplitude=1.0),
        Matching(template_id=1, window_id="w0", t_start=26, amplitude=1.0),
    ]
    result = calc_template_group_similarity({"w0": x}, dictionary, matchings)
    assert result.similarity[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert result.similarity[1, 1] == pytest.approx(1.0, abs=1e-9)
    assert abs(result.similarity[0, 1]) < 1e-9
    assert abs(result.similarity[1, 0]) < 1e-9
    assert result.group_sizes.tolist() == [2, 2]


def test_asymmetric():
    """B's patches look like A, but A's patches do not look like B."""
    a = np.zeros((2, 4), dtype=np.float64)
    a[0, :] = 1.0
    b = np.zeros((2, 4), dtype=np.float64)
    b[1, :] = 1.0
    dictionary = _dictionary(a, b)
    x = np.zeros((2, 40), dtype=np.float64)
    # G_A: pure A (B scores ~0)
    for t in (2, 8):
        x[0, t : t + 4] = 3.0
    # G_B: pure A as well (mislabel-ish) so B self is low but A scores high
    # Need mu(B,B) significant for S_BB — pure B on G_B:
    for t in (16, 24):
        x[1, t : t + 4] = 5.0
        x[0, t : t + 4] = 5.0  # both rows → both A and B score
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=2, amplitude=1.0),
        Matching(template_id=0, window_id="w0", t_start=8, amplitude=1.0),
        Matching(template_id=1, window_id="w0", t_start=16, amplitude=1.0),
        Matching(template_id=1, window_id="w0", t_start=24, amplitude=1.0),
    ]
    result = calc_template_group_similarity({"w0": x}, dictionary, matchings)
    # On G_A (pure row0): S[0,1] ≈ 0, S[0,0]=1
    assert abs(result.similarity[0, 1]) < 0.1
    # On G_B (both rows): S[1,0] is mu(B,A)/mu(B,B) = positive and can be ~1
    assert result.similarity[1, 0] == pytest.approx(1.0, abs=0.05)
    assert result.similarity[0, 1] != pytest.approx(result.similarity[1, 0], abs=0.2)


def test_empty_group_row_nan():
    pattern = np.zeros((2, 4), dtype=np.float64)
    pattern[0, :] = 1.0
    other = np.zeros((2, 4), dtype=np.float64)
    other[1, :] = 1.0
    dictionary = _dictionary(pattern, other)
    x = np.zeros((2, 30), dtype=np.float64)
    for t in (2, 10, 18):
        x[0, t : t + 4] = 2.0
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=t, amplitude=1.0)
        for t in (2, 10, 18)
    ]
    result = calc_template_group_similarity({"w0": x}, dictionary, matchings)
    assert result.group_sizes[0] == 3
    assert result.group_sizes[1] == 0
    assert np.all(np.isnan(result.similarity[1]))
    assert result.similarity[0, 0] == pytest.approx(1.0, abs=1e-9)


def test_label_assignment_changes_result():
    a = np.zeros((2, 4), dtype=np.float64)
    a[0, :] = 1.0
    b = np.zeros((2, 4), dtype=np.float64)
    b[1, :] = 1.0
    dictionary = _dictionary(a, b)
    x = np.zeros((2, 40), dtype=np.float64)
    for t in (2, 10):
        x[0, t : t + 4] = 3.0
    for t in (18, 26):
        x[1, t : t + 4] = 4.0
    # Labels agree with content
    m_aligned = [
        Matching(template_id=0, window_id="w0", t_start=2, amplitude=1.0),
        Matching(template_id=0, window_id="w0", t_start=10, amplitude=1.0),
        Matching(template_id=1, window_id="w0", t_start=18, amplitude=1.0),
        Matching(template_id=1, window_id="w0", t_start=26, amplitude=1.0),
    ]
    # Same locations, swapped labels
    m_swapped = [
        Matching(template_id=1, window_id="w0", t_start=2, amplitude=1.0),
        Matching(template_id=1, window_id="w0", t_start=10, amplitude=1.0),
        Matching(template_id=0, window_id="w0", t_start=18, amplitude=1.0),
        Matching(template_id=0, window_id="w0", t_start=26, amplitude=1.0),
    ]
    r0 = calc_template_group_similarity({"w0": x}, dictionary, m_aligned)
    r1 = calc_template_group_similarity({"w0": x}, dictionary, m_swapped)
    # Swapped labels scramble which group owns pure-A vs pure-B patches.
    assert not np.allclose(r0.mean_alpha, r1.mean_alpha, equal_nan=True)


def test_empty_matchings_raises():
    d = _dictionary(np.ones((1, 2)))
    with pytest.raises(ValueError, match="matchings"):
        calc_template_group_similarity({"w0": np.ones((1, 10))}, d, [])


def test_subspace_raises():
    basis = np.zeros((2, 2, 3), dtype=np.float64)
    basis[0, 0, :] = 0.5
    basis[1, 1, :] = 0.5
    d = Dictionary()
    d.add(Template(template_id=0, data=basis))
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=0, amplitude=1.0)
        for _ in range(2)
    ]
    with pytest.raises(ValueError, match="rank-1"):
        calc_template_group_similarity({"w0": np.ones((2, 20))}, d, matchings)
