"""Tests for patch↔template residual-energy distance."""

from __future__ import annotations


import numpy as np


from analysis.dict_learning.code.core import Template  # noqa: E402
from analysis.dict_learning.code.core import patch_template_distance  # noqa: E402


def test_rank1_distance_exact_fit():
    v = np.zeros((3, 4), dtype=np.float64)
    v[0, :] = 1.0
    v /= np.linalg.norm(v)
    a = 2.5
    patch = a * v
    d, e_x, e_fit = patch_template_distance(patch, v, return_parts=True)
    assert d == 0.0 or abs(d) < 1e-12
    assert abs(e_fit - e_x) < 1e-12


def test_rank1_distance_orthogonal_component():
    v = np.zeros((2, 3), dtype=np.float64)
    v[0, :] = 1.0
    v /= np.linalg.norm(v)
    orth = np.zeros_like(v)
    orth[1, :] = 1.0
    orth /= np.linalg.norm(orth)
    patch = 3.0 * v + 4.0 * orth
    d, e_x, e_fit = patch_template_distance(patch, v, return_parts=True)
    assert abs(e_x - (9.0 + 16.0)) < 1e-10
    assert abs(e_fit - 9.0) < 1e-10
    assert abs(d - 16.0) < 1e-10


def test_subspace_distance():
    # Two orthonormal modes in R^{2x2} flattened to dim 4.
    v1 = np.zeros((2, 2), dtype=np.float64)
    v1[0, 0] = 1.0
    v2 = np.zeros((2, 2), dtype=np.float64)
    v2[1, 1] = 1.0
    basis = np.stack([v1, v2], axis=0)
    patch = 2.0 * v1 + 3.0 * v2
    # Extra component orthogonal to both
    extra = np.zeros((2, 2), dtype=np.float64)
    extra[0, 1] = 5.0
    patch = patch + extra
    d, e_x, e_fit = patch_template_distance(patch, basis, return_parts=True)
    assert abs(e_fit - (4.0 + 9.0)) < 1e-10
    assert abs(d - 25.0) < 1e-10
    assert abs(e_x - (4.0 + 9.0 + 25.0)) < 1e-10


def test_template_object():
    v = np.ones((2, 3), dtype=np.float64)
    v /= np.linalg.norm(v)
    tmpl = Template(template_id=0, data=v)
    patch = 1.5 * v
    assert patch_template_distance(patch, tmpl) < 1e-12


if __name__ == "__main__":
    test_rank1_distance_exact_fit()
    test_rank1_distance_orthogonal_component()
    test_subspace_distance()
    test_template_object()
    print("ok")
