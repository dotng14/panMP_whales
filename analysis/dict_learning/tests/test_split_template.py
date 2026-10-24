"""Tests for randomized ±ε 1→2 template patch bipartition."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.core import Matching, Template, Window  # noqa: E402
from analysis.dict_learning.code.learn.split_template import (  # noqa: E402
    SplitOneResult,
    SplitResult,
    split_one_template,
    split_patches_rank1,
    split_template_matchings,
)


def _pattern_a(f: int = 4, w: int = 5) -> np.ndarray:
    p = np.zeros((f, w), dtype=np.float64)
    p[0, :] = 3.0
    p[1, :] = -3.0
    return p


def _pattern_b(f: int = 4, w: int = 5) -> np.ndarray:
    p = np.zeros((f, w), dtype=np.float64)
    p[2, :] = 2.0
    p[3, :] = -2.0
    return p


def test_split_recovers_two_planted_modes():
    a = _pattern_a()
    b = _pattern_b()
    patches = [a.copy() for _ in range(6)] + [b.copy() for _ in range(5)]
    rng = np.random.default_rng(0)
    patches = [p + 0.05 * rng.standard_normal(p.shape) for p in patches]

    # Base word ≈ mixture PC / mean — ±ε + refine should recover the modes.
    result = split_patches_rank1(
        patches,
        base_template=np.mean(patches, axis=0),
        epsilon=0.25,
        n_trials=20,
        min_relative_reduction=0.05,
        min_size=2,
        rng=1,
    )
    assert isinstance(result, SplitResult)
    assert result.accepted
    assert result.n_trials == 20
    assert result.best_trial >= 0
    assert result.n_a + result.n_b == len(patches)
    assert result.n_a >= 2 and result.n_b >= 2
    assert result.sse_two < result.sse_one
    assert result.relative_reduction >= 0.05

    labels = result.labels
    left = set(labels[:6].tolist())
    right = set(labels[6:].tolist())
    assert len(left) == 1 and len(right) == 1 and left != right

    tmpl_for_a = result.template_a if labels[0] == 0 else result.template_b
    tmpl_for_b = result.template_b if labels[0] == 0 else result.template_a
    for got, ref in ((tmpl_for_a, a), (tmpl_for_b, b)):
        corr = float(
            np.dot(got.ravel(), ref.ravel())
            / (np.linalg.norm(got) * np.linalg.norm(ref))
        )
        assert abs(corr) > 0.95


def test_unimodal_patches_reject_split():
    a = _pattern_a()
    patches = [a.copy() for _ in range(10)]
    result = split_patches_rank1(
        patches,
        base_template=a,
        epsilon=0.1,
        n_trials=10,
        min_relative_reduction=0.1,
        min_size=2,
        rng=0,
    )
    assert not result.accepted
    assert result.n_b == 0
    assert result.sse_one < 1e-9
    assert result.relative_reduction < 0.1


def test_tiny_n_returns_no_split():
    a = _pattern_a()
    b = _pattern_b()
    result = split_patches_rank1([a, b], min_size=2, n_trials=10, rng=0)
    assert not result.accepted
    assert result.n_a == 2 and result.n_b == 0


def test_empty_patches_raises():
    with pytest.raises(ValueError, match="non-empty"):
        split_patches_rank1([])


def test_mismatched_shapes_raise():
    a = _pattern_a(w=5)
    b = _pattern_a(w=4)
    with pytest.raises(ValueError, match="shape"):
        split_patches_rank1([a, a, b])


def test_invalid_n_trials_raises():
    with pytest.raises(ValueError, match="n_trials"):
        split_patches_rank1([_pattern_a()] * 4, n_trials=0)


def test_split_one_template_returns_unit_templates_and_patches():
    a = _pattern_a()
    b = _pattern_b()
    rng = np.random.default_rng(0)
    patches = [a + 0.05 * rng.standard_normal(a.shape) for _ in range(6)]
    patches += [b + 0.05 * rng.standard_normal(b.shape) for _ in range(5)]

    result = split_one_template(
        patches,
        base_template=np.mean(patches, axis=0),
        n_trials=20,
        n_refine=5,
        epsilon=0.25,
        min_size=2,
        rng=1,
    )
    assert isinstance(result, SplitOneResult)
    assert abs(float(np.linalg.norm(result.template_a)) - 1.0) < 1e-6
    assert abs(float(np.linalg.norm(result.template_b)) - 1.0) < 1e-6
    assert len(result.patches_a) + len(result.patches_b) == len(patches)
    assert len(result.patches_a) >= 2 and len(result.patches_b) >= 2
    assert result.n_trials == 20
    assert result.n_refine == 5
    assert result.sse_two <= result.sse_one + 1e-9


def test_split_one_template_too_few_raises():
    with pytest.raises(ValueError, match="at least"):
        split_one_template([_pattern_a(), _pattern_b()], min_size=2, n_trials=5)


def test_split_template_matchings_wrapper():
    f, t = 4, 80
    a = _pattern_a()
    b = _pattern_b()
    data = np.zeros((f, t), dtype=np.float64)
    starts_a = (5, 20, 35)
    starts_b = (50, 60, 70)
    for t0 in starts_a:
        data[:, t0 : t0 + 5] += a
    for t0 in starts_b:
        data[:, t0 : t0 + 5] += b
    window = Window(
        window_id="w0",
        data=data,
        start_sample=1,
        end_sample=1 + t * 10 - 1,
        overlap=10,
    )
    # Shared base word is a mixture-like seed; ±ε refine should separate modes.
    template = Template(template_id=0, data=(a + b) / 2.0)
    matchings = [
        *(Matching(0, "w0", t0, 1.0) for t0 in starts_a),
        *(Matching(0, "w0", t0, 1.0) for t0 in starts_b),
    ]
    result, ma, mb = split_template_matchings(
        [window],
        template,
        matchings,
        epsilon=0.3,
        n_trials=20,
        min_relative_reduction=0.05,
        min_size=2,
        rng=2,
    )
    assert result.accepted
    assert len(ma) + len(mb) == len(matchings)
    assert {m.t_start for m in ma} | {m.t_start for m in mb} == set(starts_a + starts_b)
    starts_group = (
        {m.t_start for m in ma}
        if starts_a[0] in {m.t_start for m in ma}
        else {m.t_start for m in mb}
    )
    assert set(starts_a).issubset(starts_group) or set(starts_b).issubset(starts_group)


if __name__ == "__main__":
    test_split_recovers_two_planted_modes()
    test_unimodal_patches_reject_split()
    test_tiny_n_returns_no_split()
    test_empty_patches_raises()
    test_mismatched_shapes_raise()
    test_invalid_n_trials_raises()
    test_split_one_template_returns_unit_templates_and_patches()
    test_split_one_template_too_few_raises()
    test_split_template_matchings_wrapper()
    print("test_split_template: OK")
