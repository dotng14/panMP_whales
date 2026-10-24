"""Tests for the subspace (top-k PCA) matching pursuit prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.mp import (  # noqa: E402
    residual_energy,
    run_mp_loop,
    single_template_dictionary,
)
from analysis.dict_learning.code.core import Matching  # noqa: E402
from analysis.dict_learning.code.mp import (  # noqa: E402
    SubspaceAtom,
    run_subspace_mp_loop,
    subspace_mp_step,
)
from analysis.dict_learning.code.tools import pca_basis  # noqa: E402


def _ortho_pair(freq: int, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    b1 = rng.standard_normal((freq, width))
    b1 /= np.linalg.norm(b1)
    b2 = rng.standard_normal((freq, width))
    b2 -= float(np.sum(b2 * b1)) * b1  # Gram-Schmidt against b1
    b2 /= np.linalg.norm(b2)
    return b1, b2


def _two_mode_signal(b1, b2, *, freq, width, seed=7):
    """Two windows, three well-separated events each = a1*b1 + a2*b2."""
    rng = np.random.default_rng(seed)
    locs = {"w0": [2, 10, 18], "w1": [3, 12, 20]}
    sig: dict[str, np.ndarray] = {}
    for wid, starts in locs.items():
        arr = np.zeros((freq, 28), dtype=np.float64)
        for t in starts:
            arr[:, t : t + width] += rng.uniform(1.0, 3.0) * b1
            arr[:, t : t + width] += rng.uniform(-1.5, 1.5) * b2
        sig[wid] = arr
    return sig


def test_pca_basis_orthonormal_and_descending():
    freq, width = 6, 5
    b1, b2 = _ortho_pair(freq, width, seed=1)
    rng = np.random.default_rng(2)
    matchings = []
    sig = {}
    for i in range(12):
        arr = np.zeros((freq, 25))
        t = 3
        # Two dominant modes + small full-rank noise so >=3 components are real.
        arr[:, t : t + width] = (
            rng.uniform(0.5, 2.0) * b1
            + rng.uniform(-1, 1) * 0.6 * b2
            + 0.05 * rng.standard_normal((freq, width))
        )
        sig[f"w{i}"] = arr
        matchings.append(Matching(0, f"w{i}", t, 1.0))

    basis, variances = pca_basis(sig, matchings, width=width, n_freq=freq, n_components=3)
    assert basis.shape == (3, freq, width)
    flat = basis.reshape(3, -1)
    gram = flat @ flat.T
    assert np.allclose(gram, np.eye(3), atol=1e-8)
    assert np.all(np.diff(variances) <= 1e-9)  # descending
    # k is clamped to the number of available patches.
    small, _ = pca_basis(sig, matchings[:2], width=width, n_freq=freq, n_components=5)
    assert small.shape[0] == 2


def test_k1_subspace_matches_standard_mp():
    freq, width = 5, 4
    b1, b2 = _ortho_pair(freq, width, seed=3)
    sig = _two_mode_signal(b1, b2, freq=freq, width=width)
    e0 = residual_energy(sig)

    std = run_mp_loop(
        sig,
        single_template_dictionary(np.ascontiguousarray(b1)),
        eps_step_fraction=0.01,
        disallow_overlap=True,
        mp_mode="global",
        show_progress=False,
    )
    atom = SubspaceAtom(template_id=0, basis=b1[None])
    sub = run_subspace_mp_loop(
        sig, [atom], eps_step_fraction=0.01, disallow_overlap=True, show_progress=False
    )

    # Same PC1 direction => identical placements and residual energy.
    assert len(sub.matchings) == len(std.matchings)
    assert {(m.window_id, m.t_start) for m in sub.matchings} == {
        (m.window_id, m.t_start) for m in std.matchings
    }
    assert np.isclose(
        residual_energy(sub.residual_stack) / e0,
        residual_energy(std.residual_stack) / e0,
        atol=1e-9,
    )


def test_k2_removes_at_least_k1_and_spans_signal():
    freq, width = 5, 4
    b1, b2 = _ortho_pair(freq, width, seed=5)
    sig = _two_mode_signal(b1, b2, freq=freq, width=width)
    e0 = residual_energy(sig)

    atom1 = SubspaceAtom(template_id=0, basis=b1[None])
    atom2 = SubspaceAtom(template_id=0, basis=np.stack([b1, b2]))
    r1 = run_subspace_mp_loop(
        sig, [atom1], eps_step_fraction=0.01, disallow_overlap=True, show_progress=False
    )
    r2 = run_subspace_mp_loop(
        sig, [atom2], eps_step_fraction=0.01, disallow_overlap=True, show_progress=False
    )

    e1 = residual_energy(r1.residual_stack)
    e2 = residual_energy(r2.residual_stack)
    assert e2 <= e1 + 1e-9  # k=2 subspace removes at least as much as k=1
    # The signal lives entirely in span(b1, b2): k=2 should nearly zero it.
    assert e2 / e0 < 1e-6


def test_deflated_patch_orthogonal_to_basis():
    freq, width = 5, 4
    b1, b2 = _ortho_pair(freq, width, seed=9)
    sig = _two_mode_signal(b1, b2, freq=freq, width=width)
    atom = SubspaceAtom(template_id=0, basis=np.stack([b1, b2]))

    resid = {k: np.array(v, copy=True) for k, v in sig.items()}
    step = subspace_mp_step(resid, [atom], occupied_intervals=None)
    assert step.matching is not None
    m = step.matching
    patch = resid[m.window_id][:, m.t_start : m.t_start + width]
    # After removing the projection, the residual patch has no basis component.
    assert abs(float(np.sum(patch * b1))) < 1e-9
    assert abs(float(np.sum(patch * b2))) < 1e-9
    # Energy removed equals the squared coefficient norm (orthonormal basis).
    assert np.isclose(step.delta_energy, float(np.dot(m.coeffs, m.coeffs)), atol=1e-9)


def test_disallow_overlap_footprints_do_not_intersect():
    freq, width = 5, 4
    b1, b2 = _ortho_pair(freq, width, seed=11)
    sig = _two_mode_signal(b1, b2, freq=freq, width=width)
    atom = SubspaceAtom(template_id=0, basis=np.stack([b1, b2]))
    res = run_subspace_mp_loop(
        sig, [atom], eps_step_fraction=0.01, disallow_overlap=True, show_progress=False
    )
    by_window: dict[str, list[tuple[int, int]]] = {}
    for m in res.matchings:
        by_window.setdefault(m.window_id, []).append((m.t_start, m.t_start + width))
    for intervals in by_window.values():
        intervals.sort()
        for (s0, e0), (s1, _e1) in zip(intervals, intervals[1:]):
            assert s1 >= e0  # no overlap between consecutive group footprints


def test_pc1_screen_matches_full_scoring():
    freq, width = 5, 4
    b1, b2 = _ortho_pair(freq, width, seed=13)
    sig = _two_mode_signal(b1, b2, freq=freq, width=width)
    atom = SubspaceAtom(template_id=0, basis=np.stack([b1, b2]))

    full = run_subspace_mp_loop(
        sig, [atom], eps_step_fraction=0.01, disallow_overlap=True, show_progress=False
    )
    screened = run_subspace_mp_loop(
        sig,
        [atom],
        eps_step_fraction=0.01,
        disallow_overlap=True,
        pc1_screen=True,
        show_progress=False,
    )
    assert {(m.window_id, m.t_start) for m in screened.matchings} == {
        (m.window_id, m.t_start) for m in full.matchings
    }
    assert np.isclose(
        residual_energy(screened.residual_stack),
        residual_energy(full.residual_stack),
        atol=1e-9,
    )


if __name__ == "__main__":
    test_pca_basis_orthonormal_and_descending()
    test_k1_subspace_matches_standard_mp()
    test_k2_removes_at_least_k1_and_spans_signal()
    test_deflated_patch_orthogonal_to_basis()
    test_disallow_overlap_footprints_do_not_intersect()
    test_pc1_screen_matches_full_scoring()
    print("all subspace_mp tests passed")
