"""Tests for asymmetric atom divergences (calc_divergences)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.core.objects import Dictionary, Template, Window  # noqa: E402
from analysis.dict_learning.code.learn.calc_divergences import (  # noqa: E402
    calc_divergences,
)
from analysis.dict_learning.code.mp.mp_step import (  # noqa: E402
    inflate_matching,
    residual_energy,
    _deflate_matching,
)
from analysis.dict_learning.code.core.objects import Matching  # noqa: E402


def _make_window(data: np.ndarray, wid: str = "w0") -> Window:
    cols = int(data.shape[1])
    return Window(
        window_id=wid,
        data=np.asarray(data, dtype=np.float64),
        start_sample=1,
        end_sample=1 + cols * 10 - 1,
        overlap=10,
    )


def _unit_template(tid: int, data: np.ndarray) -> Template:
    arr = np.asarray(data, dtype=np.float64)
    nrm = float(np.linalg.norm(arr))
    if nrm <= 0.0:
        raise ValueError("zero template")
    return Template(template_id=tid, data=arr / nrm)


def _pattern_a(f: int = 6, w: int = 5) -> np.ndarray:
    p = np.zeros((f, w), dtype=np.float64)
    p[0, :] = 3.0
    p[1, :] = -3.0
    return p


def _pattern_b(f: int = 6, w: int = 5) -> np.ndarray:
    """Frequency-orthogonal to pattern A (rows 2–3)."""
    p = np.zeros((f, w), dtype=np.float64)
    p[2, :] = 2.0
    p[3, :] = -2.0
    return p


def test_inflate_is_inverse_of_deflate():
    basis = np.zeros((1, 4, 3), dtype=np.float64)
    basis[0, 0, :] = 1.0 / np.sqrt(3.0)
    resid = np.zeros((4, 20), dtype=np.float64)
    resid[0, 5:8] = 2.0
    before = resid.copy()
    m = Matching(template_id=0, window_id="w", t_start=5, amplitude=2.0 * np.sqrt(3.0))
    assert _deflate_matching(resid, m, basis)
    assert residual_energy({"w": resid}) < residual_energy({"w": before})
    assert inflate_matching(resid, m, basis)
    assert np.allclose(resid, before)


def test_shape_and_template_ids():
    pa = _pattern_a()
    data = np.zeros((6, 80), dtype=np.float64)
    for t0 in (5, 25, 45):
        data[:, t0 : t0 + pa.shape[1]] += pa
    window = _make_window(data)
    dictionary = Dictionary()
    dictionary.add(_unit_template(0, pa))
    dictionary.add(_unit_template(1, pa.copy()))

    result = calc_divergences(
        [window],
        dictionary,
        eps_step_fraction=1e-12,
        min_delta_fraction=1e-12,
        show_progress=False,
    )
    assert result.template_ids == [0, 1]
    assert result.divergences.shape == (2, 2)
    assert result.significant.shape == (2,)
    assert result.energy_without.shape == (2,)
    assert result.delta_energy.shape == (2,)


def test_substitute_near_identical_atoms():
    """Two equal atoms: joint MP peels via one id; the other substitutes that row."""
    pa = _pattern_a()
    data = np.zeros((6, 100), dtype=np.float64)
    for t0 in (5, 25, 45, 65):
        data[:, t0 : t0 + pa.shape[1]] += pa
    window = _make_window(data)
    dictionary = Dictionary()
    dictionary.add(_unit_template(1, pa))
    dictionary.add(_unit_template(2, pa.copy()))

    result = calc_divergences(
        [window],
        dictionary,
        eps_step_fraction=1e-12,
        min_delta_fraction=1e-12,
        show_progress=False,
    )
    # Greedy joint MP assigns all placements to one of the twins; the unused twin
    # has ΔA ≈ 0 and is insignificant.
    assert result.significant.any()
    sig_rows = np.where(result.significant)[0]
    assert len(sig_rows) >= 1
    for i in sig_rows:
        # Same atom re-matches ≈ 0; the twin substitutes ≈ 0 as well.
        assert float(result.divergences[i, 0]) < 0.25
        assert float(result.divergences[i, 1]) < 0.25
    for i in range(2):
        if not bool(result.significant[i]):
            assert np.all(np.isnan(result.divergences[i, :]))


def test_orthogonal_atoms_high_divergence():
    """Dissimilar atoms cannot substitute for each other → D ≈ 1."""
    pa, pb = _pattern_a(), _pattern_b()
    data = np.zeros((6, 120), dtype=np.float64)
    for t0 in (5, 25, 45):
        data[:, t0 : t0 + pa.shape[1]] += pa
    for t0 in (70, 90, 105):
        data[:, t0 : t0 + pb.shape[1]] += pb
    window = _make_window(data)
    dictionary = Dictionary()
    dictionary.add(_unit_template(0, pa))
    dictionary.add(_unit_template(1, pb))

    result = calc_divergences(
        [window],
        dictionary,
        eps_step_fraction=1e-12,
        min_delta_fraction=1e-12,
        show_progress=False,
    )
    assert result.significant.all()
    # Off-diagonal should be high (B does not re-explain A's energy).
    d01 = float(result.divergences[0, 1])
    d10 = float(result.divergences[1, 0])
    assert d01 > 0.7
    assert d10 > 0.7
    # Diagonal: re-matching the same atom should recover most of R.
    assert float(result.divergences[0, 0]) < 0.3
    assert float(result.divergences[1, 1]) < 0.3


def test_insignificant_dead_atom():
    """Atom with no useful content does not clear the ΔA gate."""
    pa = _pattern_a()
    dead = np.zeros_like(pa)
    dead[4, 0] = 1e-3  # tiny / nearly invisible signal
    data = np.zeros((6, 80), dtype=np.float64)
    for t0 in (5, 25, 45):
        data[:, t0 : t0 + pa.shape[1]] += pa
    window = _make_window(data)
    dictionary = Dictionary()
    dictionary.add(_unit_template(0, pa))
    dictionary.add(_unit_template(99, dead))

    result = calc_divergences(
        [window],
        dictionary,
        eps_step_fraction=1e-12,
        min_delta_fraction=1e-3,  # require ≥ 0.1% of stack energy
        show_progress=False,
    )
    assert result.template_ids == [0, 99]
    assert bool(result.significant[0]) is True
    # Dead atom may or may not match at all; if significant fails, row is nan.
    if not bool(result.significant[1]):
        assert np.all(np.isnan(result.divergences[1, :]))
    else:
        # If the tiny atom somehow matched, ΔA is still tiny relative to A.
        assert float(result.delta_energy[1]) < float(result.delta_energy[0])


def test_asymmetric_when_strengths_differ():
    """Louder atom vs weaker non-substitute: off-diagonals need not match."""
    pa = _pattern_a()
    pb = _pattern_b()
    data = np.zeros((6, 120), dtype=np.float64)
    # Strong A (loud and frequent), weak B (soft, fewer repeats).
    for t0 in (5, 25, 45, 65):
        data[:, t0 : t0 + pa.shape[1]] += 2.0 * pa
    for t0 in (90, 105):
        data[:, t0 : t0 + pb.shape[1]] += 0.5 * pb
    window = _make_window(data)
    dictionary = Dictionary()
    dictionary.add(_unit_template(0, pa))
    dictionary.add(_unit_template(1, pb))

    result = calc_divergences(
        [window],
        dictionary,
        eps_step_fraction=1e-12,
        min_delta_fraction=1e-12,
        show_progress=False,
    )
    assert result.significant.any()
    d01 = result.divergences[0, 1]
    d10 = result.divergences[1, 0]
    # Not required to be equal; just assert both are finite when significant.
    if result.significant[0] and result.significant[1]:
        assert np.isfinite(d01) and np.isfinite(d10)
        # Orthogonal patterns: neither substitutes the other.
        assert float(d01) > 0.5
        assert float(d10) > 0.5
