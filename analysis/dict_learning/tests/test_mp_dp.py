"""Tests for the exact non-overlapping DP matching pursuit."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.core import (  # noqa: E402
    Dictionary,
    Template,
    Window,
)
from analysis.dict_learning.code.mp import run_mp_dp_loop  # noqa: E402
from analysis.dict_learning.code.mp import (  # noqa: E402
    residual_energy,
    run_mp_loop,
)


def _unit(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    return arr / float(np.linalg.norm(arr))


def _dictionary(*templates: np.ndarray) -> Dictionary:
    d = Dictionary()
    for tid, data in enumerate(templates):
        d.add(Template(template_id=tid, data=_unit(data)))
    return d


def _window(data: np.ndarray, wid: str = "w0") -> Window:
    cols = int(data.shape[1])
    return Window(
        window_id=wid,
        data=np.asarray(data, dtype=np.float64),
        start_sample=1,
        end_sample=1 + cols * 10 - 1,
        overlap=10,
    )


def _brute_force_best(
    x: np.ndarray, dictionary: Dictionary, penalty: float = 0.0
) -> float:
    """Max total weight over all non-overlapping placement sets (small inputs)."""
    placements: list[tuple[int, int, float]] = []
    for template in dictionary:
        basis = np.ascontiguousarray(template.basis, dtype=np.float64)
        tdata = basis[0]
        energy = float(np.sum(tdata * tdata))
        width = int(tdata.shape[1])
        for t in range(x.shape[1] - width + 1):
            inner = float(np.sum(x[:, t : t + width] * tdata))
            drop = (inner * inner) / energy
            if drop - penalty > 0.0:
                placements.append((t, t + width, drop - penalty))

    # A window this small fits only a handful of disjoint spans, so bounding the
    # subset size keeps the exhaustive search cheap without missing the optimum.
    max_disjoint = x.shape[1] // min(
        int(t.width) for t in dictionary
    )
    best = 0.0
    for r in range(max_disjoint + 1):
        for combo in itertools.combinations(placements, r):
            spans = sorted((a, b) for a, b, _w in combo)
            if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
                continue
            total = sum(w for _a, _b, w in combo)
            best = max(best, total)
    return best


def test_matches_brute_force_optimum():
    """DP must equal exhaustive search over non-overlapping placement sets."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 11))
    pa = np.zeros((3, 4))
    pa[0, :] = 1.0
    pb = np.zeros((3, 3))
    pb[1, :] = 1.0
    dictionary = _dictionary(pa, pb)

    result = run_mp_dp_loop(
        [_window(x)], dictionary, lambda_fraction=0.0, show_progress=False
    )
    achieved = residual_energy({"w0": x}) - residual_energy(result.residual_stack)
    expected = _brute_force_best(x, dictionary, penalty=0.0)
    assert achieved == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_dp_beats_greedy_when_best_pick_blocks_two():
    """One fat placement can block two better ones; greedy takes the bait."""
    pattern = np.zeros((1, 4), dtype=np.float64)
    pattern[0, :] = 1.0
    dictionary = _dictionary(pattern)

    # Two strong non-overlapping bumps, and a medium bump straddling both, so
    # the single best placement sits between the two better ones.
    x = np.zeros((1, 14), dtype=np.float64)
    x[0, 0:4] = 3.0
    x[0, 5:9] = 3.0
    x[0, 2:6] = 3.6

    greedy = run_mp_loop(
        [_window(x)],
        dictionary,
        eps_step_fraction=1e-12,
        disallow_overlap=True,
        show_progress=False,
    )
    dp = run_mp_dp_loop(
        [_window(x)], dictionary, lambda_fraction=0.0, show_progress=False
    )

    e_greedy = residual_energy(greedy.residual_stack)
    e_dp = residual_energy(dp.residual_stack)
    assert e_dp < e_greedy
    assert dp.stopped_reason == "dp_optimal"


def test_energy_drops_are_additive():
    """Disjoint placements: total drop equals the sum of per-placement drops."""
    pattern = np.zeros((2, 5), dtype=np.float64)
    pattern[0, :] = 2.0
    pattern[1, :] = -1.0
    dictionary = _dictionary(pattern)
    atom = _unit(pattern)

    x = np.zeros((2, 60), dtype=np.float64)
    for t in (3, 20, 44):
        x[:, t : t + 5] += pattern

    result = run_mp_dp_loop(
        [_window(x)], dictionary, lambda_fraction=0.0, show_progress=False
    )
    per_placement = 0.0
    for m in result.matchings:
        patch = x[:, m.t_start : m.t_start + 5]
        inner = float(np.sum(patch * atom))
        per_placement += inner * inner / float(np.sum(atom * atom))

    total = residual_energy({"w0": x}) - residual_energy(result.residual_stack)
    assert total == pytest.approx(per_placement, rel=1e-9, abs=1e-9)


def test_occupied_intervals_are_never_claimed():
    pattern = np.zeros((1, 4), dtype=np.float64)
    pattern[0, :] = 1.0
    dictionary = _dictionary(pattern)
    x = np.zeros((1, 30), dtype=np.float64)
    for t in (2, 12, 22):
        x[0, t : t + 4] = 5.0

    result = run_mp_dp_loop(
        [_window(x)],
        dictionary,
        lambda_fraction=0.0,
        occupied_intervals={"w0": [(10, 18)]},
        show_progress=False,
    )
    for m in result.matchings:
        assert not (m.t_start < 18 and m.t_start + 4 > 10)


def test_relative_distance_gate_excludes_far_placements():
    """A noisy bump only survives when the gate is loose."""
    pattern = np.zeros((2, 4), dtype=np.float64)
    pattern[0, :] = 1.0
    dictionary = _dictionary(pattern)

    x = np.zeros((2, 24), dtype=np.float64)
    x[0, 2:6] = 4.0  # clean match on the atom's row
    x[1, 12:16] = 4.0  # energy entirely off the atom

    loose = run_mp_dp_loop(
        [_window(x)],
        dictionary,
        lambda_fraction=0.0,
        max_relative_distance=1.0,
        show_progress=False,
    )
    strict = run_mp_dp_loop(
        [_window(x)],
        dictionary,
        lambda_fraction=0.0,
        max_relative_distance=0.05,
        show_progress=False,
    )
    assert len(strict.matchings) <= len(loose.matchings)
    for m in strict.matchings:
        assert m.t_start + 4 <= 12 or m.t_start >= 16


def test_lambda_fraction_trades_matchings_for_energy():
    rng = np.random.default_rng(3)
    pattern = np.zeros((2, 4), dtype=np.float64)
    pattern[0, :] = 1.0
    dictionary = _dictionary(pattern)

    x = rng.normal(size=(2, 80)) * 0.2
    for t, amp in ((5, 4.0), (20, 2.0), (40, 1.0), (60, 0.5)):
        x[0, t : t + 4] += amp

    counts, energies = [], []
    for frac in (0.0, 0.2, 0.6, 0.9):
        result = run_mp_dp_loop(
            [_window(x)], dictionary, lambda_fraction=frac, show_progress=False
        )
        counts.append(len(result.matchings))
        energies.append(residual_energy(result.residual_stack))

    assert counts == sorted(counts, reverse=True)
    assert energies == sorted(energies)


def test_max_repeats_caps_placements():
    pattern = np.zeros((1, 4), dtype=np.float64)
    pattern[0, :] = 1.0
    dictionary = _dictionary(pattern)
    x = np.zeros((1, 60), dtype=np.float64)
    for t in (2, 12, 22, 32, 42):
        x[0, t : t + 4] = float(10 - t // 10)

    uncapped = run_mp_dp_loop(
        [_window(x)], dictionary, lambda_fraction=0.0, show_progress=False
    )
    capped = run_mp_dp_loop(
        [_window(x)], dictionary, lambda_fraction=0.0, max_repeats=2, show_progress=False
    )
    assert len(uncapped.matchings) > 2
    assert len(capped.matchings) == 2
    assert capped.stopped_reason == "max_repeats"
    # With a budget of two, DP must pick the two strongest bumps.
    assert sorted(m.t_start for m in capped.matchings) == [2, 12]


def test_subspace_atoms_supported():
    basis = np.zeros((2, 3, 4), dtype=np.float64)
    basis[0, 0, :] = 0.5
    basis[1, 1, :] = 0.5
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=basis))

    x = np.zeros((3, 40), dtype=np.float64)
    x[0, 4:8] = 2.0
    x[1, 20:24] = 3.0

    result = run_mp_dp_loop(
        [_window(x)], dictionary, lambda_fraction=0.0, show_progress=False
    )
    assert result.matchings
    assert residual_energy(result.residual_stack) < residual_energy({"w0": x})
    for m in result.matchings:
        assert np.asarray(m.coeffs).size == 2


def test_rejects_bad_lambda_fraction():
    dictionary = _dictionary(np.ones((1, 2)))
    with pytest.raises(ValueError, match="lambda_fraction"):
        run_mp_dp_loop(
            [_window(np.ones((1, 10)))], dictionary, lambda_fraction=1.0
        )


def test_run_mp_loop_algorithm_switch():
    pattern = np.zeros((1, 4), dtype=np.float64)
    pattern[0, :] = 1.0
    dictionary = _dictionary(pattern)
    x = np.zeros((1, 30), dtype=np.float64)
    x[0, 3:7] = 4.0

    result = run_mp_loop(
        [_window(x)], dictionary, algorithm="dp", show_progress=False
    )
    assert result.stopped_reason == "dp_optimal"
    assert len(result.matchings) >= 1

    with pytest.raises(ValueError, match="algorithm"):
        run_mp_loop([_window(x)], dictionary, algorithm="nope")
