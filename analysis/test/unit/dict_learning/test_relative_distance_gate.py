"""Tests for the relative-distance gate on MP acceptance and template rebuilds."""

from __future__ import annotations


import numpy as np


from analysis.dict_learning.code.core import (  # noqa: E402
    Dictionary,
    Matching,
    Template,
)
from analysis.dict_learning.code.core import (  # noqa: E402
    filter_matchings_by_relative_distance,
    patch_template_relative_distance,
)
from analysis.dict_learning.code.mp import (  # noqa: E402
    _best_placement_on_window,
    mp_step_one_window,
    run_mp_loop,
)
from analysis.dict_learning.code.tools import (  # noqa: E402
    average_patches,
    pca_basis,
    rebuild_template_from_matchings,
)

WIDTH = 3
N_FREQ = 2


def _directions() -> tuple[np.ndarray, np.ndarray]:
    """Unit-norm template direction and an orthogonal one, both (F, W)."""
    v = np.zeros((N_FREQ, WIDTH), dtype=np.float64)
    v[0, :] = 1.0
    v /= np.linalg.norm(v)
    orth = np.zeros((N_FREQ, WIDTH), dtype=np.float64)
    orth[1, :] = 1.0
    orth /= np.linalg.norm(orth)
    return v, orth


def _make_stack_and_matchings():
    """One window holding two patches close to ``v`` and two far from it."""
    v, orth = _directions()
    arr = np.zeros((N_FREQ, WIDTH * 4), dtype=np.float64)
    patches = [
        3.0 * v + 0.1 * orth,
        2.5 * v + 0.2 * orth,
        0.2 * v + 4.0 * orth,
        0.1 * v + 5.0 * orth,
    ]
    matchings: list[Matching] = []
    for i, patch in enumerate(patches):
        t0 = i * WIDTH
        arr[:, t0 : t0 + WIDTH] = patch
        matchings.append(
            Matching(template_id=0, window_id="w0", t_start=t0, amplitude=1.0)
        )
    return {"w0": arr}, matchings, Template(template_id=0, data=v), v, orth


# -- metric -------------------------------------------------------------------


def test_relative_distance_is_scale_free():
    v, orth = _directions()
    patch = 2.0 * v + 1.0 * orth
    r = patch_template_relative_distance(patch, v)
    assert abs(r - 0.2) < 1e-12
    # Scaling the patch leaves the unexplained fraction unchanged.
    assert abs(patch_template_relative_distance(100.0 * patch, v) - r) < 1e-12


def test_relative_distance_bounds():
    v, orth = _directions()
    assert patch_template_relative_distance(3.0 * v, v) < 1e-12
    assert abs(patch_template_relative_distance(3.0 * orth, v) - 1.0) < 1e-12
    assert patch_template_relative_distance(np.zeros((N_FREQ, WIDTH)), v) == 1.0


# -- rebuild filter -----------------------------------------------------------


def test_filter_keeps_close_patches_only():
    stack, matchings, ref, *_ = _make_stack_and_matchings()
    kept = filter_matchings_by_relative_distance(
        stack, matchings, ref, max_relative_distance=0.5, min_keep=1
    )
    assert sorted(m.t_start for m in kept) == [0, WIDTH]


def test_filter_tau_one_keeps_all():
    stack, matchings, ref, *_ = _make_stack_and_matchings()
    kept = filter_matchings_by_relative_distance(
        stack, matchings, ref, max_relative_distance=1.0
    )
    assert len(kept) == len(matchings)


def test_filter_min_keep_fallback_returns_closest():
    stack, matchings, ref, *_ = _make_stack_and_matchings()
    kept = filter_matchings_by_relative_distance(
        stack, matchings, ref, max_relative_distance=0.0, min_keep=3
    )
    assert len(kept) == 3
    # Only the single farthest patch is dropped.
    assert 3 * WIDTH not in {m.t_start for m in kept}


def test_filter_skips_gate_for_zero_reference():
    stack, matchings, _ref, *_ = _make_stack_and_matchings()
    placeholder = Template(
        template_id=0, data=np.zeros((N_FREQ, WIDTH), dtype=np.float64)
    )
    kept = filter_matchings_by_relative_distance(
        stack, matchings, placeholder, max_relative_distance=0.1
    )
    assert len(kept) == len(matchings)


def test_pca_basis_uses_close_patches_only():
    stack, matchings, ref, v, _orth = _make_stack_and_matchings()
    basis_all, _ = pca_basis(stack, matchings, width=WIDTH, n_freq=N_FREQ)
    basis_close, _ = pca_basis(
        stack,
        matchings,
        width=WIDTH,
        n_freq=N_FREQ,
        reference_template=ref,
        max_relative_distance=0.5,
    )
    assert abs(float(np.sum(basis_close[0] * v))) > 0.95
    assert abs(float(np.sum(basis_close[0] * v))) > abs(
        float(np.sum(basis_all[0] * v))
    )


def test_average_patches_gate_applies():
    stack, matchings, ref, v, _orth = _make_stack_and_matchings()
    mean_all = average_patches(stack, matchings, width=WIDTH, n_freq=N_FREQ)
    mean_close = average_patches(
        stack,
        matchings,
        width=WIDTH,
        n_freq=N_FREQ,
        reference_template=ref,
        max_relative_distance=0.5,
    )
    cos_close = float(np.sum(mean_close * v)) / float(np.linalg.norm(mean_close))
    cos_all = float(np.sum(mean_all * v)) / float(np.linalg.norm(mean_all))
    assert cos_close > cos_all


def test_rebuild_gates_average_path_too():
    stack, matchings, ref, v, _orth = _make_stack_and_matchings()
    ungated = rebuild_template_from_matchings(
        stack, ref, matchings, method="average"
    )
    gated = rebuild_template_from_matchings(
        stack, ref, matchings, method="average", max_relative_distance=0.5
    )
    assert patch_template_relative_distance(
        gated.data, v
    ) < patch_template_relative_distance(ungated.data, v)


def test_rebuild_gates_subspace_path():
    stack, matchings, ref, v, _orth = _make_stack_and_matchings()
    rebuilt = rebuild_template_from_matchings(
        stack, ref, matchings, n_components=2, method="pca", max_relative_distance=0.5
    )
    assert rebuilt.n_components == 2
    assert abs(float(np.sum(rebuilt.data[0] * v))) > 0.9


# -- MP acceptance ------------------------------------------------------------


def _window_with_dirty_peak() -> tuple[np.ndarray, np.ndarray]:
    """Window whose strongest placement is a poor fit and whose second is exact.

    Columns 0-2 hold ``10v + 10*orth`` (large score, half its energy off the
    template); columns 4-6 hold ``3v`` (smaller score, perfect fit).
    """
    v, orth = _directions()
    resid = np.zeros((N_FREQ, 8), dtype=np.float64)
    resid[:, 0:3] = 10.0 * v + 10.0 * orth
    resid[:, 4:7] = 3.0 * v
    return resid, v


def test_mp_gate_falls_through_to_next_best_shift():
    resid, v = _window_with_dirty_peak()
    basis = v[np.newaxis, ...]
    ungated = _best_placement_on_window(
        basis,
        resid,
        template_id=0,
        window_id="w0",
        occupied_intervals=None,
        normalize=True,
    )
    assert ungated is not None and ungated[1].t_start == 0

    gated = _best_placement_on_window(
        basis,
        resid,
        template_id=0,
        window_id="w0",
        occupied_intervals=None,
        normalize=True,
        max_relative_distance=0.1,
    )
    assert gated is not None
    assert gated[1].t_start == 4
    assert abs(float(gated[1].amplitude) - 3.0) < 1e-9


def test_mp_gate_matches_under_unnormalized_scores():
    resid, v = _window_with_dirty_peak()
    # Template deliberately not unit-norm: explained energy must still divide
    # the inner product by ||T||².
    basis = (2.0 * v)[np.newaxis, ...]
    gated = _best_placement_on_window(
        basis,
        resid,
        template_id=0,
        window_id="w0",
        occupied_intervals=None,
        normalize=False,
        max_relative_distance=0.1,
    )
    assert gated is not None and gated[1].t_start == 4


def test_mp_gate_returns_none_when_every_shift_is_far():
    resid, v = _window_with_dirty_peak()
    resid[:, 4:7] = 0.0  # remove the clean patch; only the dirty peak remains
    placed = _best_placement_on_window(
        v[np.newaxis, ...],
        resid,
        template_id=0,
        window_id="w0",
        occupied_intervals=None,
        normalize=True,
        max_relative_distance=0.1,
    )
    assert placed is None


def test_mp_gate_does_not_deflate_rejected_shift():
    resid, v = _window_with_dirty_peak()
    before = resid.copy()
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=v))
    after, step = mp_step_one_window(
        resid, dictionary, window_id="w0", max_relative_distance=0.1
    )
    assert step.matching is not None and step.matching.t_start == 4
    # The rejected peak keeps every bit of its energy.
    assert np.allclose(after[:, 0:3], before[:, 0:3])
    assert np.allclose(after[:, 4:7], 0.0, atol=1e-9)


def test_mp_gate_on_subspace_atom():
    v, orth = _directions()
    third = np.zeros((N_FREQ, WIDTH), dtype=np.float64)
    third[0, 0] = 1.0
    third[0, 1] = -1.0
    third /= np.linalg.norm(third)
    basis = np.stack([v, third], axis=0)  # orthonormal 2-mode subspace

    resid = np.zeros((N_FREQ, 8), dtype=np.float64)
    resid[:, 0:3] = 8.0 * v + 8.0 * orth  # high projection energy, poor fit
    resid[:, 4:7] = 2.0 * v + 2.0 * third  # entirely inside the subspace

    ungated = _best_placement_on_window(
        basis,
        resid,
        template_id=0,
        window_id="w0",
        occupied_intervals=None,
        normalize=False,
    )
    assert ungated is not None and ungated[1].t_start == 0

    gated = _best_placement_on_window(
        basis,
        resid,
        template_id=0,
        window_id="w0",
        occupied_intervals=None,
        normalize=False,
        max_relative_distance=0.1,
    )
    assert gated is not None and gated[1].t_start == 4


def test_mp_gate_pc1_screen_falls_back_to_full_scan():
    v, orth = _directions()
    third = np.zeros((N_FREQ, WIDTH), dtype=np.float64)
    third[0, 0] = 1.0
    third[0, 1] = -1.0
    third /= np.linalg.norm(third)
    basis = np.stack([v, third], axis=0)

    resid = np.zeros((N_FREQ, 8), dtype=np.float64)
    resid[:, 0:3] = 8.0 * v + 8.0 * orth
    resid[:, 4:7] = 2.0 * v + 2.0 * third

    # screen_top=1 puts only the (rejected) dirty peak in front of the gate.
    screened = _best_placement_on_window(
        basis,
        resid,
        template_id=0,
        window_id="w0",
        occupied_intervals=None,
        normalize=False,
        pc1_screen=True,
        screen_top=1,
        max_relative_distance=0.1,
    )
    assert screened is not None and screened[1].t_start == 4


def _random_stack(n_windows: int = 3, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    v, orth = _directions()
    stack: dict[str, np.ndarray] = {}
    for w in range(n_windows):
        arr = 0.05 * rng.standard_normal((N_FREQ, 24))
        for t0 in (2, 10, 18):
            arr[:, t0 : t0 + WIDTH] += rng.uniform(1.0, 3.0) * v + rng.uniform(
                0.0, 1.5
            ) * orth
        stack[f"w{w}"] = arr
    return stack


def _single_atom_dictionary() -> Dictionary:
    v, _orth = _directions()
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=v))
    return dictionary


def _keys(matchings) -> list[tuple[str, int, int]]:
    return [(m.window_id, int(m.t_start), int(m.template_id)) for m in matchings]


def test_tau_none_matches_tau_one():
    stack = _random_stack()
    dictionary = _single_atom_dictionary()
    kwargs = dict(
        eps_step_fraction=0.1,
        max_repeats=4,
        disallow_overlap=True,
        mp_mode="global",
        show_progress=False,
    )
    off = run_mp_loop(stack, dictionary, max_relative_distance=None, **kwargs)
    wide = run_mp_loop(stack, dictionary, max_relative_distance=1.0, **kwargs)
    assert _keys(off.matchings) == _keys(wide.matchings)
    assert off.stopped_reason == wide.stopped_reason


def test_global_and_per_window_agree_under_same_tau():
    stack = _random_stack(n_windows=1)
    dictionary = _single_atom_dictionary()
    kwargs = dict(
        eps_step_fraction=0.1,
        max_repeats=4,
        disallow_overlap=True,
        show_progress=False,
        max_relative_distance=0.4,
    )
    glob = run_mp_loop(stack, dictionary, mp_mode="global", **kwargs)
    per_window = run_mp_loop(stack, dictionary, mp_mode="per_window", **kwargs)
    assert _keys(glob.matchings) == _keys(per_window.matchings)


def test_tighter_tau_never_accepts_more():
    stack = _random_stack()
    dictionary = _single_atom_dictionary()
    kwargs = dict(
        eps_step_fraction=0.1,
        max_repeats=6,
        disallow_overlap=True,
        mp_mode="global",
        show_progress=False,
    )
    loose = run_mp_loop(stack, dictionary, max_relative_distance=0.9, **kwargs)
    tight = run_mp_loop(stack, dictionary, max_relative_distance=0.2, **kwargs)
    v, _orth = _directions()
    for m in tight.matchings:
        patch = stack[m.window_id][:, m.t_start : m.t_start + WIDTH]
        assert patch_template_relative_distance(patch, v) <= 0.2 + 1e-9
    assert len(tight.matchings) <= len(loose.matchings)


def test_run_mp_loop_rejects_out_of_range_tau():
    stack = _random_stack(n_windows=1)
    dictionary = _single_atom_dictionary()
    try:
        run_mp_loop(stack, dictionary, max_relative_distance=1.5)
    except ValueError as exc:
        assert "max_relative_distance" in str(exc)
    else:  # pragma: no cover - guard must fire
        raise AssertionError("expected ValueError for tau > 1")
