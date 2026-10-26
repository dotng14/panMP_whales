"""Tests for MP acceleration: Numba parity and per-window mode."""

from __future__ import annotations


import numpy as np


from analysis.dict_learning.code.mp import run_mp_loop, single_template_dictionary  # noqa: E402
from analysis.dict_learning.code.core import Dictionary, Segment, Template  # noqa: E402
from analysis.dict_learning.code.mp import numba_available, score_shifts  # noqa: E402

EPS_PEEL = 1e-12


def test_score_shifts_numba_matches_numpy():
    rng = np.random.default_rng(42)
    f, w, t = 8, 7, 40
    tmpl = rng.standard_normal((f, w))
    resid = rng.standard_normal((f, t))
    numpy_scores = score_shifts(tmpl, resid, normalize=True, use_numba=False, compute="cpu")
    if not numba_available():
        assert numpy_scores.shape == (t - w + 1,)
        return
    numba_scores = score_shifts(tmpl, resid, normalize=True, use_numba=True, compute="cpu")
    np.testing.assert_allclose(numba_scores, numpy_scores, rtol=1e-12, atol=1e-12)


def test_gpu_auto_falls_back_to_cpu_when_unusable():
    from analysis.dict_learning.code.mp import cupy_available, gpu_beneficial

    rng = np.random.default_rng(0)
    # Large enough to prefer GPU when usable.
    f, w, t = 64, 32, 2000
    assert gpu_beneficial(f, w, t)
    tmpl = rng.standard_normal((f, w))
    resid = rng.standard_normal((f, t))
    cpu = score_shifts(tmpl, resid, compute="cpu", use_numba=False)
    auto = score_shifts(tmpl, resid, compute="auto")
    # Whether CuPy works or not, auto must return a finite matching-length vector.
    assert auto.shape == cpu.shape
    if not cupy_available():
        np.testing.assert_allclose(auto, cpu, rtol=1e-10, atol=1e-10)


def test_per_window_equals_global_on_single_window():
    f, t, w = 4, 50, 5
    tmpl = np.ones((f, w), dtype=np.float64)
    residual = np.zeros((f, t), dtype=np.float64)
    residual[:, 5 : 5 + w] = 3.0 * tmpl
    residual[:, 25 : 25 + w] = 2.0 * tmpl
    d = Dictionary()
    d.add(Template(template_id=0, data=tmpl))
    stack = {"w0": residual}
    g = run_mp_loop(stack, d, eps_step_fraction=EPS_PEEL, mp_mode="global", n_workers=1)
    p = run_mp_loop(stack, d, eps_step_fraction=EPS_PEEL, mp_mode="per_window", n_workers=1)
    assert len(g.matchings) == len(p.matchings)
    assert {(m.t_start, round(m.amplitude, 9)) for m in g.matchings} == {
        (m.t_start, round(m.amplitude, 9)) for m in p.matchings
    }


def test_per_window_places_atoms_in_each_window():
    f, t, w = 3, 40, 4
    tmpl = np.ones((f, w), dtype=np.float64)
    stack = {}
    for i, amp in enumerate((5.0, 4.0, 3.0)):
        r = np.zeros((f, t), dtype=np.float64)
        r[:, 10 : 10 + w] = amp * tmpl
        stack[f"w{i}"] = r
    d = single_template_dictionary(Template(template_id=0, data=tmpl))
    loop = run_mp_loop(
        stack, d, eps_step_fraction=EPS_PEEL, mp_mode="per_window", n_workers=2
    )
    windows_hit = {m.window_id for m in loop.matchings}
    assert windows_hit == set(stack)


def test_numba_score_shifts_releases_gil():
    from analysis.dict_learning.code.mp import _score_shifts_numba, numba_available

    if not numba_available():
        return
    rng = np.random.default_rng(0)
    tmpl = rng.standard_normal((8, 5))
    resid = rng.standard_normal((8, 40))
    _score_shifts_numba(tmpl, resid, True)
    key = list(_score_shifts_numba.overloads.keys())[0]
    # Numba stores nogil on the compile flags / target options.
    opts = getattr(_score_shifts_numba, "targetoptions", {}) or {}
    assert opts.get("nogil", False) is True or opts.get("nopython", False) is True
    # Re-check decorator intent via dispatcher targetoptions after compile.
    assert _score_shifts_numba.targetoptions.get("nogil") is True


def test_per_window_inplace_matches_manual_copy_loop():
    """Regression lock for the mp_step.py copy-elision fix (Plan 13 item #5).

    ``_run_mp_loop_one_window`` now owns one private residual array and
    mutates it in place via ``_mp_step_one_window_inplace`` instead of
    re-copying the whole ``(F, T)`` array through ``mp_step_one_window`` on
    every step. Manually drive the old, always-copies public API by hand and
    assert the hot loop's output is bit-identical.
    """
    from analysis.dict_learning.code.mp import mp_step_one_window

    rng = np.random.default_rng(7)
    f, t = 6, 120
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=rng.standard_normal((f, 5))))
    dictionary.add(Template(template_id=1, data=rng.standard_normal((f, 9))))
    residual = rng.standard_normal((f, t))
    eps_step_fraction = 1e-9

    manual_resid = np.array(residual, dtype=np.float64, copy=True)
    manual_matchings = []
    first_delta = None
    while True:
        manual_resid, step = mp_step_one_window(
            manual_resid, dictionary, window_id="w0", normalize=True,
        )
        if step.matching is None or step.delta_energy <= 0.0:
            break
        if first_delta is None:
            first_delta = step.delta_energy
        elif step.delta_energy < eps_step_fraction * first_delta:
            break
        manual_matchings.append(step.matching)

    loop = run_mp_loop(
        {"w0": residual},
        dictionary,
        eps_step_fraction=eps_step_fraction,
        normalize=True,
        mp_mode="per_window",
        n_workers=1,
    )

    # `_run_mp_loop_per_window` re-sorts matchings by (window_id, t_start,
    # template_id) for reproducible catalogs (pre-existing, unrelated to this
    # fix) and only reports [energy_before, energy_after] rather than a full
    # per-step trace — so compare the placement *set* and final state, not
    # positional/placement order.
    assert len(loop.matchings) == len(manual_matchings) > 0
    loop_triples = sorted((m.template_id, m.t_start, m.amplitude) for m in loop.matchings)
    manual_triples = sorted((m.template_id, m.t_start, m.amplitude) for m in manual_matchings)
    assert loop_triples == manual_triples
    np.testing.assert_array_equal(loop.residual_stack["w0"], manual_resid)
    assert loop.energies[-1] == float(np.sum(manual_resid**2))


def test_run_mp_loop_does_not_mutate_input_residual():
    """run_mp_loop deep-copies on entry; caller's arrays stay intact (Plan 14)."""
    rng = np.random.default_rng(11)
    f, t = 5, 80
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=rng.standard_normal((f, 6))))
    residual = rng.standard_normal((f, t))
    before = residual.copy()
    loop = run_mp_loop(
        {"w0": residual, "w1": residual * 0.5},
        dictionary,
        eps_step_fraction=1e-9,
        normalize=True,
        mp_mode="per_window",
        n_workers=2,
    )
    np.testing.assert_array_equal(residual, before)
    assert isinstance(loop.matchings, list)


def _single_pattern_stack(width: int = 5, t0: int = 20) -> tuple[dict, Dictionary]:
    f, t = 4, 60
    pattern = np.zeros((f, width), dtype=np.float64)
    pattern[0, :] = 2.0
    pattern[1, :] = -2.0
    resid = np.zeros((f, t), dtype=np.float64)
    resid[:, t0 : t0 + width] += pattern
    return {"w0": resid}, single_template_dictionary(pattern, template_id=0)


def test_occupied_intervals_bind_even_without_disallow_overlap():
    """Caller-declared spans are reserved; disallow_overlap only adds new ones."""
    stack, dictionary = _single_pattern_stack()
    for mode in ("global", "per_window"):
        free = run_mp_loop(
            stack, dictionary, eps_step_fraction=EPS_PEEL, mp_mode=mode, n_workers=1
        )
        assert [m.t_start for m in free.matchings] == [20]

        blocked = run_mp_loop(
            stack,
            dictionary,
            eps_step_fraction=EPS_PEEL,
            disallow_overlap=False,
            occupied_intervals={"w0": [(20, 25)]},
            mp_mode=mode,
            n_workers=1,
        )
        assert blocked.matchings == []


def test_refit_and_deflate_refits_amplitude_and_reports_spans():
    from analysis.dict_learning.code.core import Matching
    from analysis.dict_learning.code.mp import refit_and_deflate, residual_energy

    stack, dictionary = _single_pattern_stack()
    stack = {k: v * 3.0 for k, v in stack.items()}
    refitted, occupied = refit_and_deflate(
        stack, dictionary, [Matching(0, "w0", 20, 1.0)], normalize=True
    )
    assert [(m.template_id, m.t_start) for m in refitted] == [(0, 20)]
    assert refitted[0].amplitude == 3.0
    assert occupied == {"w0": [(20, 25)]}
    assert residual_energy(stack) < 1e-12

    # Unknown template ids and windows are dropped rather than raising.
    stack, dictionary = _single_pattern_stack()
    dropped, spans = refit_and_deflate(
        stack,
        dictionary,
        [Matching(7, "w0", 20, 1.0), Matching(0, "nope", 20, 1.0)],
    )
    assert dropped == [] and spans == {}


