"""Tests for MP acceleration: Numba parity, per-window mode, parallel candidates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.mp.mp_step import run_mp_loop, single_template_dictionary  # noqa: E402
from analysis.dict_learning.code.core.objects import Dictionary, Segment, Template  # noqa: E402
from analysis.dict_learning.code.mp.score_shifts import numba_available, score_shifts  # noqa: E402
from analysis.dict_learning.code.learn.seed_from_gpl import Seed  # noqa: E402

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


def test_allocate_nested_workers_saturates_cpus():
    from analysis.dict_learning.code.mp.mp_parallel import allocate_nested_workers, available_cpu_count

    cpus = available_cpu_count()
    assert cpus >= 1
    outer, inner = allocate_nested_workers(1, total_workers=cpus)
    assert outer == 1 and inner == cpus
    outer, inner = allocate_nested_workers(cpus * 2, total_workers=cpus)
    assert outer == cpus and inner == 1
    outer, inner = allocate_nested_workers(4, total_workers=16)
    assert outer == 4 and inner == 4
    assert outer * inner <= 16


def test_gpu_auto_falls_back_to_cpu_when_unusable():
    from analysis.dict_learning.code.mp.score_shifts import cupy_available, gpu_beneficial

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


def test_parallel_candidates_same_winner():
    f, t, w = 4, 60, 5
    pattern = np.ones((f, w), dtype=np.float64) * 2.0
    residual = np.zeros((f, t), dtype=np.float64)
    for t0 in (5, 20, 35):
        residual[:, t0 : t0 + w] += pattern
    spike = np.ones((f, 3), dtype=np.float64) * 3.0
    residual[:, 50:53] += spike
    from analysis.dict_learning.code.core.objects import Window

    window = Window(
        window_id="w0",
        data=residual,
        start_sample=1,
        end_sample=1 + t * 10 - 1,
        overlap=10,
    )
    true_seg = Segment(
        segment_id="true",
        data=pattern.copy(),
        window_id="w0",
        t_start=5,
        t_end=5 + w,
        global_start_sample=1,
        global_end_sample=2,
        detection_index=0,
    )
    spike_seg = Segment(
        segment_id="spike",
        data=spike.copy(),
        window_id="w0",
        t_start=50,
        t_end=53,
        global_start_sample=1,
        global_end_sample=2,
        detection_index=1,
    )
    serial = Seed.from_gpl(
        [window],
        [true_seg, spike_seg],
        eps_step_fraction=EPS_PEEL,
        min_repeats=1,
        n_workers=1,
        mp_mode="global",
    )
    parallel = Seed.from_gpl(
        [window],
        [true_seg, spike_seg],
        eps_step_fraction=EPS_PEEL,
        min_repeats=1,
        n_workers=2,
        mp_mode="global",
    )
    assert serial.get_selected_segment_id() == "true"
    assert parallel.get_selected_segment_id() == "true"
    # Winner rematerialization must provide a residual stack.
    assert serial.get_residual_after() is not None
    assert set(serial.get_residual_after()) == {"w0"}


def test_candidate_trial_omits_residual_by_default():
    from analysis.dict_learning.code.mp.mp_workers import (
        CandidateTrialPayload,
        rematerialize_candidate_trial,
        run_candidate_trial,
    )

    f, t, w = 3, 30, 4
    tmpl = np.ones((f, w), dtype=np.float64)
    resid = np.zeros((f, t), dtype=np.float64)
    resid[:, 5 : 5 + w] = 2.0 * tmpl
    payload = CandidateTrialPayload(
        segment_id="s0",
        template_id=0,
        template_data=tmpl,
        residual={"w0": resid},
        eps_step_fraction=EPS_PEEL,
        normalize=True,
        disallow_overlap=True,
        occupied_intervals=None,
        mp_mode="global",
        n_workers=1,
    )
    scored = run_candidate_trial(payload)
    assert scored.n_matchings >= 1
    assert scored.residual_stack == {}
    winner = rematerialize_candidate_trial(payload)
    assert winner.residual_stack
    assert "w0" in winner.residual_stack
    assert winner.energy_after == scored.energy_after


def test_numba_score_shifts_releases_gil():
    from analysis.dict_learning.code.mp.score_shifts import _score_shifts_numba, numba_available

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


def test_process_pool_shared_residual_matches_serial():
    from analysis.dict_learning.code.mp.mp_parallel import map_parallel
    from analysis.dict_learning.code.mp.mp_workers import (
        CandidateTrialPayload,
        init_trial_worker_shared,
        run_candidate_trial,
        slim_payloads_for_process_pool,
    )

    f, t, w = 4, 40, 5
    pattern = np.ones((f, w), dtype=np.float64) * 2.0
    resid = np.zeros((f, t), dtype=np.float64)
    for t0 in (3, 15, 27):
        resid[:, t0 : t0 + w] += pattern
    residual = {"w0": resid}
    payloads = [
        CandidateTrialPayload(
            segment_id=f"c{i}",
            template_id=0,
            template_data=pattern.copy(),
            residual=residual,
            eps_step_fraction=EPS_PEEL,
            normalize=True,
            disallow_overlap=True,
            occupied_intervals=None,
            mp_mode="global",
            n_workers=1,
        )
        for i in range(4)
    ]
    serial = [run_candidate_trial(p) for p in payloads]
    parallel = map_parallel(
        run_candidate_trial,
        slim_payloads_for_process_pool(payloads),
        n_workers=2,
        prefer_threads=False,
        initializer=init_trial_worker_shared,
        initargs=(residual, None),
    )
    assert [r.energy_after for r in parallel] == [r.energy_after for r in serial]
    assert all(r.residual_stack == {} for r in parallel)


def test_per_window_inplace_matches_manual_copy_loop():
    """Regression lock for the mp_step.py copy-elision fix (Plan 13 item #5).

    ``_run_mp_loop_one_window`` now owns one private residual array and
    mutates it in place via ``_mp_step_one_window_inplace`` instead of
    re-copying the whole ``(F, T)`` array through ``mp_step_one_window`` on
    every step. Manually drive the old, always-copies public API by hand and
    assert the hot loop's output is bit-identical.
    """
    from analysis.dict_learning.code.mp.mp_step import mp_step_one_window

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
    from analysis.dict_learning.code.core.objects import Matching
    from analysis.dict_learning.code.mp.mp_step import refit_and_deflate, residual_energy

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


def test_thread_pool_candidate_trials_match_serial():
    """DICT_LEARNING_MP_CANDIDATE_POOL=threads (the new default) must select
    the same templates/matchings as serial (n_workers=1) — Plan 13's
    thread-safety argument rests on run_mp_loop always deep-copying its input
    before mutating; this is the regression lock for that invariant.
    """
    import os

    from analysis.dict_learning.code.learn.learn_dictionary import learn_averaged_templates_from_gpl
    from analysis.dict_learning.code.core.objects import Window

    f, t = 4, 120
    pattern_a = np.zeros((f, 5), dtype=np.float64)
    pattern_a[0, :] = 3.0
    pattern_a[1, :] = -3.0
    pattern_b = np.zeros((f, 4), dtype=np.float64)
    pattern_b[2, :] = 2.0
    pattern_b[3, :] = -2.0
    residual = np.zeros((f, t), dtype=np.float64)
    for t0 in (5, 25, 45):
        residual[:, t0 : t0 + 5] += pattern_a
    for t0 in (70, 90):
        residual[:, t0 : t0 + 4] += pattern_b
    window = Window(
        window_id="w0", data=residual, start_sample=1, end_sample=1 + t * 10 - 1, overlap=10
    )
    seg_a = Segment(
        segment_id="segA", data=pattern_a.copy(), window_id="w0", t_start=5, t_end=10,
        global_start_sample=1, global_end_sample=2, detection_index=0,
    )
    seg_b = Segment(
        segment_id="segB", data=pattern_b.copy(), window_id="w0", t_start=70, t_end=74,
        global_start_sample=1, global_end_sample=2, detection_index=1,
    )

    serial = learn_averaged_templates_from_gpl(
        [window], [seg_a, seg_b],
        n_rounds=2, eps_step_fraction=EPS_PEEL, min_repeats=2, n_workers=1,
    )

    prior = os.environ.get("DICT_LEARNING_MP_CANDIDATE_POOL")
    os.environ["DICT_LEARNING_MP_CANDIDATE_POOL"] = "threads"
    try:
        threaded = learn_averaged_templates_from_gpl(
            [window], [seg_a, seg_b],
            n_rounds=2, eps_step_fraction=EPS_PEEL, min_repeats=2, n_workers=2,
        )
    finally:
        if prior is None:
            os.environ.pop("DICT_LEARNING_MP_CANDIDATE_POOL", None)
        else:
            os.environ["DICT_LEARNING_MP_CANDIDATE_POOL"] = prior

    s_tmpls = serial.get_templates()
    t_tmpls = threaded.get_templates()
    assert len(s_tmpls) == len(t_tmpls) == 2
    for s, t_ in zip(s_tmpls, t_tmpls):
        assert s.source_segment_id == t_.source_segment_id
        np.testing.assert_allclose(s.data, t_.data)
    assert len(serial.get_matchings()) == len(threaded.get_matchings())


if __name__ == "__main__":
    test_score_shifts_numba_matches_numpy()
    test_per_window_equals_global_on_single_window()
    test_per_window_places_atoms_in_each_window()
    test_parallel_candidates_same_winner()
    test_candidate_trial_omits_residual_by_default()
    test_numba_score_shifts_releases_gil()
    test_process_pool_shared_residual_matches_serial()
    test_per_window_inplace_matches_manual_copy_loop()
    test_run_mp_loop_does_not_mutate_input_residual()
    test_occupied_intervals_bind_even_without_disallow_overlap()
    test_refit_and_deflate_refits_amplitude_and_reports_spans()
    test_thread_pool_candidate_trials_match_serial()
    print("test_mp_fast: OK")
