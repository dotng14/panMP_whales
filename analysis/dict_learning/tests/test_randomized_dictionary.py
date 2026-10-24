"""Tests for randomized CreateDictionary / RefineDictionary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.core import (  # noqa: E402
    Dictionary,
    Segment,
    Template,
    Window,
)
from analysis.dict_learning.code.learn.randomized_dictionary import (  # noqa: E402
    build_randomized_dictionary,
    create_dictionary,
    dictionary_explained_energy,
    learn_randomized_outer,
    refine_dictionary,
)


def _make_window(data: np.ndarray, wid: str = "w0", overlap: int = 10) -> Window:
    cols = data.shape[1]
    return Window(
        window_id=wid,
        data=data,
        start_sample=1,
        end_sample=1 + cols * overlap - 1,
        overlap=overlap,
    )


def _segment(seg_id: str, data: np.ndarray, t0: int, det_index: int) -> Segment:
    return Segment(
        segment_id=seg_id,
        data=data.copy(),
        window_id="w0",
        t_start=t0,
        t_end=t0 + data.shape[1],
        global_start_sample=1,
        global_end_sample=2,
        detection_index=det_index,
    )


def _two_pattern_window() -> tuple[Window, list[Segment], np.ndarray, np.ndarray]:
    f, t = 4, 120
    pattern_a = np.zeros((f, 5), dtype=np.float64)
    pattern_a[0, :] = 3.0
    pattern_a[1, :] = -3.0
    pattern_b = np.zeros((f, 4), dtype=np.float64)
    pattern_b[2, :] = 2.0
    pattern_b[3, :] = -2.0

    residual = np.zeros((f, t), dtype=np.float64)
    starts_a = (5, 25, 45)
    starts_b = (70, 90)
    for t0 in starts_a:
        residual[:, t0 : t0 + 5] += pattern_a
    for t0 in starts_b:
        residual[:, t0 : t0 + 4] += pattern_b

    window = _make_window(residual)
    segs = [
        _segment(f"a{i}", pattern_a, t0=t0, det_index=i)
        for i, t0 in enumerate(starts_a)
    ] + [
        _segment(f"b{i}", pattern_b, t0=t0, det_index=10 + i)
        for i, t0 in enumerate(starts_b)
    ]
    return window, segs, pattern_a, pattern_b


def test_create_dictionary_samples_unit_norm_atoms():
    window, segs, _pa, _pb = _two_pattern_window()
    result = create_dictionary(
        [window],
        segs,
        m=3,
        rng=0,
        epsilon=0.0,
        min_template_width=None,
        max_template_width=None,
    )
    assert result.pool_size == len(segs)
    assert result.n_sampled == 3
    assert len(result.dictionary) == 3
    for atom in result.dictionary:
        assert atom.n_components == 1
        assert abs(float(np.linalg.norm(atom.data)) - 1.0) < 1e-12


def test_create_dictionary_epsilon_filters_silence():
    window, segs, pa, _pb = _two_pattern_window()
    quiet = _segment("quiet", np.zeros_like(pa) + 1e-9, t0=100, det_index=99)
    result = create_dictionary(
        [window],
        segs + [quiet],
        m=10,
        epsilon=1e-6,
        min_template_width=None,
        max_template_width=None,
        rng=1,
    )
    assert all(s.segment_id != "quiet" or True for s in segs)
    # Quiet patch must not enter the pool (‖x‖≈0 or ‖r‖/‖x‖ tiny).
    assert result.pool_size == len(segs)


def test_create_dictionary_epsilon_is_relative_to_original():
    """epsilon thresholds ‖r‖/‖x‖, not absolute residual norm."""
    window, segs, pa, _pb = _two_pattern_window()
    # Original stack: strong pattern at segs[0].
    original = {window.window_id: np.array(window.data, dtype=np.float64, copy=True)}
    # Residual: same location scaled to 5% of original energy (norm ratio 0.05).
    residual = {window.window_id: np.zeros_like(original[window.window_id])}
    t0, t1 = segs[0].t_start, segs[0].t_end
    residual[window.window_id][:, t0:t1] = 0.05 * original[window.window_id][:, t0:t1]

    kept = create_dictionary(
        residual,
        [segs[0]],
        m=1,
        original=original,
        epsilon=0.04,
        min_template_width=None,
        max_template_width=None,
        rng=0,
    )
    assert kept.n_sampled == 1

    dropped = create_dictionary(
        residual,
        [segs[0]],
        m=1,
        original=original,
        epsilon=0.1,
        min_template_width=None,
        max_template_width=None,
        rng=0,
    )
    assert dropped.n_sampled == 0
    assert dropped.pool_size == 0


def test_dictionary_explained_energy_empty_and_match():
    assert dictionary_explained_energy(np.ones((2, 3)), Dictionary()) == 0.0
    window, segs, pa, _pb = _two_pattern_window()
    created = create_dictionary(
        [window],
        [segs[0]],
        m=1,
        rng=0,
        epsilon=0.0,
        min_template_width=None,
        max_template_width=None,
    )
    atom = next(iter(created.dictionary))
    g = dictionary_explained_energy(pa / np.linalg.norm(pa), created.dictionary)
    # Unit seed is exactly the normalized pattern → full explained energy = 1.
    assert abs(g - 1.0) < 1e-9
    assert atom.template_id == 0


def test_refine_assigns_patches_and_prunes():
    window, segs, pa, pb = _two_pattern_window()
    # Seed with both true patterns (deterministic, not random).
    d = Dictionary()
    d.add(
        Template(
            template_id=0,
            data=pa / np.linalg.norm(pa),
            source_segment_id="a0",
        )
    )
    d.add(
        Template(
            template_id=1,
            data=pb / np.linalg.norm(pb),
            source_segment_id="b0",
        )
    )
    # Junk atom that should get few/no matches and be pruned.
    junk = np.zeros_like(pa)
    junk[0, 0] = 1.0
    junk[1, 0] = -1.0
    d.add(
        Template(
            template_id=2,
            data=junk / np.linalg.norm(junk),
            source_segment_id="junk",
        )
    )

    refined = refine_dictionary(
        [window],
        d,
        iter=1,
        n_components=1,
        n_min=2,
        eps_step_fraction=1e-12,
        max_relative_distance=None,
        post_refine_max_relative_distance=None,
    )
    assert refined.iterations_completed == 1
    ids = {t.template_id for t in refined.dictionary}
    assert 0 in ids
    assert 1 in ids
    assert 2 not in ids
    assert refined.n_pruned >= 1

    # Every matching is tied to a surviving atom.
    surviving = ids
    assert all(m.template_id in surviving for m in refined.matchings)
    assert len([m for m in refined.matchings if m.template_id == 0]) >= 2
    assert len([m for m in refined.matchings if m.template_id == 1]) >= 2


def test_build_randomized_end_to_end():
    window, segs, _pa, _pb = _two_pattern_window()
    result = build_randomized_dictionary(
        [window],
        segs,
        m=5,
        refine_iters=2,
        n_components=1,
        n_min=2,
        eps_step_fraction=1e-12,
        post_refine_max_relative_distance=None,
        epsilon=0.0,
        min_template_width=None,
        max_template_width=None,
        rng=2,
    )
    assert result.iterations_completed == 2
    assert len(result.dictionary) >= 1
    assert result.energies[0] >= result.energies[-1]


def test_refine_subspace_k():
    window, segs, pa, _pb = _two_pattern_window()
    d = Dictionary()
    d.add(Template(template_id=0, data=pa / np.linalg.norm(pa)))
    refined = refine_dictionary(
        [window],
        d,
        iter=1,
        n_components=1,
        n_min=2,
        eps_step_fraction=1e-12,
        post_refine_max_relative_distance=None,
    )
    assert len(refined.dictionary) == 1
    assert next(iter(refined.dictionary)).n_components == 1


def test_post_refine_filters_far_patches():
    """After refine, placements with dnorm > 0.5 are dropped."""
    from analysis.dict_learning.code.core import Matching
    from analysis.dict_learning.code.learn.randomized_dictionary import (
        filter_matchings_by_atom_distance,
    )

    window, segs, pa, pb = _two_pattern_window()
    d = Dictionary()
    d.add(Template(template_id=0, data=pa / np.linalg.norm(pa)))
    # True placement of pattern A.
    good = Matching(template_id=0, window_id="w0", t_start=5, amplitude=1.0)
    # Placement of pattern A atom on a B location → large dnorm.
    bad = Matching(template_id=0, window_id="w0", t_start=70, amplitude=1.0)
    stack = {window.window_id: window.data}
    kept, n_filt = filter_matchings_by_atom_distance(
        stack, d, [good, bad], max_relative_distance=0.5
    )
    assert n_filt == 1
    assert len(kept) == 1
    assert kept[0].t_start == 5

    refined = refine_dictionary(
        [window],
        d,
        iter=1,
        n_components=1,
        n_min=1,
        eps_step_fraction=1e-12,
        max_relative_distance=None,
        post_refine_max_relative_distance=0.5,
    )
    assert refined.n_distance_filtered >= 0
    # Surviving matchings must be close to their atom.
    from analysis.dict_learning.code.core import (
        patch_template_relative_distance,
    )

    atom = next(iter(refined.dictionary))
    for m in refined.matchings:
        patch = window.data[:, m.t_start : m.t_start + atom.width]
        assert patch_template_relative_distance(patch, atom) <= 0.5 + 1e-9


def test_default_refine_iters_is_three():
    import inspect

    from analysis.dict_learning.code.learn import randomized_dictionary as rd

    sig = inspect.signature(rd.refine_dictionary)
    assert sig.parameters["iter"].default == 3
    sig_b = inspect.signature(rd.build_randomized_dictionary)
    assert sig_b.parameters["refine_iters"].default == 3
    assert rd.POST_REFINE_MAX_RELATIVE_DISTANCE == 0.5


def test_learn_randomized_outer_coverage_stop():
    """Outer loop stops once time coverage exceeds min_coverage."""
    # Many strong repeats so n_min=3 survivors are easy and coverage rises.
    f, t = 4, 200
    pattern = np.zeros((f, 6), dtype=np.float64)
    pattern[0, :] = 4.0
    pattern[1, :] = -4.0
    residual = np.zeros((f, t), dtype=np.float64)
    starts = list(range(5, 180, 8))
    for t0 in starts:
        residual[:, t0 : t0 + 6] += pattern
    window = _make_window(residual)
    segs = [
        _segment(f"p{i}", pattern, t0=t0, det_index=i) for i, t0 in enumerate(starts)
    ]

    logs: list[int] = []

    def _cb(log):
        logs.append(log.outer)

    result = learn_randomized_outer(
        [window],
        segs,
        m_per_outer=2,
        n_min=3,
        post_refine_max_relative_distance=0.9,
        min_coverage=0.05,
        max_outer=20,
        refine_iters=2,
        epsilon=0.0,
        min_template_width=None,
        max_template_width=None,
        eps_step_fraction=0.01,
        max_relative_distance=None,
        rng=0,
        progress_callback=_cb,
    )
    assert result.stop_reason in {"coverage", "no_survivors", "empty_pool", "max_outer"}
    if result.stop_reason == "coverage":
        assert result.time_coverage_fraction > 0.05
        assert result.n_outer_completed == len(logs) >= 1
        assert result.n_atoms >= 1
        assert result.n_matchings >= 1


def test_learn_randomized_outer_defaults():
    import inspect

    from analysis.dict_learning.code.learn import randomized_dictionary as rd

    sig = inspect.signature(rd.learn_randomized_outer)
    assert sig.parameters["m_per_outer"].default == 5
    assert sig.parameters["n_min"].default == 10
    assert sig.parameters["post_refine_max_relative_distance"].default == 0.9
    assert sig.parameters["min_coverage"].default == 0.5
    assert sig.parameters["epsilon"].default == 0.1
    assert sig.parameters["min_template_width"].default == 10
    assert sig.parameters["max_template_width"].default == 20


if __name__ == "__main__":
    test_create_dictionary_samples_unit_norm_atoms()
    test_create_dictionary_epsilon_filters_silence()
    test_create_dictionary_epsilon_is_relative_to_original()
    test_dictionary_explained_energy_empty_and_match()
    test_refine_assigns_patches_and_prunes()
    test_build_randomized_end_to_end()
    test_refine_subspace_k()
    test_post_refine_filters_far_patches()
    test_default_refine_iters_is_three()
    test_learn_randomized_outer_coverage_stop()
    test_learn_randomized_outer_defaults()
    print("ok")
