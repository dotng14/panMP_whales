"""Tests for whole-dictionary refinement (average every atom, then re-run MP)."""

from __future__ import annotations


import numpy as np


from analysis.dict_learning.code.learn.refine import (  # noqa: E402
    DictionaryLearningError,
    DictionaryRefinementResult,
    NoMatchingPatchesError,
    refine_dictionary_by_averaging,
)
from analysis.dict_learning.code.core import Dictionary, Matching, Segment, Template, Window  # noqa: E402


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


def _two_pattern_window() -> tuple[Window, Segment, Segment, np.ndarray, np.ndarray]:
    """Window containing two disjoint-frequency repeating patterns.

    Pattern A (rows 0-1, ±3, width 5) repeats 3×; pattern B (rows 2-3, ±2,
    width 4) repeats 2×. Disjoint rows keep the normalized correlation between
    the patterns at zero, and both patterns (hence the whole window) are
    zero-mean so mean-centering the stack is a no-op and exact
    deflation still drives the residual to zero.
    """
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

    window = _make_window(residual)
    seg_a = _segment("segA", pattern_a, t0=5, det_index=0)
    seg_b = _segment("segB", pattern_b, t0=70, det_index=1)
    return window, seg_a, seg_b, pattern_a, pattern_b


def test_refine_dictionary_by_averaging_repeats_multi_atom_mp():
    window, _seg_a, _seg_b, pattern_a, pattern_b = _two_pattern_window()
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=pattern_a.copy()))
    dictionary.add(Template(template_id=1, data=pattern_b.copy()))
    initial_matchings = [
        *(Matching(0, "w0", t0, 1.0) for t0 in (5, 25, 45)),
        *(Matching(1, "w0", t0, 1.0) for t0 in (70, 90)),
    ]

    result = refine_dictionary_by_averaging(
        [window],
        dictionary,
        initial_matchings,
        iter=2,
        eps_step_fraction=1e-12,
    )

    assert isinstance(result, DictionaryRefinementResult)
    assert result.iterations_completed == 2
    assert len(result.dictionary) == 2
    assert len(result.matchings) == 5
    assert {m.template_id for m in result.matchings} == {0, 1}
    assert np.allclose(result.dictionary.templates[0].data, pattern_a)
    assert np.allclose(result.dictionary.templates[1].data, pattern_b)
    assert result.energies[-1] < 1e-9 * result.energies[0]


def test_refine_dictionary_requires_matches_for_each_atom():
    window, _seg_a, _seg_b, pattern_a, pattern_b = _two_pattern_window()
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=pattern_a))
    dictionary.add(Template(template_id=1, data=pattern_b))

    try:
        refine_dictionary_by_averaging(
            [window],
            dictionary,
            [Matching(0, "w0", 5, 1.0)],
        )
    except NoMatchingPatchesError as exc:
        assert exc.template_id == 1
        assert "template_id=1" in str(exc)
        # Remains catchable via the base class and legacy ValueError.
        assert isinstance(exc, DictionaryLearningError)
        assert isinstance(exc, ValueError)
    else:
        raise AssertionError(
            "expected missing template matches to raise NoMatchingPatchesError"
        )


def _loud_and_faint_window() -> tuple[Window, np.ndarray, np.ndarray]:
    """Loud pattern repeated 3×, plus one faint pattern MP stops before reaching."""
    f, t = 4, 120
    loud = np.zeros((f, 5), dtype=np.float64)
    loud[0, :] = 3.0
    loud[1, :] = -3.0
    faint = np.zeros((f, 4), dtype=np.float64)
    faint[2, :] = 0.05
    faint[3, :] = -0.05

    data = np.zeros((f, t), dtype=np.float64)
    for t0 in (5, 25, 45):
        data[:, t0 : t0 + 5] += loud
    data[:, 70:74] += faint
    return _make_window(data), loud, faint


def _loud_faint_dictionary(loud: np.ndarray, faint: np.ndarray) -> Dictionary:
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=loud.copy()))
    dictionary.add(Template(template_id=1, data=faint.copy()))
    return dictionary


def test_refine_drops_faint_atom_without_pinning():
    window, loud, faint = _loud_and_faint_window()
    labels = [
        *(Matching(0, "w0", t0, 1.0) for t0 in (5, 25, 45)),
        Matching(1, "w0", 70, 1.0),
    ]

    # One pass: the eps stopping rule fires before the faint atom is placed, so
    # its labeled patch is silently dropped from the matchings.
    one_pass = refine_dictionary_by_averaging(
        [window], _loud_faint_dictionary(loud, faint), labels, iter=1
    )
    assert {m.template_id for m in one_pass.matchings} == {0}

    # Two passes: the now-unmatched atom has nothing to average.
    try:
        refine_dictionary_by_averaging(
            [window], _loud_faint_dictionary(loud, faint), labels, iter=2
        )
    except NoMatchingPatchesError as exc:
        assert exc.template_id == 1
    else:
        raise AssertionError("expected the starved faint atom to raise")


def test_pinned_matchings_keep_faint_atom_and_its_patch():
    window, loud, faint = _loud_and_faint_window()
    pinned = [Matching(1, "w0", 70, 1.0)]
    labels = [*(Matching(0, "w0", t0, 1.0) for t0 in (5, 25, 45)), *pinned]

    result = refine_dictionary_by_averaging(
        [window],
        _loud_faint_dictionary(loud, faint),
        labels,
        iter=2,
        pinned_matchings=pinned,
    )

    assert result.iterations_completed == 2
    assert result.n_pinned_matchings == 1
    assert len(result.dictionary) == 2
    kept = [m for m in result.matchings if m.template_id == 1]
    assert [m.t_start for m in kept] == [70]
    assert {m.template_id for m in result.matchings} == {0, 1}
    got = result.dictionary.templates[1].data.ravel()
    ref = faint.ravel()
    corr = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
    assert abs(corr) > 0.99


def test_pinned_matching_is_not_reassigned_to_a_competing_atom():
    """Two atoms of identical shape: MP hands every patch to the first one."""
    window, loud, _faint = _loud_and_faint_window()
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=loud.copy()))
    dictionary.add(Template(template_id=1, data=loud.copy()))
    pinned = [Matching(1, "w0", 45, 1.0)]
    labels = [*(Matching(0, "w0", t0, 1.0) for t0 in (5, 25)), *pinned]

    unpinned = refine_dictionary_by_averaging(
        [window], dictionary, labels, iter=1, eps_step_fraction=1e-12
    )
    assert (1, 45) not in {(m.template_id, m.t_start) for m in unpinned.matchings}

    result = refine_dictionary_by_averaging(
        [window],
        dictionary,
        labels,
        iter=1,
        eps_step_fraction=1e-12,
        pinned_matchings=pinned,
    )
    assert (1, 45) in {(m.template_id, m.t_start) for m in result.matchings}
    # The pinned span is off limits, so atom 0 cannot double-claim it.
    assert (0, 45) not in {(m.template_id, m.t_start) for m in result.matchings}


