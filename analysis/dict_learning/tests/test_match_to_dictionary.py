"""Tests for match_to_dictionary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.core import Matching, Window  # noqa: E402
from analysis.dict_learning.code.learn.learn_dictionary import (  # noqa: E402
    DictionaryRefinementResult,
    NoMatchingPatchesError,
)
from analysis.dict_learning.code.learn.match_to_dictionary import (  # noqa: E402
    match_to_dictionary,
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


def _two_pattern_window() -> tuple[Window, np.ndarray, np.ndarray]:
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
    return _make_window(residual), pattern_a, pattern_b


def _planted_matchings() -> list[Matching]:
    return [
        *(Matching(0, "w0", t0, 1.0) for t0 in (5, 25, 45)),
        *(Matching(1, "w0", t0, 1.0) for t0 in (70, 90)),
    ]


def test_match_to_dictionary_recovers_two_templates():
    window, pattern_a, pattern_b = _two_pattern_window()
    result = match_to_dictionary(
        [window],
        _planted_matchings(),
        widths={0: 5, 1: 4},
        refine_iters=2,
        eps_step_fraction=1e-12,
        show_progress=False,
    )
    assert isinstance(result, DictionaryRefinementResult)
    assert len(result.dictionary) == 2
    assert result.dictionary.templates[0].width == 5
    assert result.dictionary.templates[1].width == 4
    assert result.dictionary.templates[0].n_freq == 4
    assert result.iterations_completed == 2
    # Sign-agnostic correlation: PCA orients toward the mean patch.
    for tid, pattern in ((0, pattern_a), (1, pattern_b)):
        got = result.dictionary.templates[tid].data.ravel()
        ref = pattern.ravel()
        corr = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
        assert abs(corr) > 0.99
    assert {m.template_id for m in result.matchings} == {0, 1}
    assert result.energies[-1] < result.energies[0]


def test_match_to_dictionary_shared_width_int():
    window, pattern_a, _pattern_b = _two_pattern_window()
    # Only atom 0 (width 5) so a shared width is valid.
    matchings = [Matching(0, "w0", t0, 1.0) for t0 in (5, 25, 45)]
    result = match_to_dictionary(
        [window],
        matchings,
        widths=5,
        refine_iters=0,
        show_progress=False,
    )
    assert len(result.dictionary) == 1
    assert result.dictionary.templates[0].width == 5
    assert result.iterations_completed == 0
    assert len(result.matchings) == 3
    got = result.dictionary.templates[0].data.ravel()
    ref = pattern_a.ravel()
    corr = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
    assert abs(corr) > 0.99


def test_match_to_dictionary_refine_iters_zero_keeps_input_matchings():
    window, _, _ = _two_pattern_window()
    matchings = _planted_matchings()
    result = match_to_dictionary(
        [window],
        matchings,
        widths={0: 5, 1: 4},
        refine_iters=0,
        show_progress=False,
    )
    assert result.iterations_completed == 0
    assert len(result.matchings) == len(matchings)
    assert {(m.template_id, m.t_start) for m in result.matchings} == {
        (m.template_id, m.t_start) for m in matchings
    }
    assert len(result.energies) == 1


def _loud_and_faint_labels() -> tuple[Window, list[Matching], dict[int, int]]:
    """Loud label repeated 3×, plus a faint one MP's eps rule stops before."""
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
    labels = [
        *(Matching(0, "w0", t0, 1.0) for t0 in (5, 25, 45)),
        Matching(1, "w0", 70, 1.0),
    ]
    return _make_window(data), labels, {0: 5, 1: 4}


def test_refine_keeps_labels_by_default():
    window, labels, widths = _loud_and_faint_labels()
    result = match_to_dictionary(
        [window], labels, widths=widths, refine_iters=2, show_progress=False
    )
    assert result.n_pinned_matchings == len(labels)
    assert {(m.template_id, m.t_start) for m in labels}.issubset(
        {(m.template_id, m.t_start) for m in result.matchings}
    )


def test_no_pin_labels_lets_refine_reassign_and_starve():
    window, labels, widths = _loud_and_faint_labels()
    unpinned = match_to_dictionary(
        [window],
        labels,
        widths=widths,
        refine_iters=1,
        pin_labels=False,
        show_progress=False,
    )
    assert {m.template_id for m in unpinned.matchings} == {0}
    with pytest.raises(NoMatchingPatchesError):
        match_to_dictionary(
            [window],
            labels,
            widths=widths,
            refine_iters=2,
            pin_labels=False,
            show_progress=False,
        )


def test_match_to_dictionary_missing_width_raises():
    window, _, _ = _two_pattern_window()
    with pytest.raises(ValueError, match="missing width for template_id=1"):
        match_to_dictionary(
            [window],
            _planted_matchings(),
            widths={0: 5},
            refine_iters=0,
            show_progress=False,
        )


def test_match_to_dictionary_empty_matchings_raises():
    window, _, _ = _two_pattern_window()
    with pytest.raises(ValueError, match="matchings must be non-empty"):
        match_to_dictionary(
            [window],
            [],
            widths=5,
            refine_iters=0,
            show_progress=False,
        )


def test_match_to_dictionary_invalid_refine_iters_raises():
    window, _, _ = _two_pattern_window()
    with pytest.raises(ValueError, match="refine_iters"):
        match_to_dictionary(
            [window],
            _planted_matchings(),
            widths={0: 5, 1: 4},
            refine_iters=-1,
            show_progress=False,
        )
