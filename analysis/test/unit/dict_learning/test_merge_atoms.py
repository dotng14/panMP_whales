"""Folding atoms that refine to the same shape (``merge_similar_atoms``) and the shifted cosine it uses."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.dict_learning.code.core import Dictionary, Matching, Template
from analysis.dict_learning.code.learn.randomized_dictionary import merge_similar_atoms, refine_dictionary
from analysis.dict_learning.code.learn.shape_match import best_shift
from analysis.test.synthetic import motif_shapes, planted_stack

T = 260


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def _shifted(shape: np.ndarray, by: int) -> np.ndarray:
    """The same call starting ``by`` columns later in the frame."""
    out = np.zeros_like(shape)
    out[:, by:] = shape[:, :-by]
    return out


def test_best_shift_gives_the_offset_between_copies():
    call = np.random.default_rng(0).standard_normal((4, 10))
    padded = np.concatenate([np.zeros((4, 3)), call, np.zeros((4, 5))], axis=1)
    assert best_shift(call, padded) == (pytest.approx(1.0), 3)
    assert best_shift(padded, call) == (pytest.approx(1.0), -3)


def test_a_shifted_twin_hands_its_placements_to_the_busier_atom():
    first, second = motif_shapes(12, 12, 2)
    a = Template(template_id=0, data=_unit(first))
    twin = Template(template_id=1, data=_unit(_shifted(first, 2)))
    other = Template(template_id=2, data=_unit(second))
    dictionary = Dictionary()
    for template in (a, twin, other):
        dictionary.add(template)

    # The twin holds the call 2 columns into its frame, so a twin placement at
    # t is a placement of ``a`` at t + 2.
    a_at, twin_at, other_at = [10, 50, 90, 200], [48, 130, T - 12], [30, 150]
    stack = {"w": 0.01 * np.random.default_rng(1).standard_normal((12, T))}
    for t in a_at + [t + 2 for t in twin_at[:2]]:
        stack["w"][:, t : t + 12] += first
    for t in other_at:
        stack["w"][:, t : t + 12] += second
    matchings = [
        Matching(template_id=tid, window_id="w", t_start=t, amplitude=1.0)
        for tid, starts in ((0, a_at), (1, twin_at), (2, other_at))
        for t in starts
    ]

    merged, out, n_merged = merge_similar_atoms(stack, dictionary, matchings, min_corr=0.9, n_components=1)

    assert n_merged == 1
    assert [int(t.template_id) for t in merged] == [0, 2]
    starts = {tid: sorted(m.t_start for m in out if m.template_id == tid) for tid in (0, 2)}
    # 48 + 2 is already one of a's placements; T - 12 + 2 would leave the window.
    assert starts == {0: [10, 50, 90, 132, 200], 2: other_at}
    rebuilt = {int(t.template_id): t for t in merged}
    assert abs(float(np.sum(rebuilt[0].data * _unit(first)))) > 0.99
    np.testing.assert_array_equal(rebuilt[2].data, other.data)


def test_distinct_atoms_are_left_alone():
    first, second = motif_shapes(12, 12, 2)
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=_unit(first)))
    dictionary.add(Template(template_id=1, data=_unit(second)))
    matchings = [Matching(template_id=1, window_id="w", t_start=5, amplitude=1.0)]
    merged, out, n_merged = merge_similar_atoms({"w": np.zeros((12, 40))}, dictionary, matchings, min_corr=0.9)
    assert n_merged == 0 and merged is dictionary and out == matchings


@pytest.mark.parametrize("merge_corr, n_atoms, n_merged", [(None, 3, 0), (0.9, 2, 1)])
def test_refine_folds_two_seeds_of_one_call(merge_corr, n_atoms, n_merged):
    planted = planted_stack(seed=0)
    seeds = Dictionary()
    for tid, shape in enumerate([planted.motifs[0], _shifted(planted.motifs[0], 1), planted.motifs[1]]):
        seeds.add(Template(template_id=tid, data=_unit(shape)))
    result = refine_dictionary(
        planted.stack, seeds, iter=1, n_components=1, n_min=2,
        post_refine_max_relative_distance=None, merge_corr=merge_corr, n_workers=1,
    )
    assert len(result.dictionary) == n_atoms and result.n_merged == n_merged
    ids = {int(t.template_id) for t in result.dictionary}
    assert {int(m.template_id) for m in result.matchings} <= ids
    if merge_corr is not None:
        # Re-deflated from the input with the folded placements.
        assert len(result.energies) == 3
        assert result.energies[-1] < result.energies[0]
