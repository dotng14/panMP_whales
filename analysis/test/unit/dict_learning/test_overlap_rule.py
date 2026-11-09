"""Limited overlap between placements (``mp.OverlapRule``, ``--overlap limited``)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from analysis.dict_learning.code.core import Dictionary, Matching, Template
from analysis.dict_learning.code.learn.randomized_dictionary import learn_randomized_outer, merge_similar_atoms
from analysis.dict_learning.code.learn.pan_seeds import PanSeedConfig, PanSeeder
from analysis.dict_learning.code.mp import OverlapRule, overlap_rule, run_mp_loop
from analysis.test.synthetic import motif_shapes, planted_stack

RULE = OverlapRule()


def _spans(matchings, widths):
    out: dict[str, list[tuple[int, int, int]]] = {}
    for m in matchings:
        t = int(m.t_start)
        out.setdefault(m.window_id, []).append((t, t + widths[int(m.template_id)], int(m.template_id)))
    return out


def _breaks_rule(matchings, widths) -> list[tuple]:
    bad = []
    for spans in _spans(matchings, widths).values():
        for (a0, a1, ta), (b0, b1, tb) in combinations(spans, 2):
            if not RULE.allows(a0, a1, b0, b1):
                bad.append(((a0, a1, ta), (b0, b1, tb)))
    return bad


@pytest.mark.parametrize(
    "a, b, allowed",
    [
        ((0, 12), (12, 24), True),  # side by side
        ((0, 12), (9, 21), True),  # 3 columns: a quarter of 12
        ((0, 12), (8, 20), False),  # 4 columns
        ((0, 12), (1, 13), False),  # the same call, one column on
        ((0, 12), (0, 12), False),  # stacked
        ((4, 10), (0, 12), True),  # a 6-wide unit inside a 12-wide phrase
        ((0, 12), (4, 10), True),  # and the other way round
        ((2, 9), (0, 12), False),  # 7 wide is more than half of 12
        ((8, 14), (0, 12), False),  # half outside
    ],
)
def test_allows(a, b, allowed):
    assert RULE.allows(*a, *b) is allowed


def test_forbid_agrees_with_allows_at_every_start():
    rng = np.random.default_rng(0)
    for _ in range(300):
        width = int(rng.integers(2, 30))
        start = int(rng.integers(0, 60))
        held = int(rng.integers(2, 30))
        valid = np.ones(100 - width + 1, dtype=bool)
        RULE.forbid(valid, width, start, start + held)
        expected = [RULE.allows(s, s + width, start, start + held) for s in range(len(valid))]
        np.testing.assert_array_equal(valid, expected)


def test_overlap_modes():
    assert overlap_rule("free") is None and overlap_rule(None) is None
    assert overlap_rule("limited") == OverlapRule()
    assert overlap_rule("limited", 0.0, 0.0) == OverlapRule(max_partial=0.0, max_nested=0.0)


def test_no_overlap_rule_allows_only_disjoint_spans():
    none = OverlapRule(max_partial=0.0, max_nested=0.0)
    assert none.allows(0, 12, 12, 24)
    assert not none.allows(0, 12, 11, 23)
    assert not none.allows(4, 10, 0, 24)  # not even a unit in a phrase
    valid = np.ones(50, dtype=bool)
    none.forbid(valid, 6, 20, 32)
    np.testing.assert_array_equal(np.flatnonzero(~valid), np.arange(15, 32))


def _one_call_window():
    """Two copies of one call 4 columns apart: free MP places both, limited one."""
    rng = np.random.default_rng(3)
    (shape,) = motif_shapes(12, 12, 1)
    window = 0.05 * rng.standard_normal((12, 120))
    window[:, 40:52] += 3.0 * shape
    window[:, 44:56] += 2.0 * shape
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=shape / np.linalg.norm(shape)))
    return {"w": window}, dictionary


def test_free_mp_stacks_one_call_and_limited_does_not():
    stack, dictionary = _one_call_window()
    free = run_mp_loop(stack, dictionary, eps_step_fraction=0.01, mp_mode="per_window", n_workers=1)
    limited = run_mp_loop(
        stack, dictionary, eps_step_fraction=0.01, mp_mode="per_window", n_workers=1, overlap=RULE
    )
    assert _breaks_rule(free.matchings, {0: 12})
    assert not _breaks_rule(limited.matchings, {0: 12})
    assert len([m for m in limited.matchings if 30 <= m.t_start <= 50]) == 1


def test_cached_window_loop_matches_the_full_rescan_with_the_rule():
    planted = planted_stack(seed=2, n_windows=1, counts=(3, 2, 2), amplitudes=(2.0, 1.0, 1.0))
    dictionary = Dictionary()
    for tid, shape in enumerate(planted.motifs):
        dictionary.add(Template(template_id=tid, data=shape / np.linalg.norm(shape)))
    narrow = planted.motifs[0][:, 3:9]
    dictionary.add(Template(template_id=3, data=narrow / np.linalg.norm(narrow)))
    kwargs = dict(eps_step_fraction=0.01, n_workers=1, overlap=RULE, fixed_spans={"w0": [(5, 17)]})
    cached = run_mp_loop(planted.stack, dictionary, mp_mode="per_window", **kwargs)
    rescan = run_mp_loop(planted.stack, dictionary, mp_mode="global", **kwargs)
    key = lambda m: (m.template_id, m.t_start)  # noqa: E731
    assert sorted(map(key, cached.matchings)) == sorted(map(key, rescan.matchings))
    widths = {0: 12, 1: 12, 2: 12, 3: 6}
    assert not _breaks_rule(cached.matchings, widths)
    # Nothing sits on more than a quarter of the earlier span [5, 17).
    assert all(RULE.allows(m.t_start, m.t_start + widths[m.template_id], 5, 17) for m in cached.matchings)


def test_outer_loop_keeps_every_pass_within_the_rule():
    planted = planted_stack(seed=1)
    result = learn_randomized_outer(
        planted.stack, None, seeder=PanSeeder(PanSeedConfig(lengths=(12, 6), start_freq=10.0, end_freq=130.0)),
        merge_corr=0.9, m_per_outer=3, n_min=2, min_coverage=1.0, max_outer=2, refine_iters=1, n_workers=1,
        overlap=RULE,
    )
    assert result.n_outer_completed == 2 and result.matchings
    widths = {int(t.template_id): int(t.width) for t in result.dictionary}
    assert not _breaks_rule(result.matchings, widths)


def test_merge_drops_a_moved_placement_the_rule_forbids():
    (shape,) = motif_shapes(12, 12, 1)
    unit = shape / np.linalg.norm(shape)
    twin = np.zeros_like(unit)
    twin[:, 1:] = unit[:, :-1]
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=unit))
    dictionary.add(Template(template_id=1, data=twin / np.linalg.norm(twin)))
    stack = {"w": np.zeros((12, 100))}
    for t in (10, 40, 70):
        stack["w"][:, t : t + 12] += shape
    # The twin's placement at 43 moves to 44: 6 columns into the kept one at 40.
    matchings = [Matching(template_id=0, window_id="w", t_start=t, amplitude=1.0) for t in (10, 40, 70)]
    matchings.append(Matching(template_id=1, window_id="w", t_start=43, amplitude=1.0))
    for overlap, starts in ((None, [10, 40, 44, 70]), (RULE, [10, 40, 70])):
        _merged, out, n_merged = merge_similar_atoms(
            stack, dictionary, matchings, min_corr=0.9, n_components=1, overlap=overlap
        )
        assert n_merged == 1 and sorted(m.t_start for m in out) == starts
