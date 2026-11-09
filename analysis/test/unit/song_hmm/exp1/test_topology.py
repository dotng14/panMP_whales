import numpy as np

from analysis.song_hmm.exp1_slot_hmm.hmm import initial_params
from analysis.song_hmm.exp1_slot_hmm.topology import (
    BG,
    BG_FIRST,
    IS_BG,
    ENTRY_STATES,
    LAST_SLOT,
    N_DUR,
    N_STATES,
    N_THEMES,
    PAUSE0,
    STATE_SLOT,
    STATE_THEME,
    build_graph,
    duration_pmf,
    exit_prob_from_lengths,
    slot_state,
)


def _params(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((500, 4))
    slots = rng.integers(0, 28, 500)
    hist = rng.integers(0, 20, (28, N_DUR)).astype(float)
    return initial_params(x, slots, hist, level_col=0)


def test_rows_sum_to_one():
    g = build_graph(_params())
    totals = np.bincount(g.src, weights=g.prob, minlength=N_STATES)
    np.testing.assert_allclose(totals, 1.0, atol=1e-12)
    assert abs(g.start.sum() - 1.0) < 1e-12


def test_theme_order_is_a_cycle_without_skips():
    g = build_graph(_params())
    for s, d in zip(g.src, g.dst, strict=True):
        if IS_BG[s] or IS_BG[d]:
            continue
        a, b = STATE_THEME[s], STATE_THEME[d]
        assert b in (a, (a + 1) % N_THEMES), (s, d)


def test_only_theme6_enters_theme1_from_another_theme():
    g = build_graph(_params())
    into_t1 = g.dst == ENTRY_STATES[0]
    sources = {int(s) for s in g.src[into_t1] if STATE_THEME[s] not in (0, -1)}
    t6_last = {slot_state(int(LAST_SLOT[5]), d) for d in range(N_DUR)}
    assert sources <= t6_last | {PAUSE0 + 5}
    assert np.all(g.is_boundary == (into_t1 & np.isin(STATE_THEME[g.src], [5])))


def test_background_enters_only_first_slots_after_its_minimum():
    g = build_graph(_params())
    out = set(g.dst[g.src == BG].tolist())
    assert out == set(ENTRY_STATES.tolist()) | {BG}
    for s in range(BG_FIRST, BG):  # the chain only moves forward
        assert g.dst[g.src == s].tolist() == [s + 1]
    into_bg = set(g.dst[IS_BG[g.dst] & ~IS_BG[g.src]].tolist())
    assert into_bg == {BG_FIRST}


def test_only_last_slots_branch_out_of_a_theme():
    g = build_graph(_params())
    for s, d in zip(g.src, g.dst, strict=True):
        slot = STATE_SLOT[s]
        if slot < 0 or d == s + 1:
            continue
        if STATE_SLOT[d] != slot + 1:  # not a plain advance
            assert slot == LAST_SLOT[STATE_THEME[s]], (s, d)


def test_exit_prob_matches_length_histogram():
    hist = np.array([10.0, 40.0, 30.0, 15.0, 5.0])
    pmf = duration_pmf(exit_prob_from_lengths(hist))
    expected = (hist + 0.5) / (hist + 0.5).sum()
    np.testing.assert_allclose(pmf, expected, atol=1e-9)
