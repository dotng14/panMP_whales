import numpy as np
from scipy.special import logsumexp

from analysis.song_hmm.exp2_phrase_hmm.hmm import baum_welch, log_emit, loglik, params_from_assignment, posterior, sample, viterbi
from analysis.song_hmm.exp2_phrase_hmm.topology import (
    BG,
    BG_FIRST,
    DWELL,
    ENTRY,
    IS_BG,
    MIN_BG,
    MIN_THEME,
    N_STATES,
    N_THEMES,
    PAUSE0,
    STATE_THEME,
    build_graph,
)


def _params(seed=0, d=3):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((600, d))
    p = params_from_assignment(x, rng.integers(0, N_THEMES, 600), rng.random(600) < 0.1)
    p.means = rng.normal(0.0, 2.0, p.means.shape)
    p.bg[:] = 1.0 / 7
    p.dwell[:] = 0.25
    p.pause[:] = 1.0 / 3
    return p


def test_rows_sum_to_one():
    g = build_graph(_params())
    np.testing.assert_allclose(np.bincount(g.src, weights=g.prob, minlength=N_STATES), 1.0, atol=1e-12)
    assert abs(g.start.sum() - 1) < 1e-12


def test_cycle_without_skips_and_only_theme6_enters_theme1():
    g = build_graph(_params())
    for s, d in zip(g.src, g.dst, strict=True):
        if IS_BG[s] or IS_BG[d]:
            continue
        a, b = STATE_THEME[s], STATE_THEME[d]
        assert b in (a, (a + 1) % N_THEMES)
        if d == ENTRY[0] and a != 0:
            assert s in (DWELL[5], PAUSE0 + 5)


def test_minimum_lengths():
    g = build_graph(_params())
    for th in range(N_THEMES):
        for d in range(MIN_THEME - 1):  # chain states only move forward
            assert g.dst[g.src == ENTRY[th] + d].tolist() == [ENTRY[th] + d + 1]
    for s in range(BG_FIRST, BG):
        assert g.dst[g.src == s].tolist() == [s + 1]
    assert set(g.dst[IS_BG[g.dst] & ~IS_BG[g.src]].tolist()) == {BG_FIRST}
    assert set(g.dst[g.src == BG].tolist()) == set(ENTRY.tolist()) | {BG}
    assert MIN_BG == BG - BG_FIRST + 1


def _brute(x, p):
    g = build_graph(p)
    with np.errstate(divide="ignore"):
        la = np.full((N_STATES, N_STATES), -np.inf)
        la[g.src, g.dst] = np.log(g.prob)
        ls = np.log(g.start)
    lb = log_emit(x, p)
    sc = (ls + lb[0])[:, None, None] + la[:, :, None] + lb[1][None, :, None] + la[None] + lb[2][None, None, :]
    best = np.unravel_index(np.argmax(sc), sc.shape)
    return logsumexp(sc), np.array(best), sc[best]


def test_forward_and_viterbi_match_brute_force():
    p = _params()
    x = np.random.default_rng(1).standard_normal((3, 3))
    ll, best, best_score = _brute(x, p)
    assert abs(loglik(x, p) - ll) < 1e-8
    assert abs(posterior(x, p).loglik - ll) < 1e-8
    path, score = viterbi(x, p)
    assert abs(score - best_score) < 1e-8
    np.testing.assert_array_equal(path, best)


def test_baum_welch_never_decreases():
    true = _params(5)
    x, path = sample(true, 3000, np.random.default_rng(2))
    start = params_from_assignment(x, np.maximum(STATE_THEME[path], 0), IS_BG[path])
    _, hist = baum_welch([x], start, max_iter=8, verbose=False)
    assert np.all(np.diff(hist) > -1e-6 * abs(hist[0])), hist
