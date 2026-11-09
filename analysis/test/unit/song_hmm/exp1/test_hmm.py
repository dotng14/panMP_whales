import numpy as np
from scipy.special import logsumexp

from analysis.song_hmm.hmm import baum_welch, initial_params, log_emit, loglik, posterior, sample, viterbi
from analysis.song_hmm.topology import N_DUR, N_STATES, build_graph


def _params(seed=0, k=3):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((800, k))
    slots = rng.integers(0, 28, 800)
    hist = rng.integers(0, 20, (28, N_DUR)).astype(float)
    p = initial_params(x, slots, hist, level_col=0)
    p.means = rng.normal(0.0, 2.0, p.means.shape)
    # make every edge family non-trivial so brute force covers the whole graph
    p.bg = np.full(7, 1.0 / 7)
    p.branch[:] = 0.25
    p.pause[:] = 1.0 / 3
    return p


def _brute(x, p):
    """Score every 3-step path. Returns (log-likelihood, best path, best score)."""
    g = build_graph(p)
    with np.errstate(divide="ignore"):
        la = np.full((N_STATES, N_STATES), -np.inf)
        la[g.src, g.dst] = np.log(g.prob)
        ls = np.log(g.start)
    lb = log_emit(x, p)
    score = (ls + lb[0])[:, None, None] + la[:, :, None] + lb[1][None, :, None] + la[None, :, :] + lb[2][None, None, :]
    best = np.unravel_index(np.argmax(score), score.shape)
    return logsumexp(score), np.array(best), score[best]


def test_forward_matches_brute_force():
    p = _params()
    x = np.random.default_rng(1).standard_normal((3, 3))
    ll, _, _ = _brute(x, p)
    assert abs(loglik(x, p) - ll) < 1e-8
    assert abs(posterior(x, p).loglik - ll) < 1e-8


def test_viterbi_matches_brute_force():
    p = _params()
    x = np.random.default_rng(2).standard_normal((3, 3))
    _, best_path, best_score = _brute(x, p)
    path, score = viterbi(x, p)
    assert abs(score - best_score) < 1e-8
    np.testing.assert_array_equal(path, best_path)


def test_posterior_is_normalized():
    p = _params()
    x, _ = sample(p, 300, np.random.default_rng(3))
    post = posterior(x, p)
    np.testing.assert_allclose(post.gamma.sum(axis=1), 1.0, atol=1e-9)
    assert abs(post.edge_counts.sum() - (len(x) - 1)) < 1e-6


def test_baum_welch_never_decreases():
    rng = np.random.default_rng(4)
    true = _params(5)
    x, path = sample(true, 3000, rng)
    slots = np.where(path < 140, path // N_DUR, -1)
    start = initial_params(x, slots, np.ones((28, N_DUR)), level_col=0)
    _, history = baum_welch([x], start, max_iter=8, verbose=False)
    assert np.all(np.diff(history) > -1e-6 * abs(history[0])), history
