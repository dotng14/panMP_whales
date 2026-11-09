import numpy as np
from scipy.special import logsumexp

from analysis.song_exp3.models.graph import HMM, Categorical, DiagGMM, Family, Graph
from analysis.song_exp3.models.m1_theme import build_graph as m1_graph


def _toy():
    fam = {"a": Family("a", np.array([0.6, 0.4])), "b": Family("b", np.array([0.3, 0.7])), "s": Family("s", np.array([0.5, 0.5]))}
    g = Graph(3, np.array([0, 1, 2], np.int32), fam)
    g.add(0, 0, ("a", 0))
    g.add(0, 1, ("a", 1), ("b", 0))
    g.add(0, 2, ("a", 1), ("b", 1))
    g.add(1, 1, ("b", 1))
    g.add(1, 2, ("b", 0))
    g.add(2, 0, ("s", 0))
    g.add(2, 2, ("s", 1))
    g.add_start(0, ("s", 0))
    g.add_start(1, ("s", 1))
    return g


def test_toy_rows_sum_to_one():
    np.testing.assert_allclose(_toy().check(), 1.0)


def test_forward_and_viterbi_match_brute_force():
    g = _toy()
    emis = Categorical(np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])))
    hmm = HMM(g, emis)
    x = np.array([0, 1, 2, 2])
    src, dst, logp, ls = g.arrays()
    la = np.full((3, 3), -np.inf)
    la[src, dst] = logp
    lb = emis.logpdf(x)
    paths, scores = [], []
    for p in np.ndindex(3, 3, 3, 3):
        s = ls[p[0]] + lb[0, p[0]] + sum(la[p[i - 1], p[i]] + lb[i, p[i]] for i in range(1, 4))
        paths.append(p)
        scores.append(s)
    assert abs(hmm.loglik(x) - logsumexp(scores)) < 1e-9
    assert tuple(hmm.viterbi(x)) == paths[int(np.argmax(scores))]


def test_baum_welch_never_decreases_with_product_factors():
    rng = np.random.default_rng(0)
    g = _toy()
    x = rng.normal(0, 1, (400, 2)) + np.repeat(np.array([[0, 0], [3, 0], [0, 3]]), [150, 150, 100], axis=0)
    hmm = HMM(g, DiagGMM.from_assignment(x, np.repeat([0, 1, 2], [150, 150, 100]), 3))
    hist = hmm.fit([x], max_iter=15)
    assert np.all(np.diff(hist) > -1e-6 * abs(hist[0]))
    np.testing.assert_allclose(g.check(), 1.0, atol=1e-9)


def test_m1_graph_is_a_valid_cycle():
    g, theme_of = m1_graph()
    np.testing.assert_allclose(g.check(), 1.0, atol=1e-12)
    for s, d, _ in g.edges:
        a, b = theme_of[s], theme_of[d]
        if a >= 0 and b >= 0:
            assert b in (a, (a + 1) % 6)
