import numpy as np

from analysis.song_hmm.exp3.models import m3_units, m4_theme_hmms
from analysis.song_hmm.exp3.models.m2_songfirst import align, fit_template, name_phases, unit_rows


def _cyclic_song(rng, n_songs=8, P=60, d=12):
    """Songs made of a fixed sequence of P phase vectors, each song sung at its own tempo (0.7-1.4x)."""
    template = unit_rows(rng.normal(0, 1, (P, d)))
    rows, phase, starts = [], [], []
    for _ in range(n_songs):
        starts.append(len(rows))
        tempo = rng.uniform(0.7, 1.4)
        n = int(P / tempo)
        ph = np.minimum((np.arange(n) * tempo).astype(int), P - 1)
        rows.extend(template[ph] + rng.normal(0, 0.25, (n, d)))
        phase.extend(ph)
    return np.array(rows), np.array(phase), np.array(starts), template


def test_cyclic_alignment_recovers_song_starts_with_variable_tempo():
    rng = np.random.default_rng(0)
    x, phase, starts, _ = _cyclic_song(rng)
    u = unit_rows(x)
    template, _ = fit_template(u, 60, rounds=5)
    est, _ = align(u, template)
    # the template's phase 0 is arbitrary; what matters is that every song start lands on the same template phase
    at_start = est[starts[1:]]
    ref = at_start[0]
    circ = np.minimum((at_start - ref) % 60, (ref - at_start) % 60)
    assert np.all(circ <= 3), at_start
    # and one cycle per song: the estimated phase wraps once inside every song (its origin is mid-song)
    assert len(np.flatnonzero(np.diff(est) < -30)) == len(starts)


def test_phase_naming_finds_ordered_themes_including_short_ones():
    P = 60
    truth = np.repeat(np.arange(6), [2, 15, 12, 2, 14, 15])  # T1 and T4 are short
    truth = np.roll(truth, 7)
    counts = np.zeros((P, 6))
    counts[np.arange(P), truth] = 5.0
    counts += 0.2
    m, _, matched = name_phases(counts)
    np.testing.assert_array_equal(m, truth)
    assert matched > 0.8  # 5 of every 6.2 counts sit on the true theme


def _check_graph(g, theme_of):
    tot = g.check()
    has_out = np.bincount([e[0] for e in g.edges], minlength=g.n_states) > 0
    np.testing.assert_allclose(tot[has_out], 1.0, atol=1e-9)
    for s, d, _ in g.edges:
        a, b = theme_of[s], theme_of[d]
        if a >= 0 and b >= 0 and a != b:
            assert b == (a + 1) % 6
            if b == 0:
                assert a == 5


def test_m3_and_m4_graphs_follow_the_cycle():
    _check_graph(m3_units.build_graph(), m3_units.STATE_THEME)
    fam = m4_theme_hmms.families()
    _check_graph(m4_theme_hmms.song_graph(fam), m4_theme_hmms.STATE_THEME)


def test_m3_tokens_mark_breaths_and_session_gaps():
    u = np.zeros(4, dtype=[("start_s", "f8"), ("end_s", "f8")])
    u["start_s"] = [0, 2, 10, 100]
    u["end_s"] = [1, 3, 11, 101]
    tok, t0, _ = m3_units.token_sequence(u, np.array([0, 1, 2, 0]), 3)
    assert tok.tolist() == [0, 1, 3, 2, 4, 0]  # 7 s gap -> GAP (3), 89 s gap -> BGT (4)
