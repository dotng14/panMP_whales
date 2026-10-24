"""Topology and Baum–Welch checks that do not need the 10-hour recording."""

from __future__ import annotations

import numpy as np

from analysis.HMM.hmm import SoftHMM, baum_welch, fit_metrics, sample_sequence
from analysis.HMM.topology import (
    N_SLOTS,
    N_STATES,
    N_THEMES,
    REPEAT_INIT,
    SLOT_THEME,
    START_STATE,
    exit_probs_from_duration_hist,
    row_probability_sums,
    transition_tables,
)


def test_duration_chain_matches_a_two_second_histogram():
    exit_prob = exit_probs_from_duration_hist(np.array([0.0, 1.0, 0.0, 0.0, 0.0]))
    stay = 1.0 - exit_prob[0]
    leave_at_2 = exit_prob[1]
    mass_2 = stay * leave_at_2
    assert mass_2 > 0.99


def test_rows_sum_to_one_and_theme_6_does_not_advance():
    exit_prob = np.full((N_SLOTS, 4), 0.4)
    tables = transition_tables(exit_prob, REPEAT_INIT)
    sums = row_probability_sums(tables)
    np.testing.assert_allclose(sums, 1.0, atol=1e-8)
    assert tables.n_edges > N_STATES
    from analysis.HMM.topology import KIND_NEXT, TERMINAL_SLOT

    src = int(TERMINAL_SLOT[-1]) * 5
    kinds = [int(tables.kind[src, k]) for k in range(int(tables.n_out[src]))]
    assert int(KIND_NEXT) not in kinds
    assert START_STATE == 0
    from analysis.HMM.topology import START_STATES

    assert list(START_STATES) == [0, 15, 30, 60, 75, 100]


def test_baum_welch_raises_likelihood():
    rng = np.random.default_rng(0)
    k_dim = 2
    means = rng.normal(scale=3.0, size=(N_SLOTS, k_dim))
    variances = np.full((N_SLOTS, k_dim), 0.05)
    exit_prob = np.full((N_SLOTS, 4), 0.35)
    model = SoftHMM(
        means,
        variances,
        exit_prob,
        REPEAT_INIT.copy(),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    sequences = []
    for _ in range(2):
        drawn, path = sample_sequence(model, 60, rng)
        sequences.append(drawn)
        assert int(SLOT_THEME[int(path[0]) // 5]) == 0
    start = model.copy()
    start.means = means + rng.normal(scale=1.0, size=means.shape)
    start.repeat_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
    fitted, history, complete = baum_welch(sequences, start, max_iter=4, tol=0.0)
    assert history[-1] > history[0]
    assert np.isfinite(complete)
    themes = [np.ones(len(sequence), dtype=np.int32) for sequence in sequences]
    metrics = fit_metrics(sequences, themes, fitted)
    assert np.isfinite(metrics["posterior_on_label"])
    assert metrics["song_loglik"].shape == (len(sequences),)
    assert fitted.repeat_prob[-1] == 1.0
    assert np.all(fitted.start_prob > 0.0)
    assert abs(float(fitted.start_prob.sum()) - 1.0) < 1e-8
    assert N_THEMES == 6


def test_sample_can_open_on_theme_3():
    rng = np.random.default_rng(1)
    means = np.zeros((N_SLOTS, 2))
    means[6] = np.array([8.0, 0.0])
    variances = np.full((N_SLOTS, 2), 0.05)
    exit_prob = np.full((N_SLOTS, 4), 0.2)
    model = SoftHMM(
        means,
        variances,
        exit_prob,
        REPEAT_INIT.copy(),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    _drawn, path = sample_sequence(model, 30, rng)
    assert int(SLOT_THEME[int(path[0]) // 5]) == 2
