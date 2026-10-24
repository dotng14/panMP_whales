"""Tied-emission Baum–Welch on the soft song graph.

The five duration states of a phrase slot share one diagonal Gaussian.
Forward–backward walks the sparse edge list from ``topology``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .topology import (
    KIND_ADVANCE,
    KIND_CONTINUE,
    KIND_NEXT,
    KIND_REPEAT,
    N_DUR,
    N_SLOTS,
    N_STATES,
    N_THEMES,
    SLOT_THEME,
    START_STATES,
    TERMINAL_SLOT,
    TransitionTables,
    transition_tables,
)

VAR_FLOOR = 1e-3
PROB_CLIP = 1e-4


@dataclass
class SoftHMM:
    means: np.ndarray  # (n_slots, K)
    vars: np.ndarray  # (n_slots, K)
    exit_prob: np.ndarray  # (n_slots, 4)
    repeat_prob: np.ndarray  # (6,)
    start_prob: np.ndarray  # (6,) probability of entering each theme

    def tables(self) -> TransitionTables:
        return transition_tables(self.exit_prob, self.repeat_prob)

    def copy(self) -> "SoftHMM":
        return SoftHMM(
            self.means.copy(),
            self.vars.copy(),
            self.exit_prob.copy(),
            self.repeat_prob.copy(),
            self.start_prob.copy(),
        )


def log_emit(x: np.ndarray, model: SoftHMM) -> np.ndarray:
    """Log diagonal-Gaussian density, repeated across the five duration states.

    ``x`` is (T, K). The result is (T, N_STATES).
    """
    x = np.asarray(x, dtype=np.float64)
    means = np.asarray(model.means, dtype=np.float64)
    var = np.maximum(np.asarray(model.vars, dtype=np.float64), VAR_FLOOR)
    diff = x[:, None, :] - means[None, :, :]
    quad = np.sum(diff * diff / var[None, :, :], axis=2)
    logdet = np.sum(np.log(2.0 * np.pi * var), axis=1)
    log_slot = -0.5 * (logdet[None, :] + quad)
    return np.repeat(log_slot, N_DUR, axis=1)


def start_logprob(model: SoftHMM) -> np.ndarray:
    """Log probability of beginning in each state. Only theme entries are finite."""
    logp = np.full(N_STATES, -np.inf, dtype=np.float64)
    for theme, state in enumerate(START_STATES):
        prob = float(model.start_prob[theme])
        if prob > 0.0:
            logp[int(state)] = np.log(prob)
    return logp


@njit(cache=True)
def _logsumexp(values: np.ndarray) -> float:
    m = -np.inf
    for i in range(values.shape[0]):
        if values[i] > m:
            m = values[i]
    if not np.isfinite(m):
        return -np.inf
    acc = 0.0
    for i in range(values.shape[0]):
        if np.isfinite(values[i]):
            acc += np.exp(values[i] - m)
    return m + np.log(acc)


@njit(cache=True)
def _forward(logb, to, logp, n_out, log_start):
    t_len, n_states = logb.shape
    log_alpha = np.empty((t_len, n_states))
    for i in range(n_states):
        if np.isfinite(log_start[i]):
            log_alpha[0, i] = log_start[i] + logb[0, i]
        else:
            log_alpha[0, i] = -np.inf
    for t in range(1, t_len):
        for j in range(n_states):
            log_alpha[t, j] = -np.inf
        for i in range(n_states):
            ai = log_alpha[t - 1, i]
            if not np.isfinite(ai):
                continue
            n_k = n_out[i]
            for k in range(n_k):
                j = to[i, k]
                log_alpha[t, j] = np.logaddexp(log_alpha[t, j], ai + logp[i, k])
        for j in range(n_states):
            if np.isfinite(log_alpha[t, j]):
                log_alpha[t, j] += logb[t, j]
    return log_alpha


@njit(cache=True)
def _e_step(logb, x, log_alpha, to, logp, n_out, edge_id, slot_theme, loglik):
    """Posterior slot moments, expected edge counts, and per-second theme mass."""
    t_len, n_states = logb.shape
    k_dim = x.shape[1]
    n_edges = 0
    for i in range(n_states):
        for k in range(n_out[i]):
            eid = edge_id[i, k]
            if eid + 1 > n_edges:
                n_edges = eid + 1
    slot_occ = np.zeros(N_SLOTS)
    slot_x = np.zeros((N_SLOTS, k_dim))
    slot_x2 = np.zeros((N_SLOTS, k_dim))
    edge_count = np.zeros(n_edges)
    theme_post = np.zeros((t_len, N_THEMES))
    emit_complete = 0.0
    trans_complete = 0.0

    log_beta = np.zeros(n_states)
    for i in range(n_states):
        log_gamma = log_alpha[t_len - 1, i] - loglik
        if not np.isfinite(log_gamma) or log_gamma <= -700.0:
            continue
        g = np.exp(log_gamma)
        slot = i // N_DUR
        slot_occ[slot] += g
        for d in range(k_dim):
            slot_x[slot, d] += g * x[t_len - 1, d]
            slot_x2[slot, d] += g * x[t_len - 1, d] * x[t_len - 1, d]
        theme_post[t_len - 1, slot_theme[slot]] += g
        emit_complete += g * logb[t_len - 1, i]

    for t in range(t_len - 2, -1, -1):
        new_beta = np.empty(n_states)
        for i in range(n_states):
            new_beta[i] = -np.inf
            ai = log_alpha[t, i]
            n_k = n_out[i]
            for k in range(n_k):
                j = to[i, k]
                beta_term = logp[i, k] + logb[t + 1, j] + log_beta[j]
                new_beta[i] = np.logaddexp(new_beta[i], beta_term)
                if np.isfinite(ai) and np.isfinite(log_beta[j]):
                    log_xi = ai + beta_term - loglik
                    if log_xi > -700.0:
                        edge_count[edge_id[i, k]] += np.exp(log_xi)
                        trans_complete += np.exp(log_xi) * logp[i, k]
        log_beta = new_beta
        for i in range(n_states):
            log_gamma = log_alpha[t, i] + log_beta[i] - loglik
            if not np.isfinite(log_gamma) or log_gamma <= -700.0:
                continue
            g = np.exp(log_gamma)
            slot = i // N_DUR
            slot_occ[slot] += g
            for d in range(k_dim):
                slot_x[slot, d] += g * x[t, d]
                slot_x2[slot, d] += g * x[t, d] * x[t, d]
            theme_post[t, slot_theme[slot]] += g
            emit_complete += g * logb[t, i]
    return slot_occ, slot_x, slot_x2, edge_count, emit_complete, trans_complete, theme_post


def sequence_loglik(x: np.ndarray, model: SoftHMM, tables: TransitionTables | None = None) -> float:
    tables = model.tables() if tables is None else tables
    logb = np.ascontiguousarray(log_emit(x, model))
    log_alpha = _forward(
        logb,
        tables.to,
        tables.logp,
        tables.n_out,
        np.ascontiguousarray(start_logprob(model)),
    )
    return float(_logsumexp(log_alpha[-1]))


def total_loglik(sequences: list[np.ndarray], model: SoftHMM) -> float:
    tables = model.tables()
    total = 0.0
    for x in sequences:
        total += sequence_loglik(x, model, tables)
    return total


def _m_step(
    model: SoftHMM,
    tables: TransitionTables,
    slot_occ: np.ndarray,
    slot_x: np.ndarray,
    slot_x2: np.ndarray,
    edge_count: np.ndarray,
    start_count: np.ndarray,
) -> SoftHMM:
    means = model.means.copy()
    var = model.vars.copy()
    for slot in range(N_SLOTS):
        w = float(slot_occ[slot])
        if w < 1e-3:
            continue
        means[slot] = slot_x[slot] / w
        second = slot_x2[slot] / w
        var[slot] = np.maximum(second - means[slot] ** 2, VAR_FLOOR)

    exit_prob = model.exit_prob.copy()
    cont = np.zeros((N_SLOTS, N_DUR - 1), dtype=np.float64)
    adv = np.zeros((N_SLOTS, N_DUR - 1), dtype=np.float64)
    rep = np.zeros(N_THEMES, dtype=np.float64)
    nxt = np.zeros(N_THEMES, dtype=np.float64)

    for src in range(N_STATES):
        slot = src // N_DUR
        dur = src % N_DUR
        theme = int(SLOT_THEME[slot])
        for k in range(int(tables.n_out[src])):
            count = float(edge_count[tables.edge_id[src, k]])
            edge_kind = int(tables.kind[src, k])
            if edge_kind == int(KIND_CONTINUE) and dur < N_DUR - 1:
                cont[slot, dur] += count
            elif edge_kind == int(KIND_ADVANCE) and dur < N_DUR - 1:
                adv[slot, dur] += count
            elif edge_kind == int(KIND_REPEAT):
                rep[theme] += count
                if dur < N_DUR - 1:
                    adv[slot, dur] += count
            elif edge_kind == int(KIND_NEXT):
                nxt[theme] += count
                if dur < N_DUR - 1:
                    adv[slot, dur] += count

    for slot in range(N_SLOTS):
        for dur in range(N_DUR - 1):
            denom = cont[slot, dur] + adv[slot, dur]
            if denom < 1e-8:
                continue
            exit_prob[slot, dur] = float(np.clip(adv[slot, dur] / denom, PROB_CLIP, 1.0 - PROB_CLIP))

    repeat_prob = model.repeat_prob.copy()
    for theme in range(N_THEMES):
        denom = rep[theme] + nxt[theme]
        if denom < 1e-8:
            continue
        repeat_prob[theme] = float(np.clip(rep[theme] / denom, PROB_CLIP, 1.0 - PROB_CLIP))
    # One pseudocount per theme, so a theme the training songs never open can still start a new song.
    start_prob = np.asarray(start_count, dtype=np.float64) + 1.0
    start_prob = start_prob / start_prob.sum()
    return SoftHMM(means, var, exit_prob, repeat_prob, start_prob)


def e_step_sequence(x: np.ndarray, model: SoftHMM, tables: TransitionTables):
    logb = np.ascontiguousarray(log_emit(x, model))
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    log_alpha = _forward(
        logb,
        tables.to,
        tables.logp,
        tables.n_out,
        np.ascontiguousarray(start_logprob(model)),
    )
    loglik = float(_logsumexp(log_alpha[-1]))
    if not np.isfinite(loglik):
        raise RuntimeError("forward likelihood is not finite")
    stats = _e_step(
        logb,
        x64,
        log_alpha,
        tables.to,
        tables.logp,
        tables.n_out,
        tables.edge_id,
        SLOT_THEME,
        np.float64(loglik),
    )
    return loglik, stats


def baum_welch(
    sequences: list[np.ndarray],
    model: SoftHMM,
    *,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> tuple[SoftHMM, list[float], float]:
    """Fit shared parameters.

    Returns the model, the marginal log-likelihood after each iteration, and
    the posterior-weighted complete-data log-likelihood of the last E-step.
    """
    history: list[float] = []
    complete = 0.0
    current = model
    prev = -np.inf
    for _ in range(max_iter):
        tables = current.tables()
        k_dim = sequences[0].shape[1]
        slot_occ = np.zeros(N_SLOTS, dtype=np.float64)
        slot_x = np.zeros((N_SLOTS, k_dim), dtype=np.float64)
        slot_x2 = np.zeros((N_SLOTS, k_dim), dtype=np.float64)
        edge_count = np.zeros(tables.n_edges, dtype=np.float64)
        start_count = np.zeros(N_THEMES, dtype=np.float64)
        total_ll = 0.0
        emit_c = 0.0
        trans_c = 0.0
        for x in sequences:
            ll, stats = e_step_sequence(x, current, tables)
            total_ll += ll
            slot_occ += stats[0]
            slot_x += stats[1]
            slot_x2 += stats[2]
            edge_count += stats[3]
            emit_c += float(stats[4])
            trans_c += float(stats[5])
            start_count += stats[6][0]
        history.append(total_ll)
        log_pi = np.zeros(N_THEMES, dtype=np.float64)
        for theme in range(N_THEMES):
            if start_count[theme] > 0.0 and current.start_prob[theme] > 0.0:
                log_pi[theme] = np.log(current.start_prob[theme])
        complete = emit_c + trans_c + float(np.dot(start_count, log_pi))
        current = _m_step(current, tables, slot_occ, slot_x, slot_x2, edge_count, start_count)
        if np.isfinite(prev) and (total_ll - prev) <= tol * max(1.0, abs(prev)):
            break
        prev = total_ll
    return current, history, complete


def theme_posterior(x: np.ndarray, model: SoftHMM) -> np.ndarray:
    """Per-second posterior mass of each theme, shape (T, 6)."""
    tables = model.tables()
    _, stats = e_step_sequence(x, model, tables)
    return stats[6]


def _theme_runs(theme: np.ndarray) -> np.ndarray:
    counts = np.zeros(N_THEMES, dtype=np.float64)
    if len(theme) == 0:
        return counts
    first = int(theme[0])
    if first > 0:
        counts[first - 1] = 1.0
    for previous, current in zip(theme[:-1], theme[1:], strict=True):
        if int(current) > 0 and int(current) != int(previous):
            counts[int(current) - 1] += 1.0
    return counts


def fit_metrics(
    sequences: list[np.ndarray],
    themes: list[np.ndarray],
    model: SoftHMM,
) -> dict[str, np.ndarray]:
    """Likelihood and posterior summaries for a fitted model.

    ``themes[i]`` is the 1-based CSV theme at each second of ``sequences[i]``.
    The posterior is the forward–backward marginal, not a decoded path.
    """
    tables = model.tables()
    n_songs = len(sequences)
    song_loglik = np.empty(n_songs, dtype=np.float64)
    song_seconds = np.empty(n_songs, dtype=np.float64)
    expected = np.zeros(N_THEMES, dtype=np.float64)
    labeled = np.zeros(N_THEMES, dtype=np.float64)
    on_label = np.zeros(N_THEMES, dtype=np.float64)
    runs = np.zeros(N_THEMES, dtype=np.float64)
    slot_occ = np.zeros(N_SLOTS, dtype=np.float64)
    emit = 0.0
    trans = 0.0
    start_count = np.zeros(N_THEMES, dtype=np.float64)
    for index, (x, theme) in enumerate(zip(sequences, themes, strict=True)):
        ll, stats = e_step_sequence(x, model, tables)
        song_loglik[index] = ll
        song_seconds[index] = len(x)
        posterior = np.asarray(stats[6], dtype=np.float64)
        expected += posterior.sum(axis=0)
        slot_occ += np.asarray(stats[0], dtype=np.float64)
        emit += float(stats[4])
        trans += float(stats[5])
        start_count += np.asarray(posterior[0], dtype=np.float64)
        runs += _theme_runs(theme)
        for second, label in enumerate(theme):
            theme_id = int(label) - 1
            if theme_id < 0:
                continue
            labeled[theme_id] += 1.0
            on_label[theme_id] += posterior[second, theme_id]
    k_dim = int(sequences[0].shape[1])
    normal_per_second = -0.5 * k_dim * np.log(2.0 * np.pi)
    return {
        "song_loglik": song_loglik,
        "song_seconds": song_seconds,
        "emission_complete": np.array(emit),
        "transition_complete": np.array(trans),
        "start_count": start_count,
        "theme_expected_seconds": expected,
        "theme_labeled_seconds": labeled,
        "posterior_on_label_by_theme": on_label / np.maximum(labeled, 1.0),
        "posterior_on_label": np.array(float(on_label.sum() / max(labeled.sum(), 1.0))),
        "labeled_runs": runs,
        "slot_occupancy": slot_occ,
        "normal_per_second": np.array(normal_per_second),
    }


def parameter_sensitivity(
    sequences: list[np.ndarray],
    model: SoftHMM,
    *,
    branch_step: float = 0.05,
) -> dict[str, np.ndarray]:
    """One-at-a-time nudges. No refit. Reports the drop in sequence log-likelihood."""
    base = total_loglik(sequences, model)
    branch_drop = []
    for theme in range(N_THEMES):
        for sign in (1.0, -1.0):
            nudged = model.copy()
            nudged.repeat_prob = model.repeat_prob.copy()
            nudged.repeat_prob[theme] = float(
                np.clip(model.repeat_prob[theme] + sign * branch_step, PROB_CLIP, 1.0 - PROB_CLIP)
            )
            branch_drop.append(base - total_loglik(sequences, nudged))
    mean_drop = np.empty(N_SLOTS, dtype=np.float64)
    for slot in range(N_SLOTS):
        nudged = model.copy()
        nudged.means = model.means.copy()
        scale = float(np.sqrt(max(float(model.vars[slot, 0]), VAR_FLOOR)))
        nudged.means[slot, 0] = model.means[slot, 0] + 0.1 * scale
        mean_drop[slot] = base - total_loglik(sequences, nudged)
    return {
        "base_loglik": np.array(base),
        "branch_drop": np.asarray(branch_drop, dtype=np.float64),
        "mean_drop": mean_drop,
    }


def sample_sequence(model: SoftHMM, t_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw one sequence and its state path. Used by tests."""
    tables = model.tables()
    path = np.empty(t_len, dtype=np.int32)
    x = np.empty((t_len, model.means.shape[1]), dtype=np.float64)
    theme = int(rng.choice(N_THEMES, p=np.asarray(model.start_prob, dtype=np.float64)))
    state = int(START_STATES[theme])
    for t in range(t_len):
        path[t] = state
        slot = state // N_DUR
        x[t] = rng.normal(model.means[slot], np.sqrt(model.vars[slot]))
        n_k = int(tables.n_out[state])
        probs = np.exp(tables.logp[state, :n_k])
        probs = probs / probs.sum()
        choice = int(rng.choice(n_k, p=probs))
        state = int(tables.to[state, choice])
    return x, path
