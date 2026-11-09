"""Forward-backward, Viterbi and Baum-Welch on the sparse song-session graph.

Everything runs in the log domain: the posterior can sit on states whose
forward probability is far below the range of a float64 (for example inside
the background chain), which a scaled probability-domain pass cannot hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .topology import (
    BG_ENTER,
    BG_GROUP,
    BG_STAY,
    BRANCH_KINDS,
    CONT,
    ENTRY_STATES,
    N_DUR,
    N_GROUPS,
    N_SLOTS,
    N_STATES,
    N_THEMES,
    PAUSE_KINDS,
    STATE_GROUP,
    STATE_SLOT,
    BG,
    Graph,
    Params,
    build_graph,
    exit_prob_from_lengths,
)

VAR_FLOOR = 1e-2
PROB_FLOOR = 1e-6


# ---------------------------------------------------------------- emissions


def log_emit_groups(x: np.ndarray, p: Params) -> np.ndarray:
    """Diagonal-Gaussian log density per emission group. (T, N_GROUPS)."""
    var = np.maximum(p.vars, VAR_FLOOR)
    prec = 1.0 / var
    quad = (x * x) @ prec.T - 2.0 * x @ (p.means * prec).T + np.sum(p.means**2 * prec, axis=1)
    return -0.5 * (quad + np.sum(np.log(2.0 * np.pi * var), axis=1))


def log_emit(x: np.ndarray, p: Params) -> np.ndarray:
    return log_emit_groups(x, p)[:, STATE_GROUP]


# ---------------------------------------------------------------- kernels


@njit(cache=True)
def _forward(logb, log_start, src, dst, logp):
    t_len, n = logb.shape
    la = np.empty((t_len, n))
    la[0] = log_start + logb[0]
    for t in range(1, t_len):
        new = np.full(n, -np.inf)
        prev = la[t - 1]
        for e in range(src.shape[0]):
            new[dst[e]] = np.logaddexp(new[dst[e]], prev[src[e]] + logp[e])
        la[t] = new + logb[t]
    return la


@njit(cache=True)
def _backward(logb, la, loglik, src, dst, logp):
    """Posterior state probabilities and expected edge counts."""
    t_len, n = logb.shape
    gamma = np.empty((t_len, n))
    counts = np.zeros(src.shape[0])
    lb = np.zeros(n)
    gamma[t_len - 1] = np.exp(la[t_len - 1] - loglik)
    for t in range(t_len - 2, -1, -1):
        w = logb[t + 1] + lb
        new = np.full(n, -np.inf)
        for e in range(src.shape[0]):
            v = logp[e] + w[dst[e]]
            new[src[e]] = np.logaddexp(new[src[e]], v)
            counts[e] += np.exp(la[t, src[e]] + v - loglik)
        lb = new
        gamma[t] = np.exp(la[t] + lb - loglik)
    return gamma, counts


@njit(cache=True)
def _viterbi(logb, log_start, src, dst, logp):
    t_len, n = logb.shape
    back = np.empty((t_len, n), dtype=np.int32)
    delta = log_start + logb[0]
    back[0] = -1
    for t in range(1, t_len):
        new = np.full(n, -np.inf)
        ptr = np.full(n, -1, dtype=np.int32)
        for e in range(src.shape[0]):
            v = delta[src[e]] + logp[e]
            if v > new[dst[e]]:
                new[dst[e]] = v
                ptr[dst[e]] = src[e]
        delta = new + logb[t]
        back[t] = ptr
    path = np.empty(t_len, dtype=np.int32)
    path[-1] = np.argmax(delta)
    best = delta[path[-1]]
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, best


# ---------------------------------------------------------------- inference


@dataclass
class Posterior:
    loglik: float
    gamma: np.ndarray  # (T, N_STATES)
    edge_counts: np.ndarray  # (E,)
    graph: Graph


def _logs(x: np.ndarray, p: Params, g: Graph):
    with np.errstate(divide="ignore"):
        return log_emit(x, p), np.log(g.start), np.log(g.prob)


def _logsumexp(v: np.ndarray) -> float:
    m = v.max()
    return float(m + np.log(np.exp(v - m).sum()))


def loglik(x: np.ndarray, p: Params, graph: Graph | None = None) -> float:
    g = build_graph(p) if graph is None else graph
    logb, log_start, logp = _logs(x, p, g)
    return _logsumexp(_forward(logb, log_start, g.src, g.dst, logp)[-1])


def posterior(x: np.ndarray, p: Params, graph: Graph | None = None) -> Posterior:
    g = build_graph(p) if graph is None else graph
    logb, log_start, logp = _logs(x, p, g)
    la = _forward(logb, log_start, g.src, g.dst, logp)
    ll = _logsumexp(la[-1])
    gamma, counts = _backward(logb, la, ll, g.src, g.dst, logp)
    return Posterior(ll, gamma, counts, g)


def viterbi(x: np.ndarray, p: Params) -> tuple[np.ndarray, float]:
    g = build_graph(p)
    logb, log_start, logp = _logs(x, p, g)
    return _viterbi(logb, log_start, g.src, g.dst, logp)


# ---------------------------------------------------------------- learning


def _normalize(counts: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    q = counts / counts.sum() if counts.sum() > 0 else np.full(counts.shape, 1.0 / counts.size)
    q = np.maximum(q, floor)
    return q / q.sum()


def m_step(p: Params, x_list: list[np.ndarray], posts: list[Posterior]) -> Params:
    new = p.copy()
    k = x_list[0].shape[1]
    w = np.zeros(N_GROUPS)
    s1 = np.zeros((N_GROUPS, k))
    s2 = np.zeros((N_GROUPS, k))
    member = np.zeros((N_STATES, N_GROUPS))
    member[np.arange(N_STATES), STATE_GROUP] = 1.0
    edge = np.zeros_like(posts[0].edge_counts)
    start = np.zeros(N_STATES)
    for x, post in zip(x_list, posts, strict=True):
        gg = post.gamma @ member
        w += gg.sum(axis=0)
        s1 += gg.T @ x
        s2 += gg.T @ (x * x)
        edge += post.edge_counts
        start += post.gamma[0]
    ok = w > 1e-3
    new.means[ok] = s1[ok] / w[ok, None]
    new.vars[ok] = np.maximum(s2[ok] / w[ok, None] - new.means[ok] ** 2, VAR_FLOOR)

    g = posts[0].graph
    slot = STATE_SLOT[g.src]
    dur = np.where(slot >= 0, g.src % N_DUR, -1)
    stay = np.zeros((N_SLOTS, N_DUR - 1))
    leave = np.zeros((N_SLOTS, N_DUR - 1))
    inner = (slot >= 0) & (dur < N_DUR - 1)
    is_cont = g.kind == CONT
    np.add.at(stay, (slot[inner & is_cont], dur[inner & is_cont]), edge[inner & is_cont])
    np.add.at(leave, (slot[inner & ~is_cont], dur[inner & ~is_cont]), edge[inner & ~is_cont])
    seen = stay + leave > 1e-8
    new.exit_prob[seen] = np.clip(leave[seen] / (stay + leave)[seen], 1e-3, 1 - 1e-3)

    # Session on/off rates are a fixed prior (there is no labeled non-song to learn them
    # from): the END branch and the BG self-loop keep their values. Only how the rest
    # of the mass is split is learned.
    for th in range(N_THEMES):
        mine = g.theme == th
        song = np.array([edge[mine & (g.kind == kd)].sum() for kd in BRANCH_KINDS[:3]])
        new.branch[th, :3] = _normalize(song) * (1.0 - p.branch[th, 3])
        new.pause[th] = _normalize(np.array([edge[mine & (g.kind == kd)].sum() for kd in PAUSE_KINDS]))
    enter = np.array([edge[(g.kind == BG_ENTER) & (g.theme == th)].sum() for th in range(N_THEMES)])
    new.bg[1:] = _normalize(enter) * (1.0 - p.bg[0])
    new.start = _normalize(np.concatenate([[start[BG]], start[ENTRY_STATES]]) + 1e-3)
    return new


def baum_welch(
    x_list: list[np.ndarray],
    p: Params,
    *,
    max_iter: int = 30,
    tol: float = 1e-5,
    verbose: bool = True,
) -> tuple[Params, list[float]]:
    """Fit on one or more sequences. Returns the fitted parameters and the
    log-likelihood of each iteration's starting parameters, plus the final one."""
    history: list[float] = []
    for it in range(max_iter):
        posts = [posterior(x, p) for x in x_list]
        ll = sum(post.loglik for post in posts)
        history.append(ll)
        if verbose:
            n = sum(len(x) for x in x_list)
            print(f"  iter {it:2d}  loglik {ll:,.1f}  ({ll / n:.4f} / s)", flush=True)
        if it > 0 and abs(history[-1] - history[-2]) < tol * abs(history[-2]):
            break
        p = m_step(p, x_list, posts)
    else:
        history.append(sum(loglik(x, p) for x in x_list))
    return p, history


# ---------------------------------------------------------------- initialization


REPEAT_INIT = np.array([0.008, 0.984, 0.943, 0.585, 0.922, 0.95])
BREATH_INIT = 0.02
END_PROB = 1e-3  # fixed: P(session ends) at each phrase end
BG_MEAN_S = 3600.0  # fixed: expected length of a background stretch


def initial_params(x: np.ndarray, slot_per_second: np.ndarray, length_hist: np.ndarray, level_col: int) -> Params:
    """Seed from per-second slot labels (-1 = unlabeled) and the drawing's repeat weights."""
    k = x.shape[1]
    means = np.zeros((N_GROUPS, k))
    var = np.ones((N_GROUPS, k))
    for s in range(N_SLOTS):
        rows = x[slot_per_second == s]
        if len(rows) >= 2:
            means[s], var[s] = rows.mean(axis=0), np.maximum(rows.var(axis=0), VAR_FLOOR)
    quiet = x[x[:, level_col] <= np.quantile(x[:, level_col], 0.10)]
    means[BG_GROUP], var[BG_GROUP] = quiet.mean(axis=0), np.maximum(quiet.var(axis=0), VAR_FLOOR)

    exit_prob = np.array([exit_prob_from_lengths(length_hist[s]) for s in range(N_SLOTS)])
    branch = np.column_stack(
        [REPEAT_INIT, 1.0 - REPEAT_INIT - BREATH_INIT - END_PROB, np.full(N_THEMES, BREATH_INIT), np.full(N_THEMES, END_PROB)]
    )
    branch = np.maximum(branch, PROB_FLOOR)
    branch /= branch.sum(axis=1, keepdims=True)
    pause = np.tile([0.9, 0.05, 0.05], (N_THEMES, 1))
    bg = np.concatenate([[1.0 - 1.0 / BG_MEAN_S], np.full(N_THEMES, 1.0 / BG_MEAN_S / N_THEMES)])
    start = np.full(1 + N_THEMES, 1.0 / (1 + N_THEMES))
    return Params(means, var, exit_prob, branch, pause, bg, start)


# ---------------------------------------------------------------- sampling


def sample(p: Params, t_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw (observations, state path)."""
    g = build_graph(p)
    order = np.argsort(g.src, kind="stable")
    bounds = np.searchsorted(g.src[order], np.arange(N_STATES + 1))
    path = np.empty(t_len, dtype=np.int32)
    path[0] = rng.choice(N_STATES, p=g.start / g.start.sum())
    for t in range(1, t_len):
        es = order[bounds[path[t - 1]] : bounds[path[t - 1] + 1]]
        q = g.prob[es]
        path[t] = g.dst[es[rng.choice(len(es), p=q / q.sum())]]
    grp = STATE_GROUP[path]
    x = p.means[grp] + rng.standard_normal((t_len, p.means.shape[1])) * np.sqrt(np.maximum(p.vars[grp], VAR_FLOOR))
    return x, path
