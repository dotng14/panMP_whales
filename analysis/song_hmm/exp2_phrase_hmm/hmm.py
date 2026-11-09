"""Emissions, inference and Baum-Welch for the theme-level graph.

The log-domain sparse-graph kernels are shared with Experiment 1
(``analysis.song_hmm.exp1_slot_hmm.hmm``); only the parameters and the M-step differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.song_hmm.exp1_slot_hmm.hmm import _backward, _forward, _viterbi

from .topology import (
    BG,
    BG_ENTER,
    DWELL_KINDS,
    ENTRY,
    N_GROUPS,
    N_STATES,
    N_THEMES,
    PAUSE_KINDS,
    STATE_GROUP,
    Graph,
    Params,
    build_graph,
)

VAR_FLOOR = 1e-2
PROB_FLOOR = 1e-6
END_PROB = 1e-4  # fixed: P(session ends) per step from a theme's dwell state
BG_MEAN_STEPS = 1800.0  # fixed: background lasts 1 h on average (2 s steps)


def log_emit(x: np.ndarray, p: Params) -> np.ndarray:
    var = np.maximum(p.vars, VAR_FLOOR)
    prec = 1.0 / var
    quad = (x * x) @ prec.T - 2.0 * x @ (p.means * prec).T + np.sum(p.means**2 * prec, axis=1)
    return (-0.5 * (quad + np.sum(np.log(2.0 * np.pi * var), axis=1)))[:, STATE_GROUP]


def _logs(x, p, g):
    with np.errstate(divide="ignore"):
        return log_emit(x, p), np.log(g.start), np.log(g.prob)


def _lse(v):
    m = v.max()
    return float(m + np.log(np.exp(v - m).sum()))


@dataclass
class Posterior:
    loglik: float
    gamma: np.ndarray
    edge_counts: np.ndarray
    graph: Graph


def loglik(x: np.ndarray, p: Params) -> float:
    g = build_graph(p)
    logb, log_start, logp = _logs(x, p, g)
    return _lse(_forward(logb, log_start, g.src, g.dst, logp)[-1])


def posterior(x: np.ndarray, p: Params) -> Posterior:
    g = build_graph(p)
    logb, log_start, logp = _logs(x, p, g)
    la = _forward(logb, log_start, g.src, g.dst, logp)
    ll = _lse(la[-1])
    gamma, counts = _backward(logb, la, ll, g.src, g.dst, logp)
    return Posterior(ll, gamma, counts, g)


def viterbi(x: np.ndarray, p: Params) -> tuple[np.ndarray, float]:
    g = build_graph(p)
    logb, log_start, logp = _logs(x, p, g)
    return _viterbi(logb, log_start, g.src, g.dst, logp)


def _normalize(c: np.ndarray) -> np.ndarray:
    q = c / c.sum() if c.sum() > 0 else np.full(c.shape, 1.0 / c.size)
    q = np.maximum(q, PROB_FLOOR)
    return q / q.sum()


def m_step(p: Params, xs: list[np.ndarray], posts: list[Posterior]) -> Params:
    new = p.copy()
    member = np.zeros((N_STATES, N_GROUPS))
    member[np.arange(N_STATES), STATE_GROUP] = 1.0
    d = xs[0].shape[1]
    w, s1, s2 = np.zeros(N_GROUPS), np.zeros((N_GROUPS, d)), np.zeros((N_GROUPS, d))
    edge = np.zeros_like(posts[0].edge_counts)
    start = np.zeros(N_STATES)
    for x, post in zip(xs, posts, strict=True):
        gg = post.gamma @ member
        w += gg.sum(0)
        s1 += gg.T @ x
        s2 += gg.T @ (x * x)
        edge += post.edge_counts
        start += post.gamma[0]
    ok = w > 1e-3
    new.means[ok] = s1[ok] / w[ok, None]
    new.vars[ok] = np.maximum(s2[ok] / w[ok, None] - new.means[ok] ** 2, VAR_FLOOR)

    g = posts[0].graph
    for th in range(N_THEMES):
        mine = g.theme == th
        song = np.array([edge[mine & (g.kind == k)].sum() for k in DWELL_KINDS[:3]])
        new.dwell[th, :3] = _normalize(song) * (1.0 - p.dwell[th, 3])  # END stays fixed
        new.pause[th] = _normalize(np.array([edge[mine & (g.kind == k)].sum() for k in PAUSE_KINDS]))
    enter = np.array([edge[(g.kind == BG_ENTER) & (g.theme == th)].sum() for th in range(N_THEMES)])
    new.bg[1:] = _normalize(enter) * (1.0 - p.bg[0])  # BG stay stays fixed
    new.start = _normalize(np.concatenate([[start[BG]], start[ENTRY]]) + 1e-3)
    return new


def baum_welch(xs: list[np.ndarray], p: Params, *, max_iter: int = 50, tol: float = 1e-5, verbose: bool = True):
    history: list[float] = []
    for it in range(max_iter):
        posts = [posterior(x, p) for x in xs]
        ll = sum(q.loglik for q in posts)
        history.append(ll)
        if verbose:
            print(f"  iter {it:2d}  loglik {ll:,.1f}  ({ll / sum(len(x) for x in xs):.4f} / step)", flush=True)
        if it > 0 and abs(history[-1] - history[-2]) < tol * abs(history[-2]):
            break
        p = m_step(p, xs, posts)
    else:
        history.append(sum(loglik(x, p) for x in xs))
    return p, history


def default_transitions() -> dict[str, np.ndarray]:
    dwell = np.tile([0.0, 0.05, 0.02, END_PROB], (N_THEMES, 1))
    dwell[:, 0] = 1.0 - dwell[:, 1:].sum(axis=1)
    return {
        "dwell": dwell,
        "pause": np.tile([0.8, 0.15, 0.05], (N_THEMES, 1)),
        "bg": np.concatenate([[1.0 - 1.0 / BG_MEAN_STEPS], np.full(N_THEMES, 1.0 / BG_MEAN_STEPS / N_THEMES)]),
        "start": np.full(1 + N_THEMES, 1.0 / (1 + N_THEMES)),
    }


def params_from_assignment(x: np.ndarray, theme: np.ndarray, bg_mask: np.ndarray) -> Params:
    """Seed Gaussians from a per-window theme assignment (0..5) and a background mask."""
    d = x.shape[1]
    means, var = np.zeros((N_GROUPS, d)), np.ones((N_GROUPS, d))
    for k in range(N_THEMES):
        rows = x[(theme == k) & ~bg_mask]
        if len(rows) >= 2:
            means[k], var[k] = rows.mean(0), np.maximum(rows.var(0), VAR_FLOOR)
    means[N_THEMES], var[N_THEMES] = x[bg_mask].mean(0), np.maximum(x[bg_mask].var(0), VAR_FLOOR)
    return Params(means, var, **default_transitions())


def sample(p: Params, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    g = build_graph(p)
    order = np.argsort(g.src, kind="stable")
    bounds = np.searchsorted(g.src[order], np.arange(N_STATES + 1))
    path = np.empty(n, dtype=np.int32)
    path[0] = rng.choice(N_STATES, p=g.start / g.start.sum())
    for t in range(1, n):
        es = order[bounds[path[t - 1]] : bounds[path[t - 1] + 1]]
        q = g.prob[es]
        path[t] = g.dst[es[rng.choice(len(es), p=q / q.sum())]]
    grp = STATE_GROUP[path]
    x = p.means[grp] + rng.standard_normal((n, p.means.shape[1])) * np.sqrt(np.maximum(p.vars[grp], VAR_FLOOR))
    return x, path
