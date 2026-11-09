"""A small HMM engine shared by M1, M3 and M4.

Graph: sparse edges src -> dst. Each edge probability is a product of one or two
tied parameters theta[family][outcome] (e.g. "leave this state" x "go to the next
theme"). Each edge also lists which (family, outcome) pairs it uses, so the
M-step is exact: the expected count of an edge is added to every factor it uses,
and each family is re-normalised. Families marked fixed keep their values.

Emissions: one model per emission group. DiagGMM (Gaussian mixture, diagonal)
for vector inputs, Categorical for token inputs.

Inference uses the log-domain numba kernels of Experiment 1 (analysis/song_hmm/hmm.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from analysis.song_hmm.hmm import _backward, _forward, _viterbi

PROB_FLOOR = 1e-6
VAR_FLOOR = 1e-2


# ================================================================ transitions


@dataclass
class Family:
    name: str
    theta: np.ndarray  # (n_outcomes,)
    fixed: bool = False
    outcomes: tuple[str, ...] = ()


@dataclass
class Graph:
    n_states: int
    group: np.ndarray  # (S,) emission group per state
    families: dict[str, Family]
    edges: list[tuple[int, int, tuple[tuple[str, int], ...]]] = field(default_factory=list)
    start: list[tuple[int, tuple[tuple[str, int], ...]]] = field(default_factory=list)

    def add(self, src: int, dst: int, *factors: tuple[str, int]) -> None:
        self.edges.append((src, dst, tuple(factors)))

    def add_start(self, state: int, *factors: tuple[str, int]) -> None:
        self.start.append((state, tuple(factors)))

    def arrays(self):
        src = np.array([e[0] for e in self.edges], np.int32)
        dst = np.array([e[1] for e in self.edges], np.int32)
        with np.errstate(divide="ignore"):
            logp = np.array([sum(np.log(self.families[f].theta[o]) for f, o in e[2]) for e in self.edges])
            log_start = np.full(self.n_states, -np.inf)
            for s, fac in self.start:
                log_start[s] = np.logaddexp(log_start[s], sum(np.log(self.families[f].theta[o]) for f, o in fac))
        return src, dst, logp, log_start

    def check(self) -> np.ndarray:
        src, dst, logp, log_start = self.arrays()
        out = np.zeros(self.n_states)
        np.add.at(out, src, np.exp(logp))
        return out


def update_families(g: Graph, edge_counts: np.ndarray, start_post: np.ndarray) -> None:
    acc = {k: np.zeros_like(f.theta) for k, f in g.families.items()}
    for (_, _, fac), c in zip(g.edges, edge_counts, strict=True):
        for f, o in fac:
            acc[f][o] += c
    for s, fac in g.start:
        for f, o in fac:
            acc[f][o] += start_post[s]
    for k, f in g.families.items():
        if f.fixed or acc[k].sum() <= 1e-9:
            continue
        q = np.maximum(acc[k] / acc[k].sum(), PROB_FLOOR)
        f.theta = q / q.sum()


# ================================================================ emissions


@dataclass
class DiagGMM:
    w: np.ndarray  # (G, K)
    mu: np.ndarray  # (G, K, D)
    var: np.ndarray  # (G, K, D)

    def comp_logpdf(self, x: np.ndarray) -> np.ndarray:  # (T, G, K)
        var = np.maximum(self.var, VAR_FLOOR)
        prec = 1.0 / var
        quad = (x * x) @ prec.reshape(-1, x.shape[1]).T - 2 * x @ (self.mu * prec).reshape(-1, x.shape[1]).T
        quad += np.sum(self.mu**2 * prec, axis=2).reshape(-1)
        lp = -0.5 * (quad + np.sum(np.log(2 * np.pi * var), axis=2).reshape(-1))
        with np.errstate(divide="ignore"):
            return lp.reshape(len(x), *self.w.shape) + np.log(self.w)[None]

    def logpdf(self, x: np.ndarray) -> np.ndarray:  # (T, G)
        return logsumexp(self.comp_logpdf(x), axis=2)

    def m_step(self, xs: list[np.ndarray], gammas_g: list[np.ndarray]) -> None:
        G, K, D = self.mu.shape
        n = np.zeros((G, K))
        s1 = np.zeros((G, K, D))
        s2 = np.zeros((G, K, D))
        for x, gg in zip(xs, gammas_g, strict=True):
            lc = self.comp_logpdf(x)
            r = np.exp(lc - logsumexp(lc, axis=2, keepdims=True)) * gg[:, :, None]  # (T, G, K)
            n += r.sum(0)
            s1 += np.einsum("tgk,td->gkd", r, x)
            s2 += np.einsum("tgk,td->gkd", r, x * x)
        ok = n > 1e-3
        self.mu[ok] = s1[ok] / n[ok][:, None]
        self.var[ok] = np.maximum(s2[ok] / n[ok][:, None] - self.mu[ok] ** 2, VAR_FLOOR)
        tot = n.sum(1, keepdims=True)
        self.w = np.where(tot > 1e-3, np.maximum(n / np.maximum(tot, 1e-9), 1e-3), self.w)
        self.w /= self.w.sum(1, keepdims=True)

    @classmethod
    def from_assignment(cls, x: np.ndarray, group: np.ndarray, n_groups: int, k: int = 2, seed: int = 0) -> "DiagGMM":
        """Seed each group from its assigned rows; components split by a 2-means on that group."""
        rng = np.random.default_rng(seed)
        D = x.shape[1]
        w = np.full((n_groups, k), 1.0 / k)
        mu = np.zeros((n_groups, k, D))
        var = np.ones((n_groups, k, D))
        gm, gv = x.mean(0), x.var(0)
        for g in range(n_groups):
            rows = x[group == g]
            if len(rows) < 2 * k:
                mu[g] = gm + rng.normal(0, 0.1, (k, D))
                var[g] = gv
                continue
            c = rows[rng.choice(len(rows), k, replace=False)]
            for _ in range(10):
                lab = np.argmin(((rows[:, None, :] - c[None]) ** 2).sum(-1), 1)
                c = np.array([rows[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            for j in range(k):
                rj = rows[lab == j] if np.sum(lab == j) > 1 else rows
                mu[g, j], var[g, j] = rj.mean(0), np.maximum(rj.var(0), VAR_FLOOR)
                w[g, j] = max(np.mean(lab == j), 0.05)
            w[g] /= w[g].sum()
        return cls(w, mu, var)


@dataclass
class Categorical:
    logp: np.ndarray  # (G, V)

    def logpdf(self, tokens: np.ndarray) -> np.ndarray:  # (T, G)
        return self.logp[:, tokens].T

    def m_step(self, xs: list[np.ndarray], gammas_g: list[np.ndarray], pseudo: float = 0.1) -> None:
        G, V = self.logp.shape
        c = np.full((G, V), pseudo)
        for tok, gg in zip(xs, gammas_g, strict=True):
            for g in range(G):
                np.add.at(c[g], tok, gg[:, g])
        self.logp = np.log(c / c.sum(1, keepdims=True))

    @classmethod
    def from_assignment(cls, tokens: np.ndarray, group: np.ndarray, n_groups: int, vocab: int, pseudo: float = 1.0) -> "Categorical":
        c = np.full((n_groups, vocab), pseudo)
        ok = group >= 0
        np.add.at(c, (group[ok], tokens[ok]), 1.0)
        return cls(np.log(c / c.sum(1, keepdims=True)))


# ================================================================ inference and learning


@dataclass
class HMM:
    graph: Graph
    emis: object  # DiagGMM or Categorical

    def _logs(self, x):
        src, dst, logp, log_start = self.graph.arrays()
        logb = np.ascontiguousarray(self.emis.logpdf(x)[:, self.graph.group])
        return logb, src, dst, logp, log_start

    def posterior(self, x):
        logb, src, dst, logp, log_start = self._logs(x)
        la = _forward(logb, log_start, src, dst, logp)
        m = la[-1].max()
        ll = float(m + np.log(np.exp(la[-1] - m).sum()))
        gamma, counts = _backward(logb, la, ll, src, dst, logp)
        return ll, gamma, counts

    def loglik(self, x) -> float:
        logb, src, dst, logp, log_start = self._logs(x)
        la = _forward(logb, log_start, src, dst, logp)
        m = la[-1].max()
        return float(m + np.log(np.exp(la[-1] - m).sum()))

    def viterbi(self, x) -> np.ndarray:
        logb, src, dst, logp, log_start = self._logs(x)
        path, _ = _viterbi(logb, log_start, src, dst, logp)
        return path

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw (observations, state path) of length n."""
        src, dst, logp, log_start = self.graph.arrays()
        p = np.exp(logp)
        order = np.argsort(src, kind="stable")
        bounds = np.searchsorted(src[order], np.arange(self.graph.n_states + 1))
        ps = np.exp(log_start)
        path = np.empty(n, np.int64)
        path[0] = rng.choice(self.graph.n_states, p=ps / ps.sum())
        for t in range(1, n):
            es = order[bounds[path[t - 1]] : bounds[path[t - 1] + 1]]
            q = p[es]
            path[t] = dst[es[rng.choice(len(es), p=q / q.sum())]]
        grp = self.graph.group[path]
        if isinstance(self.emis, Categorical):
            pr = np.exp(self.emis.logp)
            x = np.array([rng.choice(pr.shape[1], p=pr[g] / pr[g].sum()) for g in grp])
        else:
            e = self.emis
            comp = np.array([rng.choice(e.w.shape[1], p=e.w[g]) for g in grp])
            x = e.mu[grp, comp] + rng.standard_normal((n, e.mu.shape[2])) * np.sqrt(np.maximum(e.var[grp, comp], VAR_FLOOR))
        return x, path

    def fit(self, xs: list, max_iter: int = 30, tol: float = 1e-5, update_emissions: bool = True, verbose: bool = False) -> list[float]:
        n_groups = self.emis.w.shape[0] if isinstance(self.emis, DiagGMM) else self.emis.logp.shape[0]
        member = np.zeros((self.graph.n_states, n_groups))
        member[np.arange(self.graph.n_states), self.graph.group] = 1.0
        hist: list[float] = []
        for it in range(max_iter):
            tot, gg_all, ec, sp = 0.0, [], 0.0, np.zeros(self.graph.n_states)
            for x in xs:
                ll, gamma, counts = self.posterior(x)
                tot += ll
                gg_all.append(gamma @ member)
                ec = ec + counts
                sp += gamma[0]
            hist.append(tot)
            if verbose:
                print(f"    iter {it:2d} loglik {tot:,.1f}", flush=True)
            if it > 0 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-2]):
                break
            update_families(self.graph, ec, sp)
            if update_emissions:
                self.emis.m_step(xs, gg_all)
        return hist
