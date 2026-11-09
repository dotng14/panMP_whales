"""Initial parameters for Experiment 2.

Audio seeding (main run):
  1. background = the 10% of train windows with the lowest mean level
  2. k-means (k = 6) on the other train windows
  3. majority-smooth the cluster sequence over 10 steps (20 s)
  4. count transitions between successive cluster runs; pick the cyclic order
     of the 6 clusters that the transitions follow best (all 120 orders)
  5. which cluster is Theme 1 is the one free choice left: take the rotation
     that agrees best with the CSV on the train half (the only CSV use here)

CSV seeding (ablation 2b): theme Gaussians from the CSV theme at each window centre.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from .hmm import params_from_assignment
from .topology import N_THEMES, Params

BG_QUANTILE = 0.10
SMOOTH_STEPS = 10


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 100, n_init: int = 10) -> np.ndarray:
    best, best_cost = None, np.inf
    for _ in range(n_init):
        centers = [x[rng.integers(len(x))]]
        for _ in range(k - 1):  # k-means++
            d2 = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            centers.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        c = np.array(centers)
        for _ in range(n_iter):
            lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
            new = np.array([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        cost = ((x - c[lab]) ** 2).sum()
        if cost < best_cost:
            best, best_cost = lab, cost
    return best


def majority_smooth(lab: np.ndarray, width: int, k: int) -> np.ndarray:
    out = lab.copy()
    half = width // 2
    for i in range(len(lab)):
        seg = lab[max(0, i - half) : i + half + 1]
        seg = seg[seg >= 0]
        if seg.size:
            out[i] = np.bincount(seg, minlength=k).argmax()
    return out


def run_transitions(lab: np.ndarray, k: int) -> np.ndarray:
    runs = lab[np.r_[True, lab[1:] != lab[:-1]]]
    runs = runs[runs >= 0]
    runs = runs[np.r_[True, runs[1:] != runs[:-1]]]
    t = np.zeros((k, k))
    np.add.at(t, (runs[:-1], runs[1:]), 1)
    return t


def best_cycle(t: np.ndarray) -> tuple[tuple[int, ...], float]:
    k = len(t)
    best, score = None, -1.0
    for rest in permutations(range(1, k)):
        order = (0, *rest)
        s = sum(t[order[i], order[(i + 1) % k]] for i in range(k))
        if s > score:
            best, score = order, s
    return best, score / max(t.sum(), 1)


def audio_seed(x: np.ndarray, level: np.ndarray, csv_theme: np.ndarray, seed: int = 0) -> tuple[Params, dict]:
    rng = np.random.default_rng(seed)
    bg = level <= np.quantile(level, BG_QUANTILE)
    lab = np.full(len(x), -1)
    lab[~bg] = kmeans(x[~bg], N_THEMES, rng)
    smooth = majority_smooth(lab, SMOOTH_STEPS, N_THEMES)
    trans = run_transitions(smooth, N_THEMES)
    order, cycle_share = best_cycle(trans)
    position = np.empty(N_THEMES, dtype=int)
    position[list(order)] = np.arange(N_THEMES)  # cluster -> place in the cycle
    agreement = []
    for r in range(N_THEMES):
        theme = (position[smooth] + r) % N_THEMES
        agreement.append(float(np.mean(theme[csv_theme >= 0] == csv_theme[csv_theme >= 0])))
    rot = int(np.argmax(agreement))
    theme = (position[smooth] + rot) % N_THEMES
    info = {
        "cluster_order": list(order),
        "cycle_share": cycle_share,
        "transitions": trans.tolist(),
        "rotation_agreement": agreement,
        "rotation": rot,
        "cluster_sizes": np.bincount(lab[lab >= 0], minlength=N_THEMES).tolist(),
    }
    return params_from_assignment(x, theme, bg), info


def csv_seed(x: np.ndarray, level: np.ndarray, csv_theme: np.ndarray) -> tuple[Params, dict]:
    bg = level <= np.quantile(level, BG_QUANTILE)
    return params_from_assignment(x, np.where(csv_theme >= 0, csv_theme, 0), bg), {}
