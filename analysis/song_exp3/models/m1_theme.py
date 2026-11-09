"""M1: one theme-level HMM on window inputs (Experiment 2, updated).

States, one step = one window (2 s):
  T1..T6        one state per theme; a theme can be a single window (single-phrase themes)
  P1..P6        pause inside theme k (shares the background emission)
  BG chain      30 states = 60 s minimum background, then a looping BG state
From T_k: stay / next theme (T6 -> T1 is the song boundary) / pause / end session.
Emissions: 2-component diagonal Gaussian mixture per theme and for background.
Seeded from the CSV themes at each window centre (train half); background from the
quietest 10% of windows.
"""

from __future__ import annotations

import pickle

import numpy as np

from analysis.song_hmm_phrase.decode import Decoded

from .common import HALVES, N_THEMES, WORK, at_times, boundaries_from_theme_steps, enforce_session_rule, labels, level_1s, save, steps_to_seconds
from .graph import HMM, DiagGMM, Family, Graph

BG_MIN_STEPS = 30
END_PROB = 1e-4
BG_STAY = 1 - 1 / 1800


def build_graph() -> tuple[Graph, np.ndarray]:
    T = np.arange(N_THEMES)
    P = N_THEMES + T
    bg0 = 2 * N_THEMES
    bg = bg0 + BG_MIN_STEPS - 1
    n = bg + 1
    group = np.r_[T, np.full(N_THEMES, N_THEMES), np.full(BG_MIN_STEPS, N_THEMES)]
    theme_of = np.r_[T, T, np.full(BG_MIN_STEPS, -1)]
    fam = {"one": Family("one", np.array([1.0]), fixed=True), "bg": Family("bg", np.array([BG_STAY, 1 - BG_STAY]), fixed=True),
           "bg_enter": Family("bg_enter", np.full(N_THEMES, 1 / N_THEMES)), "start": Family("start", np.full(N_THEMES + 1, 1 / (N_THEMES + 1)))}
    g = Graph(n, group.astype(np.int32), fam)
    for k in range(N_THEMES):
        nxt = (k + 1) % N_THEMES
        fam[f"end{k}"] = Family(f"end{k}", np.array([1 - END_PROB, END_PROB]), fixed=True)
        fam[f"dwell{k}"] = Family(f"dwell{k}", np.array([0.95, 0.04, 0.01]), outcomes=("stay", "next", "pause"))
        fam[f"pause{k}"] = Family(f"pause{k}", np.array([0.8, 0.15, 0.05]), outcomes=("stay", "resume", "next"))
        g.add(T[k], T[k], (f"end{k}", 0), (f"dwell{k}", 0))
        g.add(T[k], T[nxt], (f"end{k}", 0), (f"dwell{k}", 1))
        g.add(T[k], P[k], (f"end{k}", 0), (f"dwell{k}", 2))
        g.add(T[k], bg0, (f"end{k}", 1))
        g.add(P[k], P[k], (f"pause{k}", 0))
        g.add(P[k], T[k], (f"pause{k}", 1))
        g.add(P[k], T[nxt], (f"pause{k}", 2))
        g.add(bg, T[k], ("bg", 1), ("bg_enter", k))
        g.add_start(T[k], ("start", 1 + k))
    for s in range(bg0, bg):
        g.add(s, s + 1, ("one", 0))
    g.add(bg, bg, ("bg", 0))
    g.add_start(bg, ("start", 0))
    return g, theme_of


def run(name: str, inp: dict, n_pc: int = 40, verbose: bool = True) -> dict:
    """``inp``: npz-like with train/test PCA arrays and window-centre times t_train/t_test (recording s)."""
    lab = labels()
    lev = level_1s()
    xs = {"train": inp["train"][:, :n_pc].astype(np.float64), "test": inp["test"][:, :n_pc].astype(np.float64)}
    ts = {"train": inp["t_train"], "test": inp["t_test"]}
    g, theme_of = build_graph()
    csv_th = at_times(lab.theme, ts["train"])
    quiet = at_times(lev, ts["train"]) <= np.quantile(at_times(lev, ts["train"]), 0.10)
    grp = np.where(quiet, N_THEMES, csv_th)
    hmm = HMM(g, DiagGMM.from_assignment(xs["train"], grp, N_THEMES + 1))
    hist = hmm.fit([xs["train"]], max_iter=40, verbose=verbose)
    decs, lls = decode_halves(name, hmm, xs, ts)
    save(name, decs)
    model_dir = WORK / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "model.pkl", "wb") as handle:  # the fitted HMM, for detection without retraining
        pickle.dump({"hmm": hmm, "theme_of": theme_of, "n_pc": n_pc}, handle)
    return {"decoded": decs, "history": hist, "loglik_per_step": lls, "hmm": hmm}


def decode_halves(name: str, hmm: HMM, xs: dict, ts: dict) -> tuple[dict, dict]:
    """Viterbi on each half (train, test) -> common decoded form, plus log-likelihood per step."""
    _, theme_of = build_graph()
    decs, lls = {}, {}
    for h, off, end in HALVES:
        path = hmm.viterbi(xs[h])
        lls[h] = hmm.loglik(xs[h]) / len(xs[h])
        th_steps = theme_of[path]
        theme = steps_to_seconds(th_steps, ts[h], off, end - off)
        session, theme = enforce_session_rule(theme >= 0, theme)
        decs[h] = Decoded(name, theme.astype(np.int32), session, boundaries_from_theme_steps(th_steps, ts[h], off))
    return decs, lls
