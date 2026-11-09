"""M3: unit-based model. One step = one detected unit (X4), plus gap tokens.

Observation sequence: the unit types of the detected units in time order, with a
token inserted for every silence between units:
  GAP  a pause of 3-60 s (a breath; the session goes on)
  BGT  a pause longer than 60 s (the session ends)
Model: the 28 phrase slots of the drawing, one state each, emitting a unit type
(categorical). A slot may repeat (one sung unit detected as several pieces). After
a theme's last slot: repeat the phrase / next theme (T6 -> T1 is the song boundary)
/ pause (P_k emits GAP) / end (background emits BGT and loops).
Seeded from the CSV slot at each unit's time (train half).
"""

from __future__ import annotations

import numpy as np

from analysis.song_hmm.exp1_slot_hmm.topology import FIRST_SLOT, LAST_SLOT, N_SLOTS, SLOT_THEME
from analysis.song_hmm.exp2_phrase_hmm.decode import Decoded

from .common import HALVES, N_THEMES, SESSION_GAP_S, at_times, enforce_session_rule, labels, save
from .graph import HMM, Categorical, Family, Graph

GAP_MIN_S = 3.0
END_PROB = 1e-3
P0 = N_SLOTS
BG = P0 + N_THEMES
N_STATES = BG + 1
PAUSE_GROUP, BG_GROUP = N_SLOTS, N_SLOTS + 1
STATE_GROUP = np.r_[np.arange(N_SLOTS), np.full(N_THEMES, PAUSE_GROUP), [BG_GROUP]].astype(np.int32)
STATE_THEME = np.r_[SLOT_THEME, np.arange(N_THEMES), [-1]].astype(np.int32)


def token_sequence(units: np.ndarray, types: np.ndarray, k: int):
    """(tokens, start times, end times). GAP = k, BGT = k + 1."""
    order = np.argsort(units["start_s"])
    u, ty = units[order], types[order]
    tok, t0, t1 = [], [], []
    for i in range(len(u)):
        if i > 0:
            gap = u["start_s"][i] - u["end_s"][i - 1]
            if gap > GAP_MIN_S:
                tok.append(k + (1 if gap > SESSION_GAP_S else 0))
                t0.append(u["end_s"][i - 1])
                t1.append(u["start_s"][i])
        tok.append(int(ty[i]))
        t0.append(u["start_s"][i])
        t1.append(u["end_s"][i])
    return np.array(tok), np.array(t0), np.array(t1)


def build_graph() -> Graph:
    fam = {"one": Family("one", np.array([1.0]), fixed=True), "start": Family("start", np.full(N_THEMES + 1, 1 / (N_THEMES + 1))),
           "bg": Family("bg", np.array([0.5, 0.5])), "bg_enter": Family("bg_enter", np.full(N_THEMES, 1 / N_THEMES))}
    g = Graph(N_STATES, STATE_GROUP, fam)
    for s in range(N_SLOTS):
        fam[f"slot{s}"] = Family(f"slot{s}", np.array([0.2, 0.8]), outcomes=("again", "leave"))
        g.add(s, s, (f"slot{s}", 0))
    for k in range(N_THEMES):
        first, last, nxt = int(FIRST_SLOT[k]), int(LAST_SLOT[k]), int(FIRST_SLOT[(k + 1) % N_THEMES])
        for s in range(first, last):
            g.add(s, s + 1, (f"slot{s}", 1))
        fam[f"theme_end{k}"] = Family(f"theme_end{k}", np.array([0.7, 0.2, 0.1]), outcomes=("repeat", "next", "pause"))
        fam[f"endfix{k}"] = Family(f"endfix{k}", np.array([1 - END_PROB, END_PROB]), fixed=True)
        fam[f"pause{k}"] = Family(f"pause{k}", np.array([0.2, 0.6, 0.2]), outcomes=("stay", "resume", "next"))
        leave = (f"slot{last}", 1)
        g.add(last, first, leave, (f"endfix{k}", 0), (f"theme_end{k}", 0))
        g.add(last, nxt, leave, (f"endfix{k}", 0), (f"theme_end{k}", 1))
        g.add(last, P0 + k, leave, (f"endfix{k}", 0), (f"theme_end{k}", 2))
        g.add(last, BG, leave, (f"endfix{k}", 1))
        g.add(P0 + k, P0 + k, (f"pause{k}", 0))
        g.add(P0 + k, first, (f"pause{k}", 1))
        g.add(P0 + k, nxt, (f"pause{k}", 2))
        g.add(BG, first, ("bg", 1), ("bg_enter", k))
        g.add_start(first, ("start", 1 + k))
    g.add(BG, BG, ("bg", 0))
    g.add_start(BG, ("start", 0))
    return g


def run(name: str, units: dict, types: dict, k_types: int, verbose: bool = True) -> dict:
    lab = labels()
    seq = {h: token_sequence(units[h], types[h], k_types) for h in ("train", "test")}
    tok_tr, t0_tr, t1_tr = seq["train"]
    mid = 0.5 * (t0_tr + t1_tr)
    grp = at_times(lab.slot, mid)
    grp = np.where(tok_tr == k_types, PAUSE_GROUP, np.where(tok_tr == k_types + 1, BG_GROUP, grp))
    g = build_graph()
    emis = Categorical.from_assignment(tok_tr, grp, N_SLOTS + 2, k_types + 2)
    hmm = HMM(g, emis)
    hist = hmm.fit([tok_tr], max_iter=40, verbose=verbose)
    decs, lls = {}, {}
    for h, off, end in HALVES:
        tok, t0, t1 = seq[h]
        path = hmm.viterbi(tok)
        lls[h] = hmm.loglik(tok) / len(tok)
        th = STATE_THEME[path]
        n = end - off
        theme = np.full(n, -1, np.int32)
        for i in range(len(tok)):  # each step covers its own span and the time up to the next step
            a = int(np.floor(t0[i] - off))
            b = int(np.ceil((t0[i + 1] if i + 1 < len(tok) else t1[i]) - off))
            theme[max(a, 0) : min(b, n)] = th[i]
        if len(tok):
            theme[: max(int(t0[0] - off), 0)] = -1
        session, theme = enforce_session_rule(theme >= 0, theme)
        ok = (th[1:] == 0) & (th[:-1] == N_THEMES - 1)
        bnd = np.rint(t0[1:][ok] - off).astype(int)
        decs[h] = Decoded(name, theme, session, bnd)
    save(name, decs)
    return {"decoded": decs, "history": hist, "loglik_per_step": lls, "hmm": hmm, "n_tokens": {h: int(len(seq[h][0])) for h in seq}}
