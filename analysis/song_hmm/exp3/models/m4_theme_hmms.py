"""M4: six theme HMMs, trained one by one, then chained into one song model.

Like a speech recogniser that trains word models and then joins them into sentences.

Theme k = a left-to-right chain of its phrase slots from the drawing (3, 3, 6, 3, 5, 8).
Each slot is two states (a -> b), so a slot lasts at least 0.5 s; b loops while the
unit lasts, then the chain moves on. After the last slot the phrase repeats or the
theme ends. Input: X3 frame vectors (0.25 s). One 2-component Gaussian mixture per slot.

1. Isolated training: each theme HMM alone on the CSV segments of its theme (train half).
2. Detection check: each theme HMM scores every CSV segment of the test half; one-vs-rest AUC.
3. Chain: T1 -> ... -> T6 -> T1 (song boundary), pauses, background (>= 60 s), entry at any theme.
4. Embedded training: Baum-Welch on the whole train half; only the theme order is imposed.
5. Viterbi decoding.
"""

from __future__ import annotations

import numpy as np

from analysis.song_hmm.topology import FIRST_SLOT, LAST_SLOT, N_SLOTS, SLOT_THEME, THEME_N_SLOTS
from analysis.song_hmm_phrase.decode import Decoded

from .common import HALVES, N_THEMES, at_times, boundaries_from_theme_steps, enforce_session_rule, labels, level_1s, save, steps_to_seconds
from .graph import HMM, DiagGMM, Family, Graph

FRAME_S = 0.25
BG_MIN_STEPS = int(60 / FRAME_S)
END_PROB = 1e-4
BG_STAY = 1 - FRAME_S / 3600
BG_GROUP = N_SLOTS


def a_state(slot: int) -> int:
    return 2 * slot


def b_state(slot: int) -> int:
    return 2 * slot + 1


P0 = 2 * N_SLOTS
BG0 = P0 + N_THEMES
BG = BG0 + BG_MIN_STEPS - 1
N_STATES = BG + 1
STATE_GROUP = np.r_[np.repeat(np.arange(N_SLOTS), 2), np.full(N_THEMES + BG_MIN_STEPS, BG_GROUP)].astype(np.int32)
STATE_THEME = np.r_[np.repeat(SLOT_THEME, 2), np.arange(N_THEMES), np.full(BG_MIN_STEPS, -1)].astype(np.int32)


def families() -> dict[str, Family]:
    fam = {
        "one": Family("one", np.array([1.0]), fixed=True),
        "bg": Family("bg", np.array([BG_STAY, 1 - BG_STAY]), fixed=True),
        "bg_enter": Family("bg_enter", np.full(N_THEMES, 1 / N_THEMES)),
        "start": Family("start", np.full(N_THEMES + 1, 1 / (N_THEMES + 1))),
    }
    for s in range(N_SLOTS):
        fam[f"slot{s}"] = Family(f"slot{s}", np.array([0.6, 0.4]), outcomes=("stay", "leave"))
    for k in range(N_THEMES):
        fam[f"tstart{k}"] = Family(f"tstart{k}", np.full(THEME_N_SLOTS[k], 1 / THEME_N_SLOTS[k]))
        fam[f"theme_end{k}"] = Family(f"theme_end{k}", np.array([0.8, 0.2]), outcomes=("repeat", "exit"))
        fam[f"endfix{k}"] = Family(f"endfix{k}", np.array([1 - END_PROB, END_PROB]), fixed=True)
        fam[f"exit{k}"] = Family(f"exit{k}", np.array([0.9, 0.1]), outcomes=("next", "pause"))
        fam[f"pause{k}"] = Family(f"pause{k}", np.array([0.8, 0.15, 0.05]), outcomes=("stay", "resume", "next"))
    return fam


def _slots(g: Graph, k: int, combined: bool) -> None:
    first, last = int(FIRST_SLOT[k]), int(LAST_SLOT[k])
    for s in range(first, last + 1):
        g.add(a_state(s), b_state(s), ("one", 0))
        g.add(b_state(s), b_state(s), (f"slot{s}", 0))
        if s < last:
            g.add(b_state(s), a_state(s + 1), (f"slot{s}", 1))
    leave = (f"slot{last}", 1)
    if not combined:
        g.add(b_state(last), a_state(first), leave)  # a lone theme can only repeat its phrase
        return
    nxt = int(FIRST_SLOT[(k + 1) % N_THEMES])
    g.add(b_state(last), a_state(first), leave, (f"theme_end{k}", 0))
    g.add(b_state(last), a_state(nxt), leave, (f"theme_end{k}", 1), (f"endfix{k}", 0), (f"exit{k}", 0))
    g.add(b_state(last), P0 + k, leave, (f"theme_end{k}", 1), (f"endfix{k}", 0), (f"exit{k}", 1))
    g.add(b_state(last), BG0, leave, (f"theme_end{k}", 1), (f"endfix{k}", 1))
    g.add(P0 + k, P0 + k, (f"pause{k}", 0))
    g.add(P0 + k, a_state(first), (f"pause{k}", 1))
    g.add(P0 + k, a_state(nxt), (f"pause{k}", 2))


def theme_graph(k: int, fam: dict) -> Graph:
    g = Graph(N_STATES, STATE_GROUP, fam)
    _slots(g, k, combined=False)
    for i, s in enumerate(range(int(FIRST_SLOT[k]), int(LAST_SLOT[k]) + 1)):
        g.add_start(a_state(s), (f"tstart{k}", i))
    return g


def song_graph(fam: dict) -> Graph:
    g = Graph(N_STATES, STATE_GROUP, fam)
    for k in range(N_THEMES):
        _slots(g, k, combined=True)
        g.add(BG, a_state(int(FIRST_SLOT[k])), ("bg", 1), ("bg_enter", k))
        g.add_start(a_state(int(FIRST_SLOT[k])), ("start", 1 + k))
    for s in range(BG0, BG):
        g.add(s, s + 1, ("one", 0))
    g.add(BG, BG, ("bg", 0))
    g.add_start(BG, ("start", 0))
    return g


def runs(lab: np.ndarray, min_len: int):
    cut = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1], True])
    return [(int(a), int(b), int(lab[a])) for a, b in zip(cut[:-1], cut[1:], strict=True) if b - a >= min_len and lab[a] >= 0]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.r_[pos, neg]
    ranks = np.argsort(np.argsort(allv)) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def detection_aucs(fam, emis, x, theme_frames, max_segs: int = 400, seed: int = 0) -> dict:
    segs = runs(theme_frames, 8)
    rng = np.random.default_rng(seed)
    if len(segs) > max_segs:
        segs = [segs[i] for i in sorted(rng.choice(len(segs), max_segs, replace=False))]
    graphs = [theme_graph(k, fam) for k in range(N_THEMES)]
    score = np.array([[HMM(gk, emis).loglik(x[a:b]) / (b - a) for gk in graphs] for a, b, _ in segs])
    lab = np.array([k for _, _, k in segs])
    rel = score - score.mean(1, keepdims=True)
    out = {f"T{k + 1}": auc(rel[lab == k, k], rel[lab != k, k]) for k in range(N_THEMES)}
    out["top1_accuracy"] = float(np.mean(score.argmax(1) == lab))
    out["n_segments"] = len(segs)
    return out


def run(name: str, inp: dict, verbose: bool = True) -> dict:
    lab = labels()
    lev = level_1s()
    xs = {"train": inp["train"].astype(np.float64), "test": inp["test"].astype(np.float64)}
    ts = {h: off + (np.arange(len(xs[h])) + 0.5) * FRAME_S for h, off, _ in HALVES}
    fam = families()
    slot_tr = at_times(lab.slot, ts["train"])
    lv = at_times(lev, ts["train"])
    grp = np.where(lv <= np.quantile(lv, 0.10), BG_GROUP, slot_tr)
    emis = DiagGMM.from_assignment(xs["train"], np.where(grp < 0, BG_GROUP, grp), N_SLOTS + 1)

    # 1. isolated training per theme
    th_tr = at_times(lab.theme, ts["train"])
    iso_hist = {}
    for k in range(N_THEMES):
        segs = [xs["train"][a:b] for a, b, kk in runs(th_tr, 4) if kk == k]
        iso_hist[k] = HMM(theme_graph(k, fam), emis).fit(segs, max_iter=15)
        if verbose:
            print(f"  theme {k + 1}: {len(segs)} segments, isolated loglik {iso_hist[k][0]:,.0f} -> {iso_hist[k][-1]:,.0f}", flush=True)
    th_te = at_times(lab.theme, ts["test"])
    det_iso = detection_aucs(fam, emis, xs["test"], th_te)

    # 2. chain and embedded training
    song = HMM(song_graph(fam), emis)
    hist = song.fit([xs["train"]], max_iter=25, verbose=verbose)
    det_emb = detection_aucs(fam, emis, xs["test"], th_te)

    decs, lls = {}, {}
    for h, off, end in HALVES:
        path = song.viterbi(xs[h])
        lls[h] = song.loglik(xs[h]) / len(xs[h])
        th_steps = STATE_THEME[path]
        theme = steps_to_seconds(th_steps, ts[h], off, end - off)
        session, theme = enforce_session_rule(theme >= 0, theme)
        decs[h] = Decoded(name, theme.astype(np.int32), session, boundaries_from_theme_steps(th_steps, ts[h], off))
    save(name, decs)
    return {"decoded": decs, "history": hist, "isolated_history": {str(k): v for k, v in iso_hist.items()},
            "detection_isolated": det_iso, "detection_embedded": det_emb, "loglik_per_step": lls, "hmm": song}
