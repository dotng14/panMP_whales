"""M2: song-boundary-first. Find the song cycle from the audio's own repetition, then the themes.

1. Song period P: the lag (7-20 min) at which window vectors are most similar to themselves.
2. Song template: P/2 s positions ("phases"), started from one cycle of the train half.
3. Cyclic alignment: a Viterbi over phases. Each window either stays on its phase, moves
   one phase, or two (the song may be sung up to twice as slow or fast as the template);
   after the last phase the cycle wraps to the first. Score = cosine of the window with
   the template at that phase. The template is re-estimated from the aligned windows
   (train half only, a few rounds); the test half is aligned to the final template.
4. Themes: the template phases are split into 6 consecutive parts (T1..T6) in cycle
   order, choosing the start phase and the 5 cut points that best match the CSV themes
   on the train half (the CSV only names the parts). Each part may be a single phase, so
   single-phrase themes fit. The song boundary is where the cycle passes from T6 to T1.
5. Sessions: background = a run of more than 60 s in which the audio does not match the
   song template (alignment score below the train median minus 3 robust spreads). The
   recording has no silent loudness mode, so the content decides, not the level.
"""

from __future__ import annotations

import numpy as np

from analysis.song_hmm_phrase.decode import Decoded

from .common import HALVES, N_THEMES, SESSION_GAP_S, at_times, labels, save, steps_to_seconds

STEP_S = 2.0
BETA = 12.0  # sharpness of the alignment score (chosen on the train half, see run)
MOVES = (0, 1, 2)
LOG_MOVE = np.log(np.array([0.05, 0.9, 0.05]))
# Candidate (tempo prior, sharpness). Repetitive themes make "stay on this phase" cheap, so with a
# weak tempo prior the alignment drifts through the template at a fraction of real speed. The pair
# whose train-half alignment completes as many cycles as the song period predicts is kept.
TUNING = [((0.05, 0.9, 0.05), b) for b in (12.0, 6.0, 3.0)] + [((0.02, 0.96, 0.02), b) for b in (12.0, 6.0, 3.0)]


def unit_rows(z):
    z = z - z.mean(0)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)


def song_period(u: np.ndarray, lo_min=7, hi_min=20, block_min=40) -> int:
    """Median over 40-minute blocks of the lag (7-20 min) at which the block is most similar to itself.

    Songs vary in length, so over 5 h the similarity curve is a plateau; block by block it peaks."""
    lags = np.arange(int(lo_min * 60 / STEP_S), int(hi_min * 60 / STEP_S), 5)
    blk = int(block_min * 60 / STEP_S)
    per = []
    for s0 in range(0, len(u) - blk + 1, blk // 2):
        b = u[s0 : s0 + blk + lags[-1]]
        sims = [np.mean(np.sum(b[: blk] * b[L : L + blk], 1)) for L in lags if L + blk <= len(b)]
        if sims:
            per.append(lags[int(np.argmax(sims))])
    return int(np.median(per))


def align(u: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, float]:
    """Cyclic Viterbi over template phases. Returns (phase per window, mean score)."""
    P = len(template)
    e = BETA * (u @ template.T)  # (T, P)
    T = len(u)
    back = np.zeros((T, P), np.int8)
    delta = e[0].copy()
    for t in range(1, T):
        cand = np.stack([np.roll(delta, m) + LOG_MOVE[i] for i, m in enumerate(MOVES)])
        back[t] = np.argmax(cand, 0)
        delta = cand[back[t], np.arange(P)] + e[t]
    phase = np.empty(T, np.int64)
    phase[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        phase[t - 1] = (phase[t] - MOVES[back[t, phase[t]]]) % P
    return phase, float(np.mean(np.sum(u * template[phase], 1)))


def fit_template(u: np.ndarray, P: int, rounds: int = 6) -> tuple[np.ndarray, list[float]]:
    lags_ok = len(u) - 2 * P
    best = max(range(0, max(lags_ok, 1), 30), key=lambda s: np.mean(np.sum(u[s : s + P] * u[s + P : s + 2 * P], 1)))
    template = u[best : best + P].copy()
    hist = []
    for _ in range(rounds):
        phase, score = align(u, template)
        hist.append(score)
        new = np.zeros_like(template)
        np.add.at(new, phase, u)
        cnt = np.bincount(phase, minlength=P)
        filled = cnt > 0
        new[filled] /= cnt[filled, None]
        new[~filled] = template[~filled]
        template = unit_rows(new)
    return template, hist


def name_phases(phase_theme_counts: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Split the cyclic phases into T1..T6 (contiguous, in order) to best match per-phase theme counts.

    phase_theme_counts: (P, 6). Returns (theme per phase, start phase, share of counts matched).
    """
    P = len(phase_theme_counts)
    best_score, best_map, best_o = -1.0, None, 0
    for o in range(P):
        c = np.roll(phase_theme_counts, -o, axis=0)
        cum = np.vstack([np.zeros(N_THEMES), np.cumsum(c, 0)])  # prefix sums per theme
        # dp[k][j]: best score covering phases [0, j) with themes 0..k-1
        dp = np.full((N_THEMES + 1, P + 1), -np.inf)
        arg = np.zeros((N_THEMES + 1, P + 1), int)
        dp[0, 0] = 0.0
        for k in range(1, N_THEMES + 1):
            for j in range(k, P - (N_THEMES - k) + 1):
                vals = dp[k - 1, k - 1 : j] + (cum[j, k - 1] - cum[k - 1 : j, k - 1])
                i = int(np.argmax(vals))
                dp[k, j], arg[k, j] = vals[i], i + k - 1
        if dp[N_THEMES, P] > best_score:
            m = np.empty(P, int)
            j = P
            for k in range(N_THEMES, 0, -1):
                i = arg[k, j]
                m[i:j] = k - 1
                j = i
            best_score, best_map, best_o = dp[N_THEMES, P], np.roll(m, o), o
    return best_map, best_o, float(best_score / max(phase_theme_counts.sum(), 1))


def run(name: str, inp: dict, n_pc: int = 40, verbose: bool = True) -> dict:
    lab = labels()
    u = {"train": unit_rows(inp["train"][:, :n_pc].astype(np.float64)), "test": unit_rows(inp["test"][:, :n_pc].astype(np.float64))}
    ts = {"train": inp["t_train"], "test": inp["t_test"]}
    global BETA, LOG_MOVE
    P = song_period(u["train"])
    expected = len(u["train"]) / P
    tried = []
    for probs, beta in TUNING:
        BETA, LOG_MOVE = beta, np.log(np.array(probs))
        tpl, hist_ = fit_template(u["train"], P, rounds=4)
        ph, _ = align(u["train"], tpl)
        cycles = float((np.diff(ph) % P).sum() / P)
        tried.append({"tempo_prior": probs, "beta": beta, "train_cycles": cycles, "score": hist_[-1], "template": tpl, "hist": hist_})
    pick = min(tried, key=lambda r: abs(r["train_cycles"] - expected))
    BETA, LOG_MOVE = pick["beta"], np.log(np.array(pick["tempo_prior"]))
    template, hist = pick["template"], pick["hist"]
    phase_tr, _ = align(u["train"], template)
    th_tr = at_times(lab.theme, ts["train"])
    counts = np.zeros((P, N_THEMES))
    ok = th_tr >= 0
    np.add.at(counts, (phase_tr[ok], th_tr[ok]), 1.0)
    phase_theme, origin, matched = name_phases(counts)
    score_tr = np.sum(u["train"] * template[phase_tr], 1)
    med = np.median(score_tr)
    thr = float(med - 3 * 1.4826 * np.median(np.abs(score_tr - med)))
    if verbose:
        print(f"  song period {P * STEP_S / 60:.1f} min, template score {hist[0]:.3f} -> {hist[-1]:.3f}, CSV themes matched by the phase split {matched:.2f}", flush=True)
    decs, extra = {}, {}
    for h, off, end in HALVES:
        phase, score = (phase_tr, None) if h == "train" else align(u["test"], template)
        th_steps = phase_theme[phase]
        n = end - off
        theme = steps_to_seconds(th_steps, ts[h], off, n).astype(np.int32)
        match = np.sum(u[h] * template[phase], 1)
        quiet = steps_to_seconds(match < thr, ts[h], off, n)
        session = np.ones(n, bool)
        e = np.flatnonzero(np.diff(np.r_[0, quiet.astype(np.int8), 0])).reshape(-1, 2)
        for a, b in e:
            if b - a > SESSION_GAP_S:
                session[a:b] = False
        theme[~session] = -1
        rel = (phase - origin) % P  # position inside the song, 0 = song start
        wrap = rel[1:] < rel[:-1] - P // 2  # passed the song start (T1 may be a single phase and be stepped over)
        bnd = np.rint(ts[h][1:][wrap] - off).astype(int)
        bnd = bnd[session[np.clip(bnd, 0, n - 1)]]
        decs[h] = Decoded(name, theme, session, bnd)
        extra[h] = {"mean_alignment_score": float(np.mean(np.sum(u[h] * template[phase], 1)))}
    save(name, decs)
    return {"decoded": decs, "period_min": P * STEP_S / 60, "expected_train_cycles": expected,
            "tuning": [{k: v for k, v in r.items() if k not in ("template", "hist")} for r in tried],
            "chosen": {"tempo_prior": pick["tempo_prior"], "beta": pick["beta"]}, "template_history": hist, "phase_split_match": matched,
            "theme_phases": np.bincount(phase_theme, minlength=N_THEMES).tolist(), "alignment": extra, "match_threshold": thr}
