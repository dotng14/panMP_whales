"""Phrase-slot topology for song_soft_hmm_t6.dot.

A slot is one node on the drawing. It expands to five duration states
(1 s through 5 s) that share an emission. Advance edges fire when the
slot ends. Only the last slot of a theme branches, into repeat versus next.
Theme 6 repeats until the song ends and has no edge back to theme 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_THEMES = 6
N_DUR = 5
THEME_N_SLOTS = (3, 3, 6, 3, 5, 8)
# Repeat weights printed on the drawing. Next is 1 - repeat, except theme 6.
REPEAT_INIT = np.array([0.008, 0.984, 0.943, 0.585, 0.922, 1.0], dtype=np.float64)

# Unit name and nominal duration (seconds) printed on each node, theme by theme.
SLOT_UNITS: tuple[str, ...] = (
    "U7", "U2", "U9",
    "U1", "U8", "U7",
    "U9", "U11", "U9", "U6", "U8", "U7",
    "U2", "U9", "U6",
    "U8", "U9", "U3", "U8", "U7",
    "U9", "U3", "U8", "U2", "U7", "U9", "U11", "U2",
)
NOMINAL_DURATION_S: tuple[float, ...] = (
    2.0, 1.7, 1.2,
    1.0, 1.0, 1.6,
    1.3, 1.2, 1.0, 1.0, 1.0, 1.1,
    1.2, 1.0, 1.0,
    1.2, 1.1, 1.0, 1.0, 1.1,
    1.1, 1.0, 1.6, 2.0, 1.1, 2.0, 1.0, 1.3,
)

N_SLOTS = len(SLOT_UNITS)
N_STATES = N_SLOTS * N_DUR


def _theme_index() -> np.ndarray:
    theme = np.empty(N_SLOTS, dtype=np.int32)
    cursor = 0
    for theme_i, n in enumerate(THEME_N_SLOTS):
        theme[cursor : cursor + n] = theme_i
        cursor += n
    return theme


SLOT_THEME = _theme_index()
FIRST_SLOT = np.array([0, 3, 6, 12, 15, 20], dtype=np.int32)
TERMINAL_SLOT = FIRST_SLOT + np.array(THEME_N_SLOTS, dtype=np.int32) - 1


def state_index(slot: int, duration_index: int) -> int:
    """duration_index is 0..4 for 1..5 seconds."""
    return int(slot) * N_DUR + int(duration_index)


# A song enters at the 1-second state of the first slot of any theme.
START_STATES = np.array([state_index(int(slot), 0) for slot in FIRST_SLOT], dtype=np.int32)
START_STATE = int(START_STATES[0])


def exit_probs_from_duration_hist(hist5: np.ndarray) -> np.ndarray:
    """Map a 5-bin length histogram to P(advance) at durations 1..4.

    Duration 5 always advances. ``P(D = k)`` for a left-to-right chain is
    the product of the continues up to k and the exit at k.
    """
    h = np.asarray(hist5, dtype=np.float64)
    h = np.maximum(h, 0.0)
    if float(h.sum()) <= 0.0:
        h = np.ones(N_DUR, dtype=np.float64)
    h = h / h.sum()
    exit_p = np.zeros(N_DUR - 1, dtype=np.float64)
    remain = 1.0
    for k in range(N_DUR - 1):
        if remain <= 1e-12:
            exit_p[k] = 1.0
        else:
            exit_p[k] = h[k] / remain
        remain -= h[k]
    return np.clip(exit_p, 1e-4, 1.0 - 1e-4)


@dataclass(frozen=True)
class TransitionTables:
    """Sparse outgoing edges. At most three successors per state."""

    to: np.ndarray  # (N_STATES, 3) int32
    logp: np.ndarray  # (N_STATES, 3) float64
    n_out: np.ndarray  # (N_STATES,) int32
    kind: np.ndarray  # (N_STATES, 3) int8
    edge_id: np.ndarray  # (N_STATES, 3) int32
    n_edges: int


# kind codes consumed by the M-step
KIND_CONTINUE = np.int8(0)
KIND_ADVANCE = np.int8(1)
KIND_REPEAT = np.int8(2)
KIND_NEXT = np.int8(3)


def transition_tables(exit_prob: np.ndarray, repeat_prob: np.ndarray) -> TransitionTables:
    """Build the sparse graph. Structural zeros are omitted, not given mass."""
    exit_prob = np.asarray(exit_prob, dtype=np.float64)
    repeat_prob = np.asarray(repeat_prob, dtype=np.float64)
    if exit_prob.shape != (N_SLOTS, N_DUR - 1):
        raise ValueError(f"exit_prob shape {exit_prob.shape}")
    if repeat_prob.shape != (N_THEMES,):
        raise ValueError(f"repeat_prob shape {repeat_prob.shape}")

    to = np.zeros((N_STATES, 3), dtype=np.int32)
    logp = np.full((N_STATES, 3), -np.inf, dtype=np.float64)
    n_out = np.zeros(N_STATES, dtype=np.int32)
    kind = np.zeros((N_STATES, 3), dtype=np.int8)
    edge_id = np.full((N_STATES, 3), -1, dtype=np.int32)
    n_edges = 0

    def add(src: int, dst: int, prob: float, edge_kind: int) -> None:
        nonlocal n_edges
        if prob <= 0.0:
            return
        k = int(n_out[src])
        if k >= 3:
            raise RuntimeError(f"state {src} has more than 3 outgoing edges")
        to[src, k] = dst
        logp[src, k] = np.log(prob)
        kind[src, k] = edge_kind
        edge_id[src, k] = n_edges
        n_out[src] = k + 1
        n_edges += 1

    for slot in range(N_SLOTS):
        theme = int(SLOT_THEME[slot])
        terminal = slot == int(TERMINAL_SLOT[theme])
        for dur in range(N_DUR):
            src = state_index(slot, dur)
            if dur == N_DUR - 1:
                leave = 1.0
            else:
                leave = float(exit_prob[slot, dur])
            stay = 1.0 - leave
            if dur < N_DUR - 1 and stay > 0.0:
                add(src, state_index(slot, dur + 1), stay, int(KIND_CONTINUE))
            if leave <= 0.0:
                continue
            if not terminal:
                add(src, state_index(slot + 1, 0), leave, int(KIND_ADVANCE))
                continue
            p_rep = float(repeat_prob[theme])
            add(src, state_index(int(FIRST_SLOT[theme]), 0), leave * p_rep, int(KIND_REPEAT))
            if theme < N_THEMES - 1:
                p_next = 1.0 - p_rep
                add(
                    src,
                    state_index(int(FIRST_SLOT[theme + 1]), 0),
                    leave * p_next,
                    int(KIND_NEXT),
                )
    return TransitionTables(to, logp, n_out, kind, edge_id, n_edges)


def row_probability_sums(tables: TransitionTables) -> np.ndarray:
    totals = np.zeros(N_STATES, dtype=np.float64)
    for i in range(N_STATES):
        for k in range(int(tables.n_out[i])):
            totals[i] += np.exp(tables.logp[i, k])
    return totals
