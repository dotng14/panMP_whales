"""State graph of the song-session HMM.

States (176):
  0..139   phrase slots: 28 slots x 5 duration states (1..5 s)
  140..145 breath / pause P1..P6, one per theme
  146..175 background: a chain of BG_MIN_S states, so background lasts at
           least 30 s; the last one (BG) loops and can enter a theme

Themes run in a fixed cycle 1 -> 2 -> ... -> 6 -> 1. Only the last slot of a
theme branches: repeat the theme, go to the next theme, breathe, or end the
session. The theme 6 -> theme 1 edge (from T6.last or P6) is the song boundary.
BG may enter the first slot of any theme, so a session can begin mid-song.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

N_THEMES = 6
N_DUR = 5
THEME_N_SLOTS = (3, 3, 6, 3, 5, 8)
SLOT_UNITS = (
    "U7", "U2", "U9",
    "U1", "U8", "U7",
    "U9", "U11", "U9", "U6", "U8", "U7",
    "U2", "U9", "U6",
    "U8", "U9", "U3", "U8", "U7",
    "U9", "U3", "U8", "U2", "U7", "U9", "U11", "U2",
)  # fmt: skip
N_SLOTS = len(SLOT_UNITS)
FIRST_SLOT = np.cumsum((0,) + THEME_N_SLOTS[:-1]).astype(np.int32)
LAST_SLOT = FIRST_SLOT + np.array(THEME_N_SLOTS, dtype=np.int32) - 1
SLOT_THEME = np.repeat(np.arange(N_THEMES), THEME_N_SLOTS).astype(np.int32)

N_SONG_STATES = N_SLOTS * N_DUR
PAUSE0 = N_SONG_STATES
BG_MIN_S = 30
BG_FIRST = PAUSE0 + N_THEMES
BG = BG_FIRST + BG_MIN_S - 1
N_STATES = BG + 1

# Emission groups: one Gaussian per slot, plus one shared by the pauses and BG.
N_GROUPS = N_SLOTS + 1
BG_GROUP = N_SLOTS


def slot_state(slot: int, dur: int = 0) -> int:
    """``dur`` is 0..4 for 1..5 s."""
    return int(slot) * N_DUR + int(dur)


ENTRY_STATES = np.array([slot_state(s) for s in FIRST_SLOT], dtype=np.int32)
N_OTHER = N_STATES - N_SONG_STATES
STATE_GROUP = np.concatenate([np.repeat(np.arange(N_SLOTS), N_DUR), np.full(N_OTHER, BG_GROUP)]).astype(np.int32)
STATE_SLOT = np.concatenate([np.repeat(np.arange(N_SLOTS), N_DUR), np.full(N_OTHER, -1)]).astype(np.int32)
STATE_THEME = np.concatenate([SLOT_THEME[STATE_SLOT[:N_SONG_STATES]], np.arange(N_THEMES), np.full(BG_MIN_S, -1)]).astype(np.int32)
IS_BG = np.arange(N_STATES) >= BG_FIRST

# Edge kinds, used by the M-step.
CONT, ADV, REPEAT, NEXT, BREATH, END, P_STAY, P_RESUME, P_NEXT, BG_STAY, BG_ENTER, BG_CHAIN = range(12)
BRANCH_KINDS = (REPEAT, NEXT, BREATH, END)  # column order of Params.branch
PAUSE_KINDS = (P_STAY, P_RESUME, P_NEXT)  # column order of Params.pause


@dataclass
class Params:
    means: np.ndarray  # (N_GROUPS, K)
    vars: np.ndarray  # (N_GROUPS, K)
    exit_prob: np.ndarray  # (N_SLOTS, N_DUR - 1) P(leave slot) at durations 1..4 s
    branch: np.ndarray  # (N_THEMES, 4) repeat, next, breath, end
    pause: np.ndarray  # (N_THEMES, 3) stay, resume, next
    bg: np.ndarray  # (1 + N_THEMES,) stay, enter theme 1..6
    start: np.ndarray  # (1 + N_THEMES,) BG, theme 1..6

    def copy(self) -> "Params":
        return Params(**{f.name: getattr(self, f.name).copy() for f in fields(self)})

    def save(self, path: Path, **extra) -> None:
        np.savez(path, **{f.name: getattr(self, f.name) for f in fields(self)}, **extra)

    @classmethod
    def load(cls, path: Path) -> "Params":
        with np.load(path) as data:
            return cls(**{f.name: data[f.name] for f in fields(cls)})


@dataclass(frozen=True)
class Graph:
    src: np.ndarray  # (E,) int32
    dst: np.ndarray  # (E,) int32
    prob: np.ndarray  # (E,) float64
    kind: np.ndarray  # (E,) int8
    theme: np.ndarray  # (E,) int32, theme of the branch / pause / entered theme; -1 otherwise
    start: np.ndarray  # (N_STATES,) initial probability

    @property
    def is_boundary(self) -> np.ndarray:
        """Edges that start a new song: theme 6 -> T1.1."""
        return ((self.kind == NEXT) | (self.kind == P_NEXT)) & (self.theme == N_THEMES - 1)


def build_graph(p: Params) -> Graph:
    src: list[int] = []
    dst: list[int] = []
    prob: list[float] = []
    kind: list[int] = []
    theme: list[int] = []

    def add(s: int, d: int, q: float, k: int, th: int = -1) -> None:
        if q > 0.0:
            src.append(s)
            dst.append(d)
            prob.append(q)
            kind.append(k)
            theme.append(th)

    for slot in range(N_SLOTS):
        th = int(SLOT_THEME[slot])
        last = slot == int(LAST_SLOT[th])
        nxt = (th + 1) % N_THEMES
        for dur in range(N_DUR):
            s = slot_state(slot, dur)
            leave = 1.0 if dur == N_DUR - 1 else float(p.exit_prob[slot, dur])
            if dur < N_DUR - 1:
                add(s, s + 1, 1.0 - leave, CONT)
            if not last:
                add(s, slot_state(slot + 1), leave, ADV)
                continue
            rep, go, breath, end = (float(v) for v in p.branch[th])
            add(s, int(ENTRY_STATES[th]), leave * rep, REPEAT, th)
            add(s, int(ENTRY_STATES[nxt]), leave * go, NEXT, th)
            add(s, PAUSE0 + th, leave * breath, BREATH, th)
            add(s, BG_FIRST, leave * end, END, th)

    for th in range(N_THEMES):
        stay, resume, go = (float(v) for v in p.pause[th])
        add(PAUSE0 + th, PAUSE0 + th, stay, P_STAY, th)
        add(PAUSE0 + th, int(ENTRY_STATES[th]), resume, P_RESUME, th)
        add(PAUSE0 + th, int(ENTRY_STATES[(th + 1) % N_THEMES]), go, P_NEXT, th)

    for s in range(BG_FIRST, BG):
        add(s, s + 1, 1.0, BG_CHAIN)
    add(BG, BG, float(p.bg[0]), BG_STAY)
    for th in range(N_THEMES):
        add(BG, int(ENTRY_STATES[th]), float(p.bg[1 + th]), BG_ENTER, th)

    start = np.zeros(N_STATES)
    start[BG] = p.start[0]
    start[ENTRY_STATES] = p.start[1:]
    return Graph(
        np.array(src, dtype=np.int32),
        np.array(dst, dtype=np.int32),
        np.array(prob, dtype=np.float64),
        np.array(kind, dtype=np.int8),
        np.array(theme, dtype=np.int32),
        start,
    )


def exit_prob_from_lengths(hist: np.ndarray) -> np.ndarray:
    """P(leave) at durations 1..4 s so that the slot length follows ``hist`` (5 bins)."""
    h = np.maximum(np.asarray(hist, dtype=np.float64), 0.0) + 0.5  # light smoothing
    h /= h.sum()
    survive = 1.0 - np.concatenate([[0.0], np.cumsum(h)[:-1]])
    return np.clip(h[: N_DUR - 1] / survive[: N_DUR - 1], 1e-3, 1.0 - 1e-3)


def duration_pmf(exit_prob_row: np.ndarray) -> np.ndarray:
    """Slot length distribution over 1..5 s implied by one row of ``exit_prob``."""
    pmf = np.zeros(N_DUR)
    remain = 1.0
    for d in range(N_DUR - 1):
        pmf[d] = remain * exit_prob_row[d]
        remain *= 1.0 - exit_prob_row[d]
    pmf[-1] = remain
    return pmf
