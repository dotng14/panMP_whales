"""Theme-level state graph. One step = one phrase window = 2 s.

States (81):
  0..59   themes: theme k is a chain of MIN_THEME states; the last one (its
          "dwell" state) loops while the whale keeps repeating the phrase
  60..65  breath / pause P1..P6
  66..80  background: a chain of MIN_BG states, the last one loops

From a theme's dwell state: stay, next theme (theme 6 -> theme 1 is the song
boundary), breath, or end the session. Background can enter any theme.
No theme is skipped and the order never goes backward.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

N_THEMES = 6
MIN_THEME = 10  # steps: a theme lasts at least 20 s
MIN_BG = 15  # steps: background lasts at least 30 s
STEP_S = 2

N_THEME_STATES = N_THEMES * MIN_THEME
PAUSE0 = N_THEME_STATES
BG_FIRST = PAUSE0 + N_THEMES
BG = BG_FIRST + MIN_BG - 1
N_STATES = BG + 1

N_GROUPS = N_THEMES + 1  # one Gaussian per theme + background (shared with pauses)
BG_GROUP = N_THEMES

ENTRY = np.arange(N_THEMES) * MIN_THEME  # first state of each theme chain
DWELL = ENTRY + MIN_THEME - 1
STATE_THEME = np.concatenate([np.repeat(np.arange(N_THEMES), MIN_THEME), np.arange(N_THEMES), np.full(MIN_BG, -1)]).astype(np.int32)
STATE_GROUP = np.where(STATE_THEME >= 0, STATE_THEME, BG_GROUP).astype(np.int32)
STATE_GROUP[PAUSE0:BG_FIRST] = BG_GROUP
IS_BG = np.arange(N_STATES) >= BG_FIRST

CHAIN, STAY, NEXT, BREATH, END, P_STAY, P_RESUME, P_NEXT, BG_CHAIN, BG_STAY, BG_ENTER = range(11)
DWELL_KINDS = (STAY, NEXT, BREATH, END)  # columns of Params.dwell
PAUSE_KINDS = (P_STAY, P_RESUME, P_NEXT)  # columns of Params.pause


@dataclass
class Params:
    means: np.ndarray  # (N_GROUPS, D)
    vars: np.ndarray  # (N_GROUPS, D)
    dwell: np.ndarray  # (N_THEMES, 4) stay, next, breath, end (end fixed)
    pause: np.ndarray  # (N_THEMES, 3) stay, resume, next
    bg: np.ndarray  # (1 + N_THEMES,) stay (fixed), enter theme 1..6
    start: np.ndarray  # (1 + N_THEMES,) background, theme 1..6

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
    src: np.ndarray
    dst: np.ndarray
    prob: np.ndarray
    kind: np.ndarray
    theme: np.ndarray  # theme the edge belongs to (dwell / pause) or enters (background); -1 otherwise
    start: np.ndarray


def build_graph(p: Params) -> Graph:
    rows: list[tuple[int, int, float, int, int]] = []

    def add(s, d, q, k, th=-1):
        if q > 0.0:
            rows.append((s, d, float(q), k, th))

    for th in range(N_THEMES):
        nxt = (th + 1) % N_THEMES
        for d in range(MIN_THEME - 1):
            add(ENTRY[th] + d, ENTRY[th] + d + 1, 1.0, CHAIN, th)
        stay, go, breath, end = p.dwell[th]
        add(DWELL[th], DWELL[th], stay, STAY, th)
        add(DWELL[th], ENTRY[nxt], go, NEXT, th)
        add(DWELL[th], PAUSE0 + th, breath, BREATH, th)
        add(DWELL[th], BG_FIRST, end, END, th)
        p_stay, p_resume, p_next = p.pause[th]
        add(PAUSE0 + th, PAUSE0 + th, p_stay, P_STAY, th)
        add(PAUSE0 + th, ENTRY[th], p_resume, P_RESUME, th)
        add(PAUSE0 + th, ENTRY[nxt], p_next, P_NEXT, th)
    for s in range(BG_FIRST, BG):
        add(s, s + 1, 1.0, BG_CHAIN)
    add(BG, BG, p.bg[0], BG_STAY)
    for th in range(N_THEMES):
        add(BG, ENTRY[th], p.bg[1 + th], BG_ENTER, th)

    start = np.zeros(N_STATES)
    start[BG] = p.start[0]
    start[ENTRY] = p.start[1:]
    src, dst, prob, kind, theme = (np.array(c) for c in zip(*rows, strict=True))
    return Graph(src.astype(np.int32), dst.astype(np.int32), prob.astype(np.float64), kind.astype(np.int8), theme.astype(np.int32), start)
