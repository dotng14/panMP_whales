"""Turn a decoded state path into sessions and songs.

A session is a maximal run of non-background states (breaths included).
A new song starts at T1.1 when the previous second was in theme 6 (its last
slot or its breath). The segment before the first such boundary is a partial
song unless the session itself opened at T1.1.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .topology import ENTRY_STATES, IS_BG, N_THEMES, STATE_THEME


@dataclass
class Session:
    session: int
    start_s: int
    end_s: int  # exclusive


@dataclass
class Song:
    session: int
    song: int
    start_s: int
    end_s: int  # exclusive
    first_theme: int  # 1-based
    last_theme: int
    partial_start: bool
    partial_end: bool


def song_boundaries(path: np.ndarray) -> np.ndarray:
    """Seconds t where the path goes from theme 6 into T1.1."""
    t1 = int(ENTRY_STATES[0])
    prev_theme = STATE_THEME[path[:-1]]
    return np.flatnonzero((path[1:] == t1) & (prev_theme == N_THEMES - 1)) + 1


def segment(path: np.ndarray, offset: int = 0) -> tuple[list[Session], list[Song]]:
    """``offset`` is added to every time so a half can report recording seconds."""
    active = ~IS_BG[path]
    edges = np.flatnonzero(np.diff(np.r_[0, active.astype(np.int8), 0]))
    runs = edges.reshape(-1, 2)
    bounds = set(song_boundaries(path).tolist())
    sessions: list[Session] = []
    songs: list[Song] = []
    t1 = int(ENTRY_STATES[0])
    n = len(path)
    for i, (a, b) in enumerate(runs):
        sessions.append(Session(i, a + offset, b + offset))
        cuts = [a] + sorted(t for t in bounds if a < t < b) + [b]
        for j in range(len(cuts) - 1):
            s, e = cuts[j], cuts[j + 1]
            last_theme = int(STATE_THEME[path[e - 1]])
            songs.append(
                Song(
                    session=i,
                    song=len(songs),
                    start_s=s + offset,
                    end_s=e + offset,
                    first_theme=int(STATE_THEME[path[s]]) + 1,
                    last_theme=last_theme + 1,
                    partial_start=j == 0 and path[s] != t1,
                    partial_end=last_theme != N_THEMES - 1 or e == n,
                )
            )
    return sessions, songs


def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
