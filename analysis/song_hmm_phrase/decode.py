"""Decoded output in one common form, for Experiment 2, Experiment 1 and the CSV.

Everything is per second of one half (local seconds 0..n-1):
  theme[s]      0..5, or -1 for background
  session[s]    True inside a song session (breaths included)
  boundaries    seconds where a new song starts at a T6 -> T1 transition
Sessions and songs are derived from those three.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .topology import ENTRY, IS_BG, N_THEMES, STATE_THEME, STEP_S

WINDOW_CENTRE_S = 6.0


@dataclass
class Song:
    start_s: int
    end_s: int  # exclusive
    complete: bool  # starts at T1 after T6 (or session start in T1) and ends in T6 at a boundary or session end
    theme_seconds: list[int] = field(default_factory=list)  # seconds spent in each theme


@dataclass
class Decoded:
    name: str
    theme: np.ndarray
    session: np.ndarray
    boundaries: np.ndarray
    sessions: list[tuple[int, int]] = field(default_factory=list)
    songs: list[Song] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.theme)
        edges = np.flatnonzero(np.diff(np.r_[0, self.session.astype(np.int8), 0])).reshape(-1, 2)
        self.sessions = [(int(a), int(b)) for a, b in edges]
        bset = np.sort(self.boundaries)
        for a, b in self.sessions:
            cuts = [a] + [int(t) for t in bset if a < t < b] + [b]
            for j in range(len(cuts) - 1):
                s, e = cuts[j], cuts[j + 1]
                th = self.theme[s:e]
                starts_ok = (j > 0 or th[0] == 0) and s > 0
                ends_ok = (j < len(cuts) - 2 or th[th >= 0][-1:].tolist() == [N_THEMES - 1]) and e < n
                self.songs.append(Song(s, e, bool(starts_ok and ends_ok), np.bincount(th[th >= 0], minlength=N_THEMES).tolist()))


def window_to_seconds(values: np.ndarray, t_start: np.ndarray, offset: int, n_seconds: int) -> np.ndarray:
    """Give each local second the value of the window whose centre is nearest."""
    centres = t_start - offset + WINDOW_CENTRE_S
    sec = np.arange(n_seconds) + 0.5
    idx = np.clip(np.rint((sec - centres[0]) / STEP_S).astype(int), 0, len(values) - 1)
    return values[idx]


def from_phrase_path(name: str, path: np.ndarray, t_start: np.ndarray, offset: int, n_seconds: int) -> Decoded:
    theme_w = STATE_THEME[path]
    into_t1 = (path[1:] == ENTRY[0]) & (STATE_THEME[path[:-1]] == N_THEMES - 1)
    bound = np.rint(t_start[1:][into_t1] - offset + WINDOW_CENTRE_S).astype(int)
    return Decoded(
        name,
        window_to_seconds(theme_w, t_start, offset, n_seconds),
        ~window_to_seconds(IS_BG[path], t_start, offset, n_seconds),
        bound,
    )


def from_slot_path(name: str, path: np.ndarray) -> Decoded:
    """Experiment 1: per-second state path of the 176-state slot model."""
    from analysis.song_hmm.decode import song_boundaries
    from analysis.song_hmm.topology import IS_BG as IS_BG1
    from analysis.song_hmm.topology import STATE_THEME as THEME1

    return Decoded(name, THEME1[path].astype(np.int32), ~IS_BG1[path], song_boundaries(path))


def from_csv(name: str, theme: np.ndarray, song_starts: np.ndarray, offset: int) -> Decoded:
    n = len(theme)
    starts = np.rint(song_starts - offset).astype(int)
    return Decoded(name, theme.astype(np.int32), theme >= 0, starts[(starts > 0) & (starts < n)])


def write_outputs(dec: Decoded, folder: Path, offset: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "sessions.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["session", "start_s", "end_s"])
        for i, (a, b) in enumerate(dec.sessions):
            w.writerow([i, a + offset, b + offset])
    with (folder / "songs.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["song", "start_s", "end_s", "complete"] + [f"T{k + 1}_s" for k in range(N_THEMES)])
        for i, s in enumerate(dec.songs):
            w.writerow([i, s.start_s + offset, s.end_s + offset, s.complete, *s.theme_seconds])
    np.savez(folder / "decoded.npz", theme=dec.theme, session=dec.session, boundaries=dec.boundaries + offset, offset=offset)
