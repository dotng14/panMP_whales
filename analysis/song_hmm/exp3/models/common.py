"""Shared pieces for the Experiment 3 models: data paths, loudness, CSV labels at any time, output form."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from analysis.song_hmm.exp1_slot_hmm.labels import load_labels
from analysis.song_hmm.exp1_slot_hmm.train import CSV_PATH, SPLIT_S, WAV_PATH
from analysis.song_hmm.exp2_phrase_hmm.decode import Decoded, write_outputs

HERE = Path(__file__).resolve().parents[1]
WORK = HERE / "work"
N_S = 36000
N_THEMES = 6
HALVES = (("train", 0, SPLIT_S), ("test", SPLIT_S, N_S))
SESSION_GAP_S = 60  # Payne & McVay: no pause longer than 1 minute inside a session


def labels():
    return load_labels(CSV_PATH, N_S)


def level_1s() -> np.ndarray:
    """Broadband 30-1000 Hz level (dB) per second of the recording, DC removed. Cached."""
    path = WORK / "level_1s.npy"
    if path.exists():
        return np.load(path)
    sr, a = wavfile.read(WAV_PATH, mmap=True)
    sos = signal.butter(4, [30, 990], btype="band", fs=sr, output="sos")
    out = np.empty(N_S)
    zi = None
    for s0 in range(0, N_S, 600):
        x = np.asarray(a[s0 * sr : (s0 + 600) * sr], np.float64)
        x -= x.mean()
        if zi is None:
            zi = signal.sosfilt_zi(sos) * x[0]
        y, zi = signal.sosfilt(sos, x, zi=zi)
        out[s0 : s0 + 600] = 10 * np.log10(np.mean(y.reshape(-1, sr) ** 2, axis=1) + 1e-9)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def at_times(per_second: np.ndarray, t: np.ndarray) -> np.ndarray:
    return per_second[np.clip(np.floor(t).astype(int), 0, len(per_second) - 1)]


def steps_to_seconds(values: np.ndarray, t_steps: np.ndarray, offset: int, n: int) -> np.ndarray:
    """Each second of a half takes the value of the step whose time is nearest (times in recording seconds)."""
    sec = offset + np.arange(n) + 0.5
    idx = np.clip(np.searchsorted(t_steps, sec), 1, len(t_steps) - 1)
    left = np.abs(sec - t_steps[idx - 1]) <= np.abs(t_steps[idx] - sec)
    return values[np.where(left, idx - 1, idx)]


def enforce_session_rule(session: np.ndarray, theme: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Background shorter than or equal to 60 s inside song is a pause, not a session end."""
    session = session.copy()
    theme = theme.copy()
    e = np.flatnonzero(np.diff(np.r_[1, session.astype(np.int8), 1]))
    for a, b in e.reshape(-1, 2):
        if 0 < a and b < len(session) and b - a <= SESSION_GAP_S:
            session[a:b] = True
            theme[a:b] = theme[a - 1]
    return session, theme


def save(model: str, decs: dict[str, Decoded]) -> None:
    for h, off, _ in HALVES:
        write_outputs(decs[h], WORK / "models" / model / h, off)


def boundaries_from_theme_steps(theme_steps: np.ndarray, t_steps: np.ndarray, offset: int) -> np.ndarray:
    """Seconds (local to the half) where the step sequence goes from theme 6 to theme 1."""
    ok = (theme_steps[1:] == 0) & (theme_steps[:-1] == N_THEMES - 1)
    return np.rint(t_steps[1:][ok] - offset).astype(int)
