"""Run the best Experiment 3 setup (input S2_N1_X2_20s + M1 theme HMM) on the wav and cut out each detected song.

Used by notebooks/detect_songs_best_model.ipynb.

    model = load_model()                    # trains once if no saved model exists
    songs, decoded = detect_songs(model)    # every detected song, in recording seconds
    plot_song(songs[0], decoded)            # spectrogram of that song with its themes
    plot_span(14 * 60, 21 * 60, decoded)    # any stretch, e.g. minute 14 to minute 21
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from analysis.song_hmm.labels import recording_start_unix
from analysis.song_hmm.train import SPLIT_S, WAV_PATH

from .inputs.spectrogram import SR, dc_removed, normalise, spectrogram
from .inputs.window_features import standardise_pca, x2
from .models import m1_theme
from .models.common import HALVES, N_S, N_THEMES, WORK

INPUT_KEY = "S2_N1_X2_20s"
INPUT_PATH = WORK / "inputs" / f"{INPUT_KEY}.npz"
MODEL_PATH = WORK / "models" / "M1" / "model.pkl"
THEME_COLORS = plt.cm.tab10(np.arange(N_THEMES))
START_UTC = datetime.fromtimestamp(recording_start_unix(WAV_PATH.name), tz=timezone.utc)


@dataclass
class Song:
    number: int
    start_s: int  # seconds from the start of the recording
    end_s: int
    complete: bool  # starts at T1 after T6 (or at a session start in T1) and ends in T6
    theme_seconds: list[int]  # seconds of T1..T6 inside the song

    @property
    def minutes(self) -> tuple[float, float]:
        return self.start_s / 60, self.end_s / 60

    @property
    def duration_min(self) -> float:
        return (self.end_s - self.start_s) / 60

    def clock(self) -> tuple[str, str]:
        f = lambda s: (START_UTC + timedelta(seconds=int(s))).strftime("%H:%M:%S")  # noqa: E731
        return f(self.start_s), f(self.end_s)


def build_input() -> dict:
    """S2 spectrogram -> N1 floor -> X2 20 s windows -> PCA (train half). Cached in work/inputs/."""
    if INPUT_PATH.exists():
        return dict(np.load(INPUT_PATH))
    _, audio = wavfile.read(WAV_PATH, mmap=True)
    x = dc_removed(audio)
    halves = {h: normalise(spectrogram(x[a * SR : b * SR], "S2", b - a), "N1") for h, a, b in HALVES}
    fps = 40
    tr, te = x2(halves["train"], fps, 20), x2(halves["test"], fps, 20)
    red = standardise_pca(tr.raw, te.raw)
    centre = 256 / SR / 2
    out = {"train": red.train.astype(np.float32), "test": red.test.astype(np.float32),
           "t_train": tr.starts_s + 10 + centre, "t_test": SPLIT_S + te.starts_s + 10 + centre}
    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(INPUT_PATH, **out)
    return out


def load_model():
    """The fitted M1 HMM. Trains it once (about a minute) if no saved model exists."""
    if not MODEL_PATH.exists():
        m1_theme.run("M1", build_input(), verbose=False)
    with open(MODEL_PATH, "rb") as handle:
        return pickle.load(handle)


def detect_songs(model) -> tuple[list[Song], dict]:
    """Decode the whole recording (each half on its own) and list every song, in recording seconds.

    Returns (songs, decoded) where decoded has per-second ``theme`` (0-5, -1 background)
    and ``session`` arrays for the whole recording and the song ``boundaries``.
    """
    inp = build_input()
    n_pc = model["n_pc"]
    xs = {h: inp[h][:, :n_pc].astype(np.float64) for h in ("train", "test")}
    ts = {"train": inp["t_train"], "test": inp["t_test"]}
    decs, _ = m1_theme.decode_halves("M1", model["hmm"], xs, ts)
    theme = np.concatenate([decs["train"].theme, decs["test"].theme])
    session = np.concatenate([decs["train"].session, decs["test"].session])
    bounds = np.concatenate([decs["train"].boundaries, decs["test"].boundaries + SPLIT_S])
    songs = []
    for h, off, _ in HALVES:
        for s in decs[h].songs:
            songs.append(Song(len(songs) + 1, s.start_s + off, s.end_s + off, s.complete, list(s.theme_seconds)))
    return songs, {"theme": theme, "session": session, "boundaries": bounds}


def song_table(songs: list[Song]):
    import pandas as pd

    rows = []
    for s in songs:
        c0, c1 = s.clock()
        rows.append({"song": s.number, "start (min)": round(s.minutes[0], 2), "end (min)": round(s.minutes[1], 2),
                     "length (min)": round(s.duration_min, 2), "start (UTC)": c0, "end (UTC)": c1, "complete": s.complete,
                     **{f"T{k + 1} (s)": v for k, v in enumerate(s.theme_seconds)}})
    return pd.DataFrame(rows).set_index("song")


def plot_span(start_s: float, end_s: float, decoded: dict | None = None, title: str | None = None, save_to=None):
    """Spectrogram of [start_s, end_s) (recording seconds) with the decoded themes and song boundaries below it."""
    a, b = int(max(0, start_s)), int(min(N_S, end_s))
    _, audio = wavfile.read(WAV_PATH, mmap=True)
    x = np.asarray(audio[a * SR : b * SR], np.float32)
    x -= x.mean()
    sp = spectrogram(x, "S2", b - a)
    v = normalise(sp, "N3")  # display only: a low floor keeps quiet units visible
    step = max(1, len(v) // 4000)
    v = v[::step]
    hz = np.sqrt(sp.edges[:, 0] * sp.edges[:, 1])
    t0, t1 = a / 60, b / 60
    fig, axes = plt.subplots(2, 1, figsize=(14, 4.6), sharex=True, gridspec_kw={"height_ratios": [5, 0.6]})
    axes[0].imshow(v.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=np.percentile(v, 99.5), extent=(t0, t1, 0, v.shape[1]))
    ticks = [50, 100, 200, 400, 800]
    axes[0].set_yticks([np.interp(np.log(f), np.log(hz), np.arange(len(hz))) for f in ticks], [str(f) for f in ticks])
    axes[0].set_ylabel("Hz")
    c0 = (START_UTC + timedelta(seconds=a)).strftime("%H:%M:%S")
    c1 = (START_UTC + timedelta(seconds=b)).strftime("%H:%M:%S")
    axes[0].set_title(title or f"minute {t0:.1f} to {t1:.1f} ({c0}-{c1} UTC)", fontsize=10)
    if decoded is not None:
        th = decoded["theme"][a:b]
        tt = (a + np.arange(len(th)) + 0.5) / 60
        for k in range(N_THEMES):
            m = th == k
            axes[1].scatter(tt[m], np.zeros(m.sum()), marker="|", s=300, color=THEME_COLORS[k], linewidths=1.2)
        for bt in decoded["boundaries"]:
            if a <= bt < b:
                for ax in axes:
                    ax.axvline(bt / 60, color="cyan", lw=1.2)
        for k in range(N_THEMES):
            axes[1].plot([], [], color=THEME_COLORS[k], lw=4, label=f"T{k + 1}")
        axes[1].legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.9), frameon=False)
    axes[1].set_yticks([])
    axes[1].set_ylabel("theme", rotation=0, ha="right", va="center")
    axes[1].set_xlabel("minutes from the start of the recording (cyan: song boundaries)")
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=110)
    return fig


def plot_song(song: Song, decoded: dict, pad_s: float = 15, save_to=None):
    """One detected song, with ``pad_s`` seconds of context on each side."""
    kind = "complete song" if song.complete else "partial song"
    c0, c1 = song.clock()
    title = (f"Song {song.number} ({kind}): minute {song.minutes[0]:.1f} to {song.minutes[1]:.1f}, "
             f"{song.duration_min:.1f} min, {c0}-{c1} UTC")
    return plot_span(song.start_s - pad_s, song.end_s + pad_s, decoded, title=title, save_to=save_to)


def plot_overview(songs: list[Song], decoded: dict):
    """The whole 10 h: decoded themes, sessions and every detected song."""
    fig, ax = plt.subplots(figsize=(14, 2.6))
    th = decoded["theme"]
    t = (np.arange(len(th)) + 0.5) / 3600
    for k in range(N_THEMES):
        m = th == k
        ax.scatter(t[m][::5], np.ones(m.sum())[::5], marker="|", s=120, color=THEME_COLORS[k], linewidths=0.6)
    for i, s in enumerate(songs):
        shade = ("0.15", "0.45")[i % 2] if s.complete else ("0.72", "0.85")[i % 2]  # alternate so neighbours stay apart
        ax.plot([s.start_s / 3600, s.end_s / 3600], [0.55, 0.55], lw=7, color=shade, solid_capstyle="butt")
        if s.duration_min >= 4 or i % 3 == 0:
            ax.text((s.start_s + s.end_s) / 7200, 0.3, str(s.number), ha="center", fontsize=6)
    ax.axvline(SPLIT_S / 3600, color="tab:blue", lw=1)
    ax.set_yticks([1, 0.55], ["themes", "songs"])
    ax.set_ylim(0.1, 1.3)
    ax.set_xlabel("hours from the start of the recording (dark bars: complete songs, light: partial, alternating shades; blue: train/test split)")
    fig.tight_layout()
    return fig
