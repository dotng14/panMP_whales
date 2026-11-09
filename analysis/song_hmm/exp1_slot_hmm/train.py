"""Fit the song-session HMM on the first half of the recording and decode both halves.

    python -m analysis.song_hmm.exp1_slot_hmm.train [--wav PATH] [--max-iter 30]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .decode import segment, write_csv
from .features import FEATURE_NAMES, load_or_build
from .hmm import baum_welch, initial_params, posterior, viterbi
from .labels import length_hist, load_labels
from .topology import IS_BG, N_THEMES, STATE_THEME, Params

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WAV_PATH = Path(r"C:\Users\hoang\Downloads\SOCAL_H_76_230426_193500_df100.x.x.wav")
CSV_PATH = REPO / "backend/data/song_structure/SOCAL_H_76_230426_193500_df100.x.x.motif_segments.csv"
SPLIT_S = 5 * 3600
WORK = HERE / "work"
MODEL_PATH = HERE / "song_session_hmm.npz"
LEVEL_COL = FEATURE_NAMES.index("level")


def decode_half(x: np.ndarray, p: Params, name: str, offset: int) -> None:
    path, _ = viterbi(x, p)
    post = posterior(x, p)
    theme_member = np.zeros((len(STATE_THEME), N_THEMES))
    has = STATE_THEME >= 0
    theme_member[np.flatnonzero(has), STATE_THEME[has]] = 1.0
    out = WORK / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "posteriors.npz",
        path=path,
        p_song=1.0 - post.gamma[:, IS_BG].sum(axis=1),
        p_theme=post.gamma @ theme_member,
        loglik=post.loglik,
        offset=offset,
    )
    sessions, songs = segment(path, offset)
    write_csv(sessions, out / "sessions.csv")
    write_csv(songs, out / "songs.csv")
    full = sum(not (s.partial_start or s.partial_end) for s in songs)
    print(f"{name}: {len(sessions)} sessions, {len(songs)} songs ({full} complete), loglik {post.loglik / len(x):.4f} / s")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", type=Path, default=WAV_PATH)
    ap.add_argument("--max-iter", type=int, default=30)
    args = ap.parse_args(argv)

    t0 = time.time()
    feats = load_or_build(args.wav, SPLIT_S, WORK / "features.npz")
    x = feats["x"]
    labels = load_labels(CSV_PATH, len(x))
    train, test = x[:SPLIT_S], x[SPLIT_S:]
    print(f"features {x.shape} from {args.wav.name} ({time.time() - t0:.0f} s); train {len(train)} s, test {len(test)} s")

    # Seed with train-half labels only.
    p0 = initial_params(train, labels.slot[:SPLIT_S], length_hist(labels, SPLIT_S), LEVEL_COL)
    p, history = baum_welch([train], p0, max_iter=args.max_iter)
    p.save(MODEL_PATH, history=np.array(history), feat_mean=feats["mean"], feat_std=feats["std"], split_s=SPLIT_S)
    print(f"saved {MODEL_PATH.relative_to(REPO)} after {len(history) - 1} iterations ({time.time() - t0:.0f} s)")

    decode_half(train, p, "train", 0)
    decode_half(test, p, "test", SPLIT_S)


if __name__ == "__main__":
    main()
