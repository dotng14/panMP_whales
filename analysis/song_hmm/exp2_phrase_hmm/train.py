"""Fit the phrase-window theme HMM on the first half and decode both halves.

    python -m analysis.song_hmm_phrase.train [--seed audio|csv] [--wav PATH]

--seed audio  Experiment 2  (k-means seeding; CSV only names Theme 1)
--seed csv    Experiment 2b (theme Gaussians seeded from the CSV)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analysis.song_hmm.labels import load_labels
from analysis.song_hmm.train import CSV_PATH, SPLIT_S, WAV_PATH

from .decode import WINDOW_CENTRE_S, from_phrase_path, write_outputs
from .features import load_or_build
from .hmm import baum_welch, posterior, viterbi
from .init import audio_seed, csv_seed

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
RUNS = {"audio": "exp2", "csv": "exp2b"}


def window_csv_theme(theme_per_s: np.ndarray, t_start: np.ndarray) -> np.ndarray:
    return theme_per_s[np.clip(np.floor(t_start + WINDOW_CENTRE_S).astype(int), 0, len(theme_per_s) - 1)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", choices=sorted(RUNS), default="audio")
    ap.add_argument("--wav", type=Path, default=WAV_PATH)
    ap.add_argument("--max-iter", type=int, default=50)
    args = ap.parse_args(argv)
    run = RUNS[args.seed]
    t0 = time.time()

    f = load_or_build(args.wav, SPLIT_S, WORK / "phrase_features.npz")
    n_seconds = int(f["n_seconds"])
    labels = load_labels(CSV_PATH, n_seconds)
    x_tr, x_te = f["X_train"], f["X_test"]
    csv_w = window_csv_theme(labels.theme, f["t_train"])
    print(f"[{run}] features: train {x_tr.shape}, test {x_te.shape}, PCA keeps {f['explained'].sum():.2f} of variance")

    seed_fn = audio_seed if args.seed == "audio" else csv_seed
    p0, info = seed_fn(x_tr, f["level_train"], csv_w)
    if info:
        print(f"[{run}] cluster cycle {info['cluster_order']}, share of run transitions on the cycle {info['cycle_share']:.2f}, Theme 1 rotation {info['rotation']}")
    p, history = baum_welch([x_tr], p0, max_iter=args.max_iter)

    out = WORK / run
    out.mkdir(parents=True, exist_ok=True)
    p.save(HERE / f"{run}_model.npz", history=np.array(history))
    (out / "init.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    for half, x, t, offset, n in (
        ("train", x_tr, f["t_train"], 0, SPLIT_S),
        ("test", x_te, f["t_test"], SPLIT_S, n_seconds - SPLIT_S),
    ):
        path, _ = viterbi(x, p)
        post = posterior(x, p)
        dec = from_phrase_path(run, path, t, offset, n)
        write_outputs(dec, out / half, offset)
        np.savez(out / half / "posterior.npz", path=path, gamma=post.gamma.astype(np.float32), loglik=post.loglik)
        full = sum(s.complete for s in dec.songs)
        print(f"[{run}] {half}: {len(dec.sessions)} sessions, {len(dec.songs)} songs ({full} complete), loglik {post.loglik / len(x):.3f} / window")
    print(f"[{run}] done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
