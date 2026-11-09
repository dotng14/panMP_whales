"""Phase 1 screening of the window inputs: 3 S x 4 N x {X1, X2} x {12 s, 20 s} x {PCA-20, PCA-40}, plus X5.

    python -m analysis.song_exp3.inputs.screen

For every combination, on the test half (same 500 queries as the pipeline check):
  repeat_hit   nearest neighbour (>= 5 min away) lies 1 or 2 song periods away (13 +- 2, 26 +- 2 min)
  far_share    nearest neighbour lies >= 7 min away (another song, given themes last < 7 min)
  csv_same     nearest neighbour has the same CSV theme (weak reference)
Writes work/inputs/<S>_<N>_<X>_<w>s.npz (train/test PCA-40 + window times) and work/inputs/screen.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from analysis.song_hmm.labels import load_labels
from analysis.song_hmm.train import CSV_PATH, SPLIT_S, WAV_PATH
from analysis.song_hmm_phrase.pipeline_check.check import patch_similarity, query_set, retrieval, unit

from .spectrogram import N_OPTIONS, S_OPTIONS, SR, dc_removed, normalise, spectrogram
from .window_features import standardise_pca, x1, x2

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "work" / "inputs"
N_S = 36000
WINDOWS = (12, 20)


def queries() -> tuple[np.ndarray, np.ndarray]:
    f = np.load(HERE.parent / "song_hmm_phrase" / "work" / "phrase_features.npz")
    t_q, t_c, _, _ = query_set(f["level_test"], (f["t_test"] - SPLIT_S) * 20)
    return t_q, t_c


def scores(z: np.ndarray, centres: np.ndarray, t_q, t_c, csv_theme) -> dict:
    qi = np.clip(np.searchsorted(centres, t_q), 0, len(centres) - 1)
    ci = np.clip(np.searchsorted(centres, t_c), 0, len(centres) - 1)
    u = unit(z)
    sim = u[qi] @ u[ci].T
    tq, tc = centres[qi], centres[ci]
    r = retrieval(sim, tq, tc)
    far = np.abs(tc[None, :] - tq[:, None]) >= 300
    nn = np.argmax(np.where(far, sim, -np.inf), 1)
    lag = np.abs(tc[nn] - tq)
    th_q = csv_theme[np.clip(tq.astype(int), 0, N_S - 1)]
    th_n = csv_theme[np.clip(tc[nn].astype(int), 0, N_S - 1)]
    th_c = csv_theme[np.clip(tc.astype(int), 0, N_S - 1)]
    return {
        "repeat_hit": r["hit_rate"],
        "repeat_chance": r["chance"],
        "repeat_lift": r["lift"],
        "far_share": float(np.mean(lag >= 420)),
        "csv_same": float(np.mean(th_q == th_n)),
        "csv_same_chance": float(np.mean(th_c[None, :] == th_q[:, None])),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "screen.json"
    rows = json.loads(out_json.read_text()) if out_json.exists() else {}
    t_q, t_c = queries()
    csv_theme = load_labels(CSV_PATH, N_S).theme
    _, audio = wavfile.read(WAV_PATH, mmap=True)
    x = dc_removed(audio)
    for s_code in S_OPTIONS:
        t0 = time.time()
        halves = {h: spectrogram(x[a * SR : b * SR], s_code, b - a) for h, a, b in (("train", 0, SPLIT_S), ("test", SPLIT_S, N_S))}
        for n_code in N_OPTIONS:
            key_sn = f"{s_code}_{n_code}"
            if all(f"{key_sn}_X{k}_{w}s" in rows for k in (1, 2) for w in WINDOWS) and f"{key_sn}_X5_20s" in rows:
                continue
            sw = {h: normalise(sp, n_code) for h, sp in halves.items()}
            fps = halves["train"].fps
            centre0 = halves["train"].centre_s
            for xname, fn in (("X1", x1), ("X2", x2)):
                for w in WINDOWS:
                    tr, te = fn(sw["train"], fps, w), fn(sw["test"], fps, w)
                    red = standardise_pca(tr.raw, te.raw)
                    centres = SPLIT_S + te.starts_s + w / 2 + centre0
                    row = {"S": s_code, "N": n_code, "X": xname, "window_s": w, "dims_raw": int(tr.raw.shape[1]), "n_bands": int(sw["train"].shape[1])}
                    for k in (20, 40):
                        row[f"pca{k}"] = scores(red.test[:, :k], centres, t_q, t_c, csv_theme) | {"explained": float(red.explained[k - 1])}
                    key = f"{key_sn}_{xname}_{w}s"
                    rows[key] = row
                    np.savez(OUT / f"{key}.npz", train=red.train.astype(np.float32), test=red.test.astype(np.float32),
                             t_train=tr.starts_s + w / 2 + centre0, t_test=centres)
            # X5: raw patch, shift-tolerant (20 s windows), compared directly
            step = int(2 * fps)
            w = 20 * fps
            st = np.arange(0, len(sw["test"]) - w + 1, step)
            centres = SPLIT_S + st / fps + 10 + centre0
            qi = np.clip(np.searchsorted(centres, t_q), 0, len(st) - 1)
            ci = np.clip(np.searchsorted(centres, t_c), 0, len(st) - 1)
            sim = patch_similarity(sw["test"], st[qi], st[ci], w, ds=max(1, int(0.4 * fps)))
            r = retrieval(sim, centres[qi], centres[ci])
            rows[f"{key_sn}_X5_20s"] = {"S": s_code, "N": n_code, "X": "X5", "window_s": 20, "patch": {"repeat_hit": r["hit_rate"], "repeat_chance": r["chance"], "repeat_lift": r["lift"]}}
            out_json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
            best = max(rows[f"{key_sn}_{xn}_{w}s"]["pca20"]["repeat_hit"] for xn in ("X1", "X2") for w in WINDOWS)
            print(f"{key_sn}: best PCA-20 repeat hit {best:.3f}, X5 {r['hit_rate']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        del halves


if __name__ == "__main__":
    main()
