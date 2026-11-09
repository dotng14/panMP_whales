"""Phase 2: pick the inputs from the screening and train the four models.

    python -m analysis.song_hmm.exp3.run_models [--only M1,M2,M3,M4]

Input choice (written to work/models/inputs_used.json):
  M1, M2  the window input (X1 or X2, 12 or 20 s, any S x N) with the best PCA-20 song-repeat hit
  M4      X3 frame vectors from that input's S x N
  M3      X4 units from the S x N whose unit-type windows score best
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time

import numpy as np
from scipy.io import wavfile

from analysis.song_hmm.exp1_slot_hmm.train import SPLIT_S, WAV_PATH

from .inputs.frames import frames
from .inputs.spectrogram import SR, dc_removed, normalise, spectrogram
from .models import m1_theme, m2_songfirst, m3_units, m4_theme_hmms
from .models.common import N_S, WORK
from .models.graph import HMM, Categorical, DiagGMM

IN = WORK / "inputs"
OUT = WORK / "models"


def choose_inputs() -> dict:
    win = json.loads((IN / "screen.json").read_text())
    cands = {k: v for k, v in win.items() if "pca20" in v}
    best_w = max(cands, key=lambda k: cands[k]["pca20"]["repeat_hit"])
    units = json.loads((IN / "units_screen.json").read_text()) if (IN / "units_screen.json").exists() else {}
    best_u = max(units, key=lambda k: units[k]["type_window"]["repeat_hit"]) if units else None
    sn = f"{cands[best_w]['S']}_{cands[best_w]['N']}"
    return {"window": best_w, "window_scores": cands[best_w], "frames_from": sn, "units": best_u,
            "units_scores": units.get(best_u) if best_u else None}


def synthetic(hmm: HMM, n: int, seed: int = 3, group_of_state=None) -> dict:
    """Sample from a fitted model, refit from the true state groups, decode, score against the truth."""
    rng = np.random.default_rng(seed)
    x, path = hmm.sample(n, rng)
    g2 = copy.deepcopy(hmm.graph)
    grp = g2.group[path]
    if isinstance(hmm.emis, Categorical):
        emis = Categorical.from_assignment(x, grp, hmm.emis.logp.shape[0], hmm.emis.logp.shape[1])
    else:
        emis = DiagGMM.from_assignment(x, grp, hmm.emis.w.shape[0])
    h2 = HMM(g2, emis)
    h2.fit([x], max_iter=15)
    est = h2.viterbi(x)
    th_true, th_est = group_of_state[path], group_of_state[est]
    sung = th_true >= 0
    return {"n_steps": n, "theme_accuracy": float(np.mean(th_true[sung] == th_est[sung])), "song_vs_background": float(np.mean((th_true >= 0) == (th_est >= 0))),
            "true_boundaries": int(np.sum((th_true[1:] == 0) & (th_true[:-1] == 5))), "found_boundaries": int(np.sum((th_est[1:] == 0) & (th_est[:-1] == 5)))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="M1,M2,M3,M4")
    args = ap.parse_args(argv)
    only = set(args.only.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    choice = choose_inputs()
    (OUT / "inputs_used.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    print(f"window input {choice['window']}; frames from {choice['frames_from']}; units {choice['units']}", flush=True)
    res_path = OUT / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    win = dict(np.load(IN / f"{choice['window']}.npz"))

    def done(name, r, extra=None):
        out = {"loglik_per_step": r.get("loglik_per_step"), "history": r.get("history")}
        out.update({k: v for k, v in r.items() if k not in ("decoded", "hmm", "history", "loglik_per_step")})
        if extra:
            out.update(extra)
        results[name] = out
        res_path.write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

    if "M1" in only:
        t0 = time.time()
        print("M1 ...", flush=True)
        r = m1_theme.run("M1", win)
        done("M1", r, {"synthetic": synthetic(r["hmm"], 9000, group_of_state=m1_theme.build_graph()[1]), "seconds": time.time() - t0})
    if "M2" in only:
        t0 = time.time()
        print("M2 ...", flush=True)
        done("M2", m2_songfirst.run("M2", win), {"seconds": time.time() - t0})
    if "M3" in only and choice["units"]:
        t0 = time.time()
        print("M3 ...", flush=True)
        d = np.load(IN / f"units_{choice['units']}.npz")
        with open(IN / f"unit_types_{choice['units']}.pkl", "rb") as h:
            types = pickle.load(h)
        r = m3_units.run("M3", {"train": d["train"], "test": d["test"]}, {"train": d["type_train"], "test": d["type_test"]}, types.k)
        done("M3", r, {"synthetic": synthetic(r["hmm"], 20000, group_of_state=m3_units.STATE_THEME), "k_types": types.k, "seconds": time.time() - t0})
    if "M4" in only:
        t0 = time.time()
        print("M4 ...", flush=True)
        s_code, n_code = choice["frames_from"].split("_")
        _, audio = wavfile.read(WAV_PATH, mmap=True)
        x = dc_removed(audio)
        sw = {h: normalise(spectrogram(x[a * SR : b * SR], s_code, b - a), n_code) for h, a, b in (("train", 0, SPLIT_S), ("test", SPLIT_S, N_S))}
        fps = 2000 // (100 if s_code == "S1" else 50)
        fr = frames(sw["train"], sw["test"], fps)
        del sw, x
        r = m4_theme_hmms.run("M4", fr)
        done("M4", r, {"synthetic": synthetic(r["hmm"], 20000, group_of_state=m4_theme_hmms.STATE_THEME), "frames_explained": fr["explained"], "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
