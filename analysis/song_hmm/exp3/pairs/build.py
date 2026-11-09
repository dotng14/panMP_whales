"""Build the blind phrase-pair set for the judging page (Phase 0, R-pairs).

    python -m analysis.song_hmm.exp3.pairs.build

20 query windows from the test half (loud, spread in time). For each query, the
nearest window at least 5 min away under each input below, plus one random window:
  best_window  the screening winner (PCA-20)
  best_X1      the best window-statistics input (PCA-20)
  exp2_input   Experiment 2's input: S1 N1 X1 12 s, PCA-20
  patch        X5, the raw spectrogram patch of the winner's S x N (shift-tolerant)
  units        X4 bag of unit types, best unit-detection setting
  random       a random window at least 5 min away (chance level)
Candidates that two inputs share are one pair. Order is shuffled; the page never
shows which input proposed a pair. Writes, under page/: pairs.json, img/, audio/;
and the key (which input proposed which pair) to work/pairs/key.json.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from analysis.song_hmm.exp1_slot_hmm.train import SPLIT_S, WAV_PATH  # noqa: E402
from analysis.song_hmm.exp2_phrase_hmm.pipeline_check.check import patch_similarity, unit  # noqa: E402

from ..inputs.screen import queries  # noqa: E402
from ..inputs.spectrogram import SR, normalise, spectrogram  # noqa: E402
from ..inputs.units import token_windows  # noqa: E402
from ..models.common import N_S, WORK  # noqa: E402

HERE = Path(__file__).resolve().parent
PAGE = HERE / "page"
IN = WORK / "inputs"
N_QUERIES = 20
MIN_GAP_S = 300
CLIP_S = 12


def nn(sim_row: np.ndarray, tq: float, tc: np.ndarray) -> int:
    s = np.where(np.abs(tc - tq) >= MIN_GAP_S, sim_row, -np.inf)
    return int(np.argmax(s))


def main() -> None:
    win = json.loads((IN / "screen.json").read_text())
    units = json.loads((IN / "units_screen.json").read_text())
    cand = {k: v for k, v in win.items() if "pca20" in v}
    best_w = max(cand, key=lambda k: cand[k]["pca20"]["repeat_hit"])
    best_x1 = max((k for k in cand if cand[k]["X"] == "X1"), key=lambda k: cand[k]["pca20"]["repeat_hit"])
    best_u = max(units, key=lambda k: units[k]["type_window"]["repeat_hit"])
    t_q_all, t_c = queries()
    rng = np.random.default_rng(42)
    pick = np.linspace(0, len(t_q_all) - 1, N_QUERIES).astype(int)
    t_q = t_q_all[pick]

    reps = {}
    for name, key in (("best_window", best_w), ("best_X1", best_x1), ("exp2_input", "S1_N1_X1_12s")):
        d = np.load(IN / f"{key}.npz")
        cen = d["t_test"]
        u = unit(d["test"][:, :20].astype(np.float64))
        qi = np.clip(np.searchsorted(cen, t_q), 0, len(cen) - 1)
        ci = np.clip(np.searchsorted(cen, t_c), 0, len(cen) - 1)
        reps[name] = u[qi] @ u[ci].T
    # units
    ud = np.load(IN / f"units_{best_u}.npz")
    with open(IN / f"unit_types_{best_u}.pkl", "rb") as h:
        types = pickle.load(h)
    vq = token_windows(ud["test"], ud["type_test"], types.k, t_q)
    vc = token_windows(ud["test"], ud["type_test"], types.k, t_c)
    reps["units"] = unit(np.vstack([vq, vc]))[: len(t_q)] @ unit(np.vstack([vq, vc]))[len(t_q) :].T
    # raw patch of the winner's S x N
    s_code, n_code = cand[best_w]["S"], cand[best_w]["N"]
    _, audio = wavfile.read(WAV_PATH, mmap=True)
    x_te = np.asarray(audio[SPLIT_S * SR : N_S * SR], np.float32)
    x_te -= x_te.mean()
    sp = spectrogram(x_te, s_code, N_S - SPLIT_S)
    sw = normalise(sp, n_code)
    fps = sp.fps
    w = 20 * fps
    st = lambda t: np.clip(((t - SPLIT_S - 10) * fps).astype(int), 0, len(sw) - w)  # noqa: E731
    reps["patch"] = patch_similarity(sw, st(t_q), st(t_c), w, ds=max(1, int(0.4 * fps)))

    pairs: list[dict] = []
    for i, tq in enumerate(t_q):
        found: dict[int, list[str]] = {}
        for name, sim in reps.items():
            j = nn(sim[i], tq, t_c)
            near = [k for k in found if abs(t_c[k] - t_c[j]) <= 4]
            found.setdefault(near[0] if near else j, []).append(name)
        far = np.flatnonzero(np.abs(t_c - tq) >= MIN_GAP_S)
        j = int(rng.choice(far))
        near = [k for k in found if abs(t_c[k] - t_c[j]) <= 4]
        found.setdefault(near[0] if near else j, []).append("random")
        for j, src in found.items():
            pairs.append({"query": float(tq), "candidate": float(t_c[j]), "sources": src})
    order = rng.permutation(len(pairs))
    pairs = [pairs[k] | {"id": f"p{n + 1:03d}"} for n, k in enumerate(order)]

    (PAGE / "img").mkdir(parents=True, exist_ok=True)
    (PAGE / "audio").mkdir(parents=True, exist_ok=True)
    disp = spectrogram(x_te, "S2", N_S - SPLIT_S)
    dv = normalise(disp, "N3")
    vmax = np.percentile(dv[:: 50], 99.7)
    hz = np.sqrt(disp.edges[:, 0] * disp.edges[:, 1])
    clips_done: set[int] = set()

    def clip(t: float) -> str:
        k = int(round(t))
        name = f"audio/w{k}.wav"
        if k not in clips_done:
            a = x_te[int((t - SPLIT_S - CLIP_S / 2) * SR) : int((t - SPLIT_S + CLIP_S / 2) * SR)].astype(np.float64)
            y = signal.resample_poly(a, 4, 1)
            y = y / max(np.max(np.abs(y)), 1e-9) * 0.9
            wavfile.write(PAGE / name, SR * 4, (y * 32767).astype(np.int16))
            clips_done.add(k)
        return name

    def panel(ax, t, label):
        a = int((t - SPLIT_S - 10) * disp.fps)
        blk = dv[a : a + 20 * disp.fps]
        ax.imshow(blk.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax, extent=(-10, 10, 0, blk.shape[1]))
        ticks = [100, 200, 400, 800]
        ax.set_yticks([np.interp(np.log(f), np.log(hz), np.arange(len(hz))) for f in ticks], [f"{f}" for f in ticks], fontsize=7)
        ax.set_ylabel("Hz", fontsize=7)
        ax.text(0.005, 0.96, label, transform=ax.transAxes, fontsize=9, color="white", va="top", fontweight="bold")
        ax.tick_params(labelsize=7)

    out_pairs = []
    for p in pairs:
        fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
        panel(axes[0], p["query"], "A")
        panel(axes[1], p["candidate"], "B")
        axes[1].set_xlabel("seconds around the window centre", fontsize=7)
        fig.tight_layout(pad=0.4)
        img = f"img/{p['id']}.jpg"
        fig.savefig(PAGE / img, dpi=90, pil_kwargs={"quality": 82})
        plt.close(fig)
        out_pairs.append({"id": p["id"], "img": img, "audioA": clip(p["query"]), "audioB": clip(p["candidate"])})
    (PAGE / "pairs.json").write_text(json.dumps({"pairs": out_pairs, "clip_seconds": CLIP_S}, indent=0), encoding="utf-8")
    key_dir = WORK / "pairs"
    key_dir.mkdir(parents=True, exist_ok=True)
    (key_dir / "key.json").write_text(json.dumps({"inputs": {"best_window": best_w, "best_X1": best_x1, "exp2_input": "S1_N1_X1_12s",
                                                              "patch": f"{s_code}_{n_code}_X5_20s", "units": best_u},
                                                   "pairs": pairs}, indent=1), encoding="utf-8")
    print(f"{len(out_pairs)} pairs from {N_QUERIES} queries; {len(clips_done)} audio clips; inputs {best_w}, {best_x1}, units {best_u}")


if __name__ == "__main__":
    main()
