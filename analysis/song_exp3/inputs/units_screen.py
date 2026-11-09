"""Phase 1 screening of X4 unit detection for every S x N setting.

    python -m analysis.song_exp3.inputs.units_screen

Per setting: units per minute in loud and quiet minutes, duration percentiles,
share of time inside a unit, number of unit types (BIC), and the song-repeat
scores of a bag-of-unit-types window vector (same queries as the window screening).
Writes work/inputs/units_<S>_<N>.npz, work/inputs/units_screen.json and overlay figures.
"""

from __future__ import annotations

import json
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from analysis.song_hmm.labels import load_labels  # noqa: E402
from analysis.song_hmm.train import CSV_PATH, SPLIT_S, WAV_PATH  # noqa: E402

from .screen import HERE, N_S, OUT, queries, scores  # noqa: E402
from .spectrogram import N_OPTIONS, S_OPTIONS, SR, dc_removed, normalise, spectrogram  # noqa: E402
from .units import detect, fit_types, token_windows  # noqa: E402

FIG = HERE / "figures"
PC = HERE.parent / "song_hmm_phrase" / "pipeline_check" / "work" / "results.json"


def overlay(x, s_code, n_code, stretches, path):
    fig, axes = plt.subplots(1, len(stretches), figsize=(6 * len(stretches), 3.6))
    for ax, (name, (t0, t1)) in zip(axes, stretches.items(), strict=True):
        sp = spectrogram(x[t0 * SR : t1 * SR], s_code, t1 - t0)
        v = normalise(sp, n_code)
        u = detect(v, sp.fps, sp.edges, t0=sp.centre_s)
        ax.imshow(v.T, aspect="auto", origin="lower", cmap="magma", extent=(0, t1 - t0, 0, v.shape[1]),
                  vmin=np.percentile(v, 5), vmax=np.percentile(v, 99.5))
        hz = np.sqrt(sp.edges[:, 0] * sp.edges[:, 1])
        def band(f):
            return float(np.interp(np.log(max(f, 1)), np.log(hz), np.arange(len(hz))))
        for r in u:
            lo_b = band(min(r["start_hz"], r["end_hz"], r["peak_hz"])) - 1
            hi_b = band(max(r["start_hz"], r["end_hz"], r["peak_hz"])) + 1
            ax.add_patch(plt.Rectangle((r["start_s"], lo_b), r["duration_s"], hi_b - lo_b, fill=False, ec="cyan", lw=0.6))
        ax.set_title(f"{s_code}·{n_code} | {name}: {len(u)} units", fontsize=9)
        ax.set_xlabel("s")
        ax.set_ylabel("band")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "units_screen.json"
    rows = json.loads(out_json.read_text()) if out_json.exists() else {}
    pc = json.loads(PC.read_text(encoding="utf-8"))
    loud, quiet = np.array(pc["B"]["loud_minutes"]), np.array(pc["B"]["quiet_minutes"])
    stretches = {k: tuple(v) for k, v in pc["stretches"].items()}
    t_q, t_c = queries()
    csv_theme = load_labels(CSV_PATH, N_S).theme
    _, audio = wavfile.read(WAV_PATH, mmap=True)
    x = dc_removed(audio)
    centres = SPLIT_S + 10 + np.arange(0, N_S - SPLIT_S - 20 + 1, 2.0)
    for s_code in S_OPTIONS:
        halves = None
        for n_code in N_OPTIONS:
            key = f"{s_code}_{n_code}"
            if key in rows:
                continue
            t0 = time.time()
            if halves is None:
                halves = {h: spectrogram(x[a * SR : b * SR], s_code, b - a) for h, a, b in (("train", 0, SPLIT_S), ("test", SPLIT_S, N_S))}
            u = {}
            for h, off in (("train", 0), ("test", SPLIT_S)):
                sp = halves[h]
                u[h] = detect(normalise(sp, n_code), sp.fps, sp.edges, t0=off + sp.centre_s)
            types = fit_types(u["train"])
            lab = {h: types.predict(u[h]) for h in u}
            allu = np.concatenate([u["train"], u["test"]])
            minute = (allu["start_s"] // 60).astype(int)
            per_min = np.bincount(minute, minlength=N_S // 60)[: N_S // 60]
            covered = np.zeros(N_S * 10, bool)
            for r in allu:
                covered[int(r["start_s"] * 10) : int(r["end_s"] * 10)] = True
            vec = token_windows(u["test"], lab["test"], types.k, centres)
            sc = scores(vec, centres, t_q, t_c, csv_theme)
            rows[key] = {
                "S": s_code, "N": n_code, "n_units": int(len(allu)), "k_types": int(types.k),
                "units_per_min_loud": float(per_min[loud].mean()), "units_per_min_quiet": float(per_min[quiet].mean()),
                "duration_pct": {str(q): float(np.percentile(allu["duration_s"], q)) for q in (10, 25, 50, 75, 90)},
                "time_covered": float(covered.mean()),
                "type_window": sc,
            }  # fmt: skip
            np.savez(OUT / f"units_{key}.npz", train=u["train"], test=u["test"], type_train=lab["train"], type_test=lab["test"])
            with open(OUT / f"unit_types_{key}.pkl", "wb") as h:
                pickle.dump(types, h)
            out_json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
            print(f"{key}: {len(allu)} units, k={types.k}, loud {rows[key]['units_per_min_loud']:.0f}/min, quiet {rows[key]['units_per_min_quiet']:.0f}/min, "
                  f"median {rows[key]['duration_pct']['50']:.2f} s, repeat hit {sc['repeat_hit']:.3f} ({time.time() - t0:.0f} s)", flush=True)
            if s_code == "S2":
                overlay(x, s_code, n_code, stretches, FIG / f"units_overlay_{key}.png")
        del halves


if __name__ == "__main__":
    main()
