"""Report for Experiment 2, with Experiment 1 reviewed and compared.

    python -m analysis.song_hmm.exp2_phrase_hmm.report

Needs ``python -m analysis.song_hmm.exp2_phrase_hmm.train --seed audio`` and ``--seed csv``.
Experiment 1's outputs are regenerated if missing. Writes reports/exp2_phrase_hmm.md,
figures/ and work/report.json. The three required sections are checked
before the file is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from analysis.song_hmm.report_paths import write_report  # noqa: E402
from analysis.song_hmm.exp1_slot_hmm import report as exp1_report  # noqa: E402
from analysis.song_hmm.exp1_slot_hmm import train as exp1_train  # noqa: E402
from analysis.song_hmm.exp1_slot_hmm.labels import load_labels  # noqa: E402
from analysis.song_hmm.exp1_slot_hmm.report import diag_gauss_ll, match_events, two_state_hmm  # noqa: E402
from analysis.song_hmm.exp1_slot_hmm.train import CSV_PATH, SPLIT_S, WAV_PATH  # noqa: E402

from . import report_text  # noqa: E402
from .decode import Decoded, from_csv, from_phrase_path, from_slot_path  # noqa: E402
from .features import STATS, band_spectrogram, load_or_build, whiten  # noqa: E402
from .hmm import baum_welch, params_from_assignment, sample, viterbi  # noqa: E402
from .metrics import csv_agreement, label_free, novelty_curve, novelty_peaks, unit_rows, window_values  # noqa: E402
from .topology import IS_BG, N_THEMES, STATE_THEME, Params  # noqa: E402
from .train import HERE, WORK  # noqa: E402

FIG = HERE / "figures"
EXP1 = Path(exp1_train.__file__).resolve().parent
HALVES = ("train", "test")
MODELS = ("exp1", "exp2", "exp2b")
COLORS = plt.cm.tab10(np.arange(N_THEMES))
ZOOM = (SPLIT_S + 2 * 3600 - 600, SPLIT_S + 2 * 3600 + 1800)  # 40 min of the test half (7.0-7.5 h plus lead-in)
REQUIRED = (
    "## Experiment 1 (previous): per-second slot HMM",
    "## Experiment 2 (current): phrase-window theme HMM",
    "## Comparing Experiment 1 and Experiment 2",
)


def ensure_exp1() -> dict:
    if not (exp1_train.WORK / "test" / "posteriors.npz").exists():
        exp1_train.main([])
    if not (exp1_train.WORK / "report.json").exists():
        exp1_report.main()
    return json.loads((exp1_train.WORK / "report.json").read_text(encoding="utf-8"))


def load_decoded(f: dict, labels) -> dict[str, dict[str, Decoded]]:
    n_total = int(f["n_seconds"])
    out: dict[str, dict[str, Decoded]] = {}
    for half, offset, n in (("train", 0, SPLIT_S), ("test", SPLIT_S, n_total - SPLIT_S)):
        d = {}
        d["exp1"] = from_slot_path("exp1", np.load(exp1_train.WORK / half / "posteriors.npz")["path"])
        for run in ("exp2", "exp2b"):
            path = np.load(WORK / run / half / "posterior.npz")["path"]
            d[run] = from_phrase_path(run, path, f[f"t_{half}"], offset, n)
        d["csv"] = from_csv("csv", labels.theme[offset : offset + n], labels.song_bounds[:, 0], offset)
        out[half] = d
    return out


def fit_metrics(f: dict) -> dict:
    res = {}
    for run in ("exp2", "exp2b"):
        with np.load(HERE / f"{run}_model.npz") as m:
            hist = m["history"].tolist()
        res[run] = {
            "history": hist,
            "iterations": len(hist) - 1,
            **{h: float(np.load(WORK / run / h / "posterior.npz")["loglik"]) / len(f[f"X_{h}"]) for h in HALVES},
        }
    xtr, xte = f["X_train"], f["X_test"]
    mean, var = xtr.mean(0), xtr.var(0)
    two_tr, two_te = two_state_hmm(xtr, xte, 0)
    res["baselines"] = {
        "one_gaussian": {"train": float(diag_gauss_ll(xtr, mean, var).mean()), "test": float(diag_gauss_ll(xte, mean, var).mean())},
        "two_state": {"train": two_tr / len(xtr), "test": two_te / len(xte)},
    }
    return res


def synthetic(p: Params, n: int, seed: int = 11) -> dict:
    x, true = sample(p, n, np.random.default_rng(seed))
    p0 = params_from_assignment(x, np.maximum(STATE_THEME[true], 0), IS_BG[true])
    fit, _ = baum_welch([x], p0, verbose=False)
    path, _ = viterbi(x, fit)
    t_true, t_dec = STATE_THEME[true], STATE_THEME[path]
    sung = t_true >= 0

    def edges(pth):
        on = ~IS_BG[pth]
        return np.flatnonzero(np.diff(on.astype(int)) != 0) + 1

    def bounds(pth):
        th = STATE_THEME[pth]
        return np.flatnonzero((th[1:] == 0) & (th[:-1] == N_THEMES - 1)) + 1

    return {
        "n_steps": n,
        "true_sessions": int((np.diff(np.r_[0, (~IS_BG[true]).astype(int)]) == 1).sum()),
        "true_song_boundaries": int(len(bounds(true))),
        "song_vs_background_accuracy": float(np.mean(IS_BG[true] == IS_BG[path])),
        "theme_accuracy": float(np.mean(t_dec[sung] == t_true[sung])),
        "session_edges_10s": match_events(edges(path), edges(true), 5),
        "song_boundaries_10s": match_events(bounds(path), bounds(true), 5),
    }


def pca_readout(f: dict, n_comp: int = 3, top: int = 5) -> list[dict]:
    edges = f["band_edges"]
    nb = len(edges)
    out = []
    for c in range(n_comp):
        load = f["basis"][:, c]
        order = np.argsort(-np.abs(load))[:top]
        out.append(
            {
                "component": c + 1,
                "explained": float(f["explained"][c]),
                "top": [
                    {"stat": STATS[i // nb], "band_hz": f"{edges[i % nb, 0]:.0f}-{edges[i % nb, 1]:.0f}", "weight": float(load[i])}
                    for i in order
                ],
                "stat_share": {s: float(np.sum(load[k * nb : (k + 1) * nb] ** 2)) for k, s in enumerate(STATS)},
            }
        )
    return out


# ---------------------------------------------------------------- figures


def _track(ax, t_h, theme, y, label):
    for k in range(N_THEMES):
        m = theme == k
        ax.scatter(t_h[m], np.full(m.sum(), y), s=3, marker="|", color=COLORS[k], linewidths=0.8)
    ax.text(-0.005, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)


def fig_ssm(f, dec_test):
    lo, hi = ZOOM
    t = f["t_test"] + 6.0
    sel = (t >= lo) & (t < hi)
    zn = unit_rows(f["z_test"])[sel]
    k = zn @ zn.T
    ext = (lo / 3600, hi / 3600, hi / 3600, lo / 3600)
    names = [("csv", "CSV themes"), ("exp1", "Experiment 1"), ("exp2", "Experiment 2")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.3))
    for ax, (key, title) in zip(axes, names, strict=True):
        ax.imshow(k, cmap="magma", vmin=np.percentile(k, 5), vmax=np.percentile(k, 99), extent=ext)
        wt = window_values(dec_test[key].theme, f["t_test"], SPLIT_S)[sel]
        ch = np.flatnonzero(wt[1:] != wt[:-1]) + 1
        for c in ch:
            ax.axvline(t[sel][c] / 3600, color="cyan", lw=0.7)
            ax.axhline(t[sel][c] / 3600, color="cyan", lw=0.7)
        for i in range(len(ch) + 1):
            a = 0 if i == 0 else ch[i - 1]
            b = len(wt) if i == len(ch) else ch[i]
            if b - a > 15 and wt[a] >= 0:
                ax.text((t[sel][a] + t[sel][b - 1]) / 7200, lo / 3600 - 0.004, f"T{wt[a] + 1}", fontsize=7, ha="center", va="bottom")
        ax.set_title(f"{title}: theme changes (cyan)", fontsize=10)
        ax.set_xlabel("hours")
    fig.suptitle("Phrase-window self-similarity, 40 min of the test half. A theme is a bright block; good labels put lines at block edges.", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "compare_ssm.png", dpi=110)
    plt.close(fig)


def fig_timeline_zoom(f, dec_test, curve_test):
    lo, hi = ZOOM[0] + 600, ZOOM[1]
    sr, audio = wavfile.read(WAV_PATH, mmap=True)
    seg = np.asarray(audio[lo * sr : hi * sr + 512])
    s = whiten(band_spectrogram(seg, sr, (hi - lo) * 20), 20)
    tsec = np.arange(lo, hi)
    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.3, 1.2, 1]})
    axes[0].imshow(s[::2].T, aspect="auto", origin="lower", cmap="magma", vmax=np.percentile(s, 99.5), extent=(lo / 3600, hi / 3600, 0, s.shape[1]))
    axes[0].set_ylabel("band (30 Hz → 1 kHz)")
    axes[0].set_title("Whitened spectrogram (0.05 s frames) and decoded themes, 30 min of the test half")
    for y, key, label in ((3, "csv", "CSV"), (2, "exp1", "Exp 1"), (1, "exp2", "Exp 2"), (0, "exp2b", "Exp 2b")):
        _track(axes[1], tsec / 3600, dec_test[key].theme[lo - SPLIT_S : hi - SPLIT_S], y, label)
        for b in dec_test[key].boundaries + SPLIT_S:
            if lo <= b < hi:
                axes[1].plot([b / 3600] * 2, [y - 0.4, y + 0.4], color="k", lw=1.2)
    axes[1].set_ylim(-0.7, 3.7)
    axes[1].set_yticks([])
    for k in range(N_THEMES):
        axes[1].plot([], [], color=COLORS[k], lw=3, label=f"T{k + 1}")
    axes[1].legend(ncol=6, fontsize=7, loc="upper right", bbox_to_anchor=(1, 1.3), frameon=False)
    tw = f["t_test"] + 6.0
    m = (tw >= lo) & (tw < hi)
    axes[2].plot(tw[m] / 3600, curve_test[m], color="k", lw=0.7)
    axes[2].set_ylabel("audio novelty")
    axes[2].set_xlabel("hours (black ticks on the theme rows: song boundaries)")
    fig.tight_layout()
    fig.savefig(FIG / "compare_timeline_zoom.png", dpi=120)
    plt.close(fig)


def fig_full(decs):
    fig, ax = plt.subplots(figsize=(14, 3.2))
    for y, key, label in ((3, "csv", "CSV"), (2, "exp1", "Exp 1"), (1, "exp2", "Exp 2"), (0, "exp2b", "Exp 2b")):
        th = np.concatenate([decs["train"][key].theme, decs["test"][key].theme])
        _track(ax, np.arange(len(th))[::5] / 3600, th[::5], y, label)
        ses = np.concatenate([decs["train"][key].session, decs["test"][key].session])
        gaps = np.flatnonzero(~ses)
        ax.scatter(gaps[::5] / 3600, np.full(len(gaps[::5]), y - 0.35), s=1, color="0.4")
    ax.axvline(SPLIT_S / 3600, color="tab:blue", lw=1.5)
    ax.set_ylim(-0.7, 3.6)
    ax.set_yticks([])
    ax.set_xlabel("hours (grey dots under a row: background; blue: train/test split)")
    ax.set_title("Decoded themes over the whole recording")
    fig.tight_layout()
    fig.savefig(FIG / "compare_full_timeline.png", dpi=120)
    plt.close(fig)


def fig_confusion(rep):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, MODELS, strict=True):
        c = np.asarray(rep["B"]["test"][key]["confusion"], float)
        c = c / np.maximum(c.sum(1, keepdims=True), 1)
        ax.imshow(c, vmin=0, vmax=1, cmap="Blues")
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if c[i, j] < 0.6 else "w")
        ax.set_xticks(range(N_THEMES + 1), [f"T{k + 1}" for k in range(N_THEMES)] + ["BG"])
        ax.set_yticks(range(N_THEMES), [f"T{k + 1}" for k in range(N_THEMES)])
        ax.set(xlabel="decoded", ylabel="CSV", title=f"{report_text.NAMES[key]} (test, row-normalized)")
    fig.tight_layout()
    fig.savefig(FIG / "compare_confusion.png", dpi=120)
    plt.close(fig)


def fig_songs(decs):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    bins = np.arange(0, 2401, 120)
    for key in ("csv", *MODELS):
        lengths = [s.end_s - s.start_s for h in HALVES for s in decs[h][key].songs if s.complete]
        ax.hist(lengths, bins=bins, histtype="step", lw=1.6, label=f"{report_text.NAMES[key]} ({len(lengths)})")
    ax.set(xlabel="complete song length (s), both halves", ylabel="songs", title="Song lengths")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "compare_song_lengths.png", dpi=120)
    plt.close(fig)


def fig_loglik(fit):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for run in ("exp2", "exp2b"):
        h = np.asarray(fit[run]["history"]) / SPLIT_S * 2
        ax.plot(h, marker="o", ms=3, label=report_text.NAMES[run])
    ax.set(xlabel="Baum-Welch iteration", ylabel="train log-likelihood / window", title="Experiment 2 training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "exp2_loglik.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    exp1 = ensure_exp1()
    f = load_or_build(WAV_PATH, SPLIT_S, WORK / "phrase_features.npz")
    labels = load_labels(CSV_PATH, int(f["n_seconds"]))
    decs = load_decoded(f, labels)
    majority = int(np.bincount(labels.theme[:SPLIT_S][labels.theme[:SPLIT_S] >= 0]).argmax())

    rep: dict = {"B": {}, "C": {}, "D": {}, "init": {}}
    curves = {}
    for half, offset in (("train", 0), ("test", SPLIT_S)):
        csv_starts = labels.song_bounds[:, 0] - offset
        rep["B"][half] = {k: csv_agreement(decs[half][k], decs[half]["csv"].theme, csv_starts, majority) for k in MODELS}
        rep["B"][half]["chance_balanced"] = 1 / N_THEMES
        zn = unit_rows(f[f"z_{half}"])
        curves[half] = novelty_curve(zn)
        peaks = novelty_peaks(curves[half])
        rep["C"][half] = {k: label_free(decs[half][k], zn, f[f"t_{half}"], offset, peaks, decs["train"][k]) for k in ("csv", *MODELS)}
        rep["C"][half]["n_novelty_peaks"] = int(len(peaks))
    print("synthetic recovery ...", flush=True)
    rep["D"]["exp2"] = synthetic(Params.load(HERE / "exp2_model.npz"), len(f["X_train"]))
    rep["D"]["exp1"] = exp1["synthetic"]
    rep["A"] = fit_metrics(f)
    rep["A"]["exp1"] = {"train": exp1["train"]["loglik_per_s"], "test": exp1["test"]["loglik_per_s"], "baselines": exp1["baselines"]}
    rep["init"]["exp2"] = json.loads((WORK / "exp2" / "init.json").read_text(encoding="utf-8"))
    rep["pca"] = pca_readout(f)
    rep["features"] = {
        "spectrogram_shape": f["spectrogram_shape"].tolist(),
        "n_bands": int(len(f["band_edges"])),
        "windows": {h: int(len(f[f"X_{h}"])) for h in HALVES},
        "descriptor_dim": int(f["z_train"].shape[1]),
        "pca_dim": int(f["X_train"].shape[1]),
        "pca_explained": float(f["explained"].sum()),
    }
    rep["params"] = {}
    for run in ("exp2", "exp2b"):
        p = Params.load(HERE / f"{run}_model.npz")
        rep["params"][run] = {"dwell": p.dwell.tolist(), "pause": p.pause.tolist(), "bg": p.bg.tolist(), "start": p.start.tolist()}
    rep["exp1"] = {"params": exp1["params"], "history_len": len(exp1["history"])}

    fig_ssm(f, decs["test"])
    fig_timeline_zoom(f, decs["test"], curves["test"])
    fig_full(decs)
    fig_confusion(rep)
    fig_songs(decs)
    fig_loglik(rep["A"])

    (WORK / "report.json").write_text(json.dumps(rep, indent=2, default=lambda o: o.item()), encoding="utf-8")
    text = report_text.render(rep)
    missing = [h for h in REQUIRED if h not in text]
    if missing:
        raise RuntimeError(f"report is missing required sections: {missing}")
    print(f"wrote {write_report(text, HERE, HERE / 'report.md')}")


if __name__ == "__main__":
    main()
