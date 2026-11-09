"""Figures for the Experiment 3 report (written to analysis/song_hmm/exp3/figures/)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from analysis.song_hmm.exp1_slot_hmm.train import SPLIT_S, WAV_PATH  # noqa: E402

from .evaluate import NAMES, load_decoded  # noqa: E402
from .inputs.spectrogram import N_OPTIONS, S_OPTIONS, SR, normalise, spectrogram  # noqa: E402
from .models.common import HERE, N_THEMES, WORK, labels  # noqa: E402

FIG = HERE / "figures"
COLORS = plt.cm.tab10(np.arange(N_THEMES))
ZOOM = (SPLIT_S + 7200, SPLIT_S + 9000)  # 30 min of the test half (7.0-7.5 h)
TRACKS = ("csv", "exp2b", "M1", "M2", "M3", "M4")


def screening_heatmap() -> None:
    win = json.loads((WORK / "inputs" / "screen.json").read_text())
    rows = [f"{s}·{n}" for s in S_OPTIONS for n in N_OPTIONS]
    cols = [("X1", 12), ("X1", 20), ("X2", 12), ("X2", 20), ("X5", 20)]
    m = np.full((len(rows), len(cols)), np.nan)
    for i, s in enumerate(S_OPTIONS):
        for j_n, n in enumerate(N_OPTIONS):
            for j, (x, w) in enumerate(cols):
                r = win.get(f"{s}_{n}_{x}_{w}s")
                if r:
                    m[i * len(N_OPTIONS) + j_n, j] = r["patch"]["repeat_hit"] if x == "X5" else r["pca20"]["repeat_hit"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(m, cmap="viridis", aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if m[i, j] < np.nanmean(m) else "k")
    ax.set_xticks(range(len(cols)), [f"{x} {w} s" + (" (patch)" if x == "X5" else " PCA-20") for x, w in cols], fontsize=8, rotation=20)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_title("Input screening: song-repeat hit rate (test half; chance ≈ 0.05)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "screening.png", dpi=120)
    plt.close(fig)


def decoded_all():
    out = {}
    for h, off in (("train", 0), ("test", SPLIT_S)):
        out[h] = {m: load_decoded(m, h, off) for m in ("exp2", "exp2b", "M1", "M2", "M3", "M4")}
    return out


def timeline(decs: dict, hand: dict | None) -> None:
    lo, hi = ZOOM
    lab = labels()
    sr, a = wavfile.read(WAV_PATH, mmap=True)
    x = np.asarray(a[lo * SR : hi * SR], np.float32)
    x -= x.mean()
    sp = spectrogram(x, "S2", hi - lo)
    v = normalise(sp, "N3")
    rows = [("csv", "CSV")] + ([("hand", "hand labels")] if hand is not None else []) + [(m, NAMES[m]) for m in TRACKS[1:]]
    fig, axes = plt.subplots(2, 1, figsize=(14, 3 + 0.45 * len(rows)), sharex=True, gridspec_kw={"height_ratios": [2.2, 0.32 * len(rows) + 0.6]})
    axes[0].imshow(v[::4].T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=np.percentile(v, 99.5), extent=(lo / 3600, hi / 3600, 0, v.shape[1]))
    axes[0].set_ylabel("band (30 Hz → 1 kHz)")
    axes[0].set_title("30 minutes of the test half: spectrogram and every labelling (black ticks: song boundaries)")
    t = np.arange(lo, hi)
    for y, (key, name) in enumerate(reversed(rows)):
        if key == "csv":
            th = lab.theme[lo:hi]
            bnd = lab.song_bounds[:, 0]
        elif key == "hand":
            th = hand["theme"][lo:hi]
            bnd = hand["song_starts"]
        else:
            d = decs["test"].get(key)
            if d is None:
                continue
            th = d.theme[lo - SPLIT_S : hi - SPLIT_S]
            bnd = d.boundaries + SPLIT_S
        for k in range(N_THEMES):
            m = th == k
            axes[1].scatter(t[m] / 3600, np.full(m.sum(), y), s=4, marker="|", color=COLORS[k], linewidths=1)
        for b in bnd:
            if lo <= b < hi:
                axes[1].plot([b / 3600] * 2, [y - 0.42, y + 0.42], color="k", lw=1.3)
        axes[1].text(-0.005, y, name, transform=axes[1].get_yaxis_transform(), ha="right", va="center", fontsize=8)
    axes[1].set_yticks([])
    axes[1].set_ylim(-0.7, len(rows) - 0.3)
    for k in range(N_THEMES):
        axes[1].plot([], [], color=COLORS[k], lw=3, label=f"T{k + 1}")
    axes[1].legend(ncol=6, fontsize=7, loc="upper right", bbox_to_anchor=(1, 1.18), frameon=False)
    axes[1].set_xlabel("hours from the start of the recording")
    fig.tight_layout()
    fig.savefig(FIG / "timeline.png", dpi=115)
    plt.close(fig)


def song_lengths(decs: dict) -> None:
    lab = labels()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    bins = np.arange(0, 3601, 120) / 60
    csv_len = np.diff(lab.song_bounds, axis=1).ravel() / 60
    ax.hist(csv_len, bins=bins, histtype="step", lw=2, color="k", label=f"CSV ({len(csv_len)})")
    for m in TRACKS[1:]:
        L = [(s.end_s - s.start_s) / 60 for h in decs for s in (decs[h][m].songs if decs[h][m] else []) if s.complete]
        if L:
            ax.hist(L, bins=bins, histtype="step", lw=1.4, label=f"{NAMES[m]} ({len(L)})")
    ax.axvspan(7, 30, color="0.9", zorder=0, label="Payne & McVay range (7-30 min)")
    ax.set(xlabel="complete song length (min), both halves", ylabel="songs", xlim=(0, 60))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "song_lengths.png", dpi=120)
    plt.close(fig)


def confusions(ev: dict) -> None:
    keys = [m for m in ("exp2b", "M1", "M2", "M3", "M4") if m in ev["B"]["test"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.2))
    for ax, m in zip(np.atleast_1d(axes), keys, strict=True):
        c = np.asarray(ev["B"]["test"][m]["confusion"], float)
        c = c / np.maximum(c.sum(1, keepdims=True), 1)
        ax.imshow(c, vmin=0, vmax=1, cmap="Blues")
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=6, color="k" if c[i, j] < 0.6 else "w")
        ax.set_xticks(range(7), [f"T{k + 1}" for k in range(6)] + ["BG"], fontsize=7)
        ax.set_yticks(range(6), [f"T{k + 1}" for k in range(6)], fontsize=7)
        ax.set_title(NAMES[m], fontsize=9)
        ax.set_xlabel("decoded", fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("CSV", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "confusions.png", dpi=120)
    plt.close(fig)


def m4_auc(res: dict) -> None:
    if "M4" not in res:
        return
    iso, emb = res["M4"]["detection_isolated"], res["M4"]["detection_embedded"]
    k = [f"T{i + 1}" for i in range(N_THEMES)]
    x = np.arange(N_THEMES)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(x - 0.2, [iso[t] for t in k], 0.4, label=f"after isolated training (top-1 {iso['top1_accuracy']:.2f})")
    ax.bar(x + 0.2, [emb[t] for t in k], 0.4, label=f"after embedded training (top-1 {emb['top1_accuracy']:.2f})")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(x, k)
    ax.set(ylim=(0, 1), ylabel="one-vs-rest AUC", title="M4: how well each theme HMM picks out its own theme (test, CSV segments)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "m4_detection.png", dpi=120)
    plt.close(fig)


def all_figures(ev: dict, res: dict, hand: dict | None) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    decs = decoded_all()
    screening_heatmap()
    timeline(decs, hand)
    song_lengths(decs)
    confusions(ev)
    m4_auc(res)
