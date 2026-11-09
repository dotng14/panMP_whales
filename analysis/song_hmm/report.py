"""Model report: train vs test metrics, baselines, synthetic recovery, figures.

    python -m analysis.song_hmm.report

Run ``python -m analysis.song_hmm.train`` first. Writes ``report.md`` and
``figures/`` next to this file, and ``work/report.json``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .decode import segment, song_boundaries  # noqa: E402
from .features import FEATURE_NAMES  # noqa: E402
from .hmm import baum_welch, initial_params, viterbi  # noqa: E402
from .labels import Labels, load_labels  # noqa: E402
from .topology import (  # noqa: E402
    BG_FIRST,
    ENTRY_STATES,
    FIRST_SLOT,
    IS_BG,
    N_DUR,
    N_SLOTS,
    N_THEMES,
    PAUSE0,
    SLOT_THEME,
    SLOT_UNITS,
    STATE_SLOT,
    STATE_THEME,
    Params,
    duration_pmf,
)
from .train import CSV_PATH, LEVEL_COL, MODEL_PATH, SPLIT_S, WORK  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
HALVES = ("train", "test")
TOLS = (5, 15)
EDGE_GUARD_S = 15


# ---------------------------------------------------------------- helpers


def match_events(pred: np.ndarray, ref: np.ndarray, tol: float) -> dict:
    """Greedy one-to-one matching of event times within ``tol`` seconds."""
    pred, ref = np.sort(pred), np.sort(ref)
    used = np.zeros(len(pred), dtype=bool)
    offsets = []
    for r in ref:
        if len(pred) == 0:
            break
        d = np.abs(pred - r).astype(float)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            offsets.append(pred[j] - r)
    tp = len(offsets)
    precision = tp / len(pred) if len(pred) else float("nan")
    recall = tp / len(ref) if len(ref) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"tp": tp, "n_pred": int(len(pred)), "n_ref": int(len(ref)), "precision": precision, "recall": recall, "f1": f1}


def nearest_offsets(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if len(pred) == 0:
        return np.array([])
    return np.array([pred[np.argmin(np.abs(pred - r))] - r for r in ref], dtype=float)


def diag_gauss_ll(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (((x - mean) ** 2 / var).sum(axis=1) + np.log(2 * np.pi * var).sum())


def two_state_hmm(train: np.ndarray, test: np.ndarray, level_col: int, n_iter: int = 30) -> tuple[float, float]:
    """Baseline: background + one 'song' Gaussian, ergodic 2-state HMM."""
    quiet = train[:, level_col] <= np.quantile(train[:, level_col], 0.1)
    mean = np.array([train[quiet].mean(0), train[~quiet].mean(0)])
    var = np.array([train[quiet].var(0), train[~quiet].var(0)]) + 1e-2
    a = np.array([[0.99, 0.01], [0.01, 0.99]])
    pi = np.array([0.5, 0.5])

    def run(x):
        lb = np.column_stack([diag_gauss_ll(x, mean[k], var[k]) for k in range(2)])
        shift = lb.max(1, keepdims=True)
        b = np.exp(lb - shift)
        alpha = np.empty_like(b)
        c = np.empty(len(x))
        al = pi * b[0]
        c[0] = al.sum()
        alpha[0] = al / c[0]
        for t in range(1, len(x)):
            al = (alpha[t - 1] @ a) * b[t]
            c[t] = al.sum()
            alpha[t] = al / c[t]
        return alpha, c, b, float(np.log(c).sum() + shift.sum())

    for _ in range(n_iter):
        alpha, c, b, _ = run(train)
        beta = np.ones_like(alpha)
        xi = np.zeros((2, 2))
        for t in range(len(train) - 2, -1, -1):
            w = b[t + 1] * beta[t + 1] / c[t + 1]
            xi += alpha[t][:, None] * a * w[None, :]
            beta[t] = a @ w
        gamma = alpha * beta
        for k in range(2):
            g = gamma[:, k]
            mean[k] = g @ train / g.sum()
            var[k] = np.maximum(g @ (train**2) / g.sum() - mean[k] ** 2, 1e-2)
        a = xi / xi.sum(1, keepdims=True)
        pi = gamma[0] / gamma[0].sum()
    return run(train)[3], run(test)[3]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def phrases_per_theme(path: np.ndarray, songs: list) -> np.ndarray:
    """Mean number of phrases (entries into Tk.1) per theme over complete songs."""
    rows = []
    for s in songs:
        if s.partial_start or s.partial_end:
            continue
        seg = path[s.start_s : s.end_s]
        entered = np.r_[True, seg[1:] != seg[:-1]]
        rows.append([int(np.sum(entered & (seg == ENTRY_STATES[k]))) for k in range(N_THEMES)])
    return np.mean(rows, axis=0) if rows else np.full(N_THEMES, np.nan)


# ---------------------------------------------------------------- per-half metrics


def half_metrics(name: str, x: np.ndarray, labels: Labels, p: Params, offset: int) -> dict:
    post = np.load(WORK / name / "posteriors.npz")
    path = post["path"]
    n = len(path)
    sessions, songs = segment(path, 0)  # local seconds; offset added where reported
    csv_theme = labels.theme[offset : offset + n]
    dec_theme = STATE_THEME[path]
    is_bg = IS_BG[path]
    is_pause = (path >= PAUSE0) & (path < BG_FIRST)

    # boundaries
    pred_b = song_boundaries(path) + offset
    ref = labels.song_bounds[1:, 0]  # every CSV song start that follows another song
    ref = ref[(ref >= offset + EDGE_GUARD_S) & (ref < offset + n - EDGE_GUARD_S)]
    boundary = {f"tol_{t}s": match_events(pred_b, ref, t) for t in TOLS}
    near = nearest_offsets(pred_b, ref)

    complete = [s for s in songs if not (s.partial_start or s.partial_end)]
    in_half = (labels.song_bounds[:, 0] >= offset) & (labels.song_bounds[:, 1] <= offset + n)

    labeled = csv_theme >= 0
    conf = np.zeros((N_THEMES, N_THEMES + 1), dtype=int)  # last column = background
    np.add.at(conf, (csv_theme[labeled], np.where(dec_theme[labeled] >= 0, dec_theme[labeled], N_THEMES)), 1)

    return {
        "n_seconds": n,
        "loglik_per_s": float(post["loglik"]) / n,
        "sessions": {
            "n": len(sessions),
            "seconds_in_session": int((~is_bg).sum()),
            "song_fraction": float((~is_bg).mean()),
            "breath_fraction": float(is_pause.mean()),
            "background_fraction": float(is_bg.mean()),
            "csv_song_seconds_decoded_as_song": float((~is_bg[labeled]).mean()),
            "mean_p_song": float(post["p_song"].mean()),
            "lengths_s": [s.end_s - s.start_s for s in sessions],
        },
        "songs": {
            "n_decoded": len(songs),
            "n_complete": len(complete),
            "n_csv_in_half": int(in_half.sum()),
            "complete_lengths_s": [s.end_s - s.start_s for s in complete],
            "csv_lengths_s": (labels.song_bounds[in_half, 1] - labels.song_bounds[in_half, 0]).tolist(),
            "boundaries": boundary,
            "median_abs_offset_nearest_s": float(np.median(np.abs(near))) if near.size else float("nan"),
        },
        "themes": {
            "accuracy": float((dec_theme[labeled] == csv_theme[labeled]).mean()),
            "accuracy_in_session": float((dec_theme[labeled & ~is_bg] == csv_theme[labeled & ~is_bg]).mean()),
            "posterior_on_csv_theme": float(post["p_theme"][np.flatnonzero(labeled), csv_theme[labeled]].mean()),
            "confusion": conf.tolist(),
            "decoded_seconds": np.bincount(dec_theme[dec_theme >= 0], minlength=N_THEMES).tolist(),
            "csv_seconds": np.bincount(csv_theme[labeled], minlength=N_THEMES).tolist(),
            "decoded_phrases_per_song": phrases_per_theme(path, songs).tolist(),
            "csv_phrases_per_song": labels.phrases[in_half].mean(axis=0).tolist() if in_half.any() else [],
        },
        "_path": path,
        "_p_song": post["p_song"],
        "_boundaries": pred_b,
        "_sessions": [(s.start_s + offset, s.end_s + offset) for s in sessions],
    }


# ---------------------------------------------------------------- synthetic recovery


def synthetic_recovery(p: Params, n: int, seed: int = 7) -> dict:
    from .hmm import sample

    rng = np.random.default_rng(seed)
    x, true = sample(p, n, rng)
    slots = STATE_SLOT[true]
    hist = np.zeros((N_SLOTS, N_DUR))
    change = np.flatnonzero(np.diff(np.r_[-2, slots, -2]))
    for a, b in zip(change[:-1], change[1:], strict=True):
        if slots[a] >= 0:
            hist[slots[a], min(b - a, N_DUR) - 1] += 1
    p0 = initial_params(x, slots, hist, LEVEL_COL)
    fit, _ = baum_welch([x], p0, verbose=False)
    path, _ = viterbi(x, fit)

    def session_edges(pth):
        s, _ = segment(pth)
        return np.array([v.start_s for v in s if v.start_s > 0] + [v.end_s for v in s if v.end_s < n])

    true_th, dec_th = STATE_THEME[true], STATE_THEME[path]
    sung = true_th >= 0
    return {
        "n_seconds": n,
        "true_sessions": len(segment(true)[0]),
        "true_song_boundaries": int(len(song_boundaries(true))),
        "song_vs_background_accuracy": float((IS_BG[true] == IS_BG[path]).mean()),
        "session_edges": {f"tol_{t}s": match_events(session_edges(path), session_edges(true), t) for t in TOLS},
        "song_boundaries": {f"tol_{t}s": match_events(song_boundaries(path), song_boundaries(true), t) for t in TOLS},
        "theme_accuracy": float((dec_th[sung] == true_th[sung]).mean()),
        "slot_accuracy": float((STATE_SLOT[path][sung] == STATE_SLOT[true][sung]).mean()),
    }


# ---------------------------------------------------------------- figures

THEME_COLORS = plt.cm.tab10(np.arange(N_THEMES))


def _theme_track(ax, t, theme, y, label):
    for k in range(N_THEMES):
        m = theme == k
        ax.scatter(t[m] / 3600, np.full(m.sum(), y), s=2, marker="|", color=THEME_COLORS[k], linewidths=0.6)
    ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)


def figures(model: dict, feats: dict, labels: Labels, halves: dict, synth: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    history = model["history"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(np.arange(len(history)), np.asarray(history) / SPLIT_S, marker="o", ms=3)
    ax.set(xlabel="Baum-Welch iteration", ylabel="train log-likelihood / s", title="Training convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "loglik.png", dpi=140)
    plt.close(fig)

    raw = feats["raw"]
    n = len(raw)
    path = np.concatenate([halves["train"]["_path"], halves["test"]["_path"]])
    p_song = np.concatenate([halves["train"]["_p_song"], halves["test"]["_p_song"]])
    bounds = np.concatenate([halves["train"]["_boundaries"], halves["test"]["_boundaries"]])
    sessions = halves["train"]["_sessions"] + halves["test"]["_sessions"]
    t = np.arange(n)

    def timeline(lo, hi, fname, bin_s):
        sl = slice(lo, hi)
        k = (hi - lo) // bin_s
        binned = raw[sl][: k * bin_s].reshape(k, bin_s, -1).mean(1)
        tb = (lo + np.arange(k) * bin_s + bin_s / 2) / 3600
        fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1, 1.1]})
        edges = feats["band_edges"]
        axes[0].pcolormesh(np.r_[tb - bin_s / 7200, tb[-1] + bin_s / 7200], edges, binned[:, :16].T, shading="flat", cmap="magma", vmin=0, vmax=np.percentile(binned[:, :16], 99))
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Hz (level above floor)")
        axes[1].plot(tb, binned[:, LEVEL_COL], lw=0.7, color="k")
        axes[1].set_ylabel("broadband\nlevel")
        axes[2].plot(t[sl] / 3600, p_song[sl], lw=0.6, color="tab:green")
        axes[2].set_ylabel("P(song)")
        axes[2].set_ylim(-0.05, 1.05)
        _theme_track(axes[3], t[sl], STATE_THEME[path[sl]], 1.0, "decoded")
        _theme_track(axes[3], t[sl], labels.theme[sl], 0.0, "CSV")
        axes[3].set_ylim(-0.6, 1.6)
        axes[3].set_yticks([])
        for k in range(N_THEMES):
            axes[3].plot([], [], color=THEME_COLORS[k], lw=3, label=f"T{k + 1}")
        axes[3].legend(ncol=6, fontsize=7, loc="upper right", bbox_to_anchor=(1, 1.35), frameon=False)
        for a, b in sessions:
            if b > lo and a < hi:
                for ax in axes[1:3]:
                    ax.axvspan(max(a, lo) / 3600, min(b, hi) / 3600, color="tab:green", alpha=0.08, lw=0)
        for bt in bounds[(bounds >= lo) & (bounds < hi)]:
            for ax in axes[1:]:
                ax.axvline(bt / 3600, color="tab:red", lw=0.6, alpha=0.7)
        for bt in labels.song_bounds[:, 0]:
            if lo <= bt < hi:
                axes[3].axvline(bt / 3600, color="k", lw=0.6, ls=":", alpha=0.7)
        if lo < SPLIT_S < hi:
            for ax in axes:
                ax.axvline(SPLIT_S / 3600, color="tab:blue", lw=1.5)
        axes[-1].set_xlabel("hours from recording start (red: decoded T6→T1, dotted: CSV song start, blue: train/test split)")
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=130)
        plt.close(fig)

    timeline(0, n, "timeline.png", 10)
    timeline(SPLIT_S + 2 * 3600, SPLIT_S + 2 * 3600 + 1800, "timeline_zoom_test.png", 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, h in zip(axes, HALVES, strict=True):
        conf = np.asarray(halves[h]["themes"]["confusion"], dtype=float)
        conf = conf / np.maximum(conf.sum(1, keepdims=True), 1)
        im = ax.imshow(conf, vmin=0, vmax=1, cmap="Blues")
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if conf[i, j] < 0.6 else "w")
        ax.set_xticks(range(N_THEMES + 1), [f"T{k + 1}" for k in range(N_THEMES)] + ["BG"])
        ax.set_yticks(range(N_THEMES), [f"T{k + 1}" for k in range(N_THEMES)])
        ax.set(xlabel="decoded", ylabel="CSV", title=f"{h}: theme confusion (row-normalized)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(FIG / "confusion.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    bins = np.arange(0, 1801, 60)
    for ax, h in zip(axes, HALVES, strict=True):
        s = halves[h]["songs"]
        ax.hist(s["csv_lengths_s"], bins=bins, alpha=0.6, label=f"CSV ({len(s['csv_lengths_s'])})")
        ax.hist(s["complete_lengths_s"], bins=bins, alpha=0.6, label=f"decoded complete ({len(s['complete_lengths_s'])})")
        ax.set(xlabel="song length (s)", title=h)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("songs")
    fig.tight_layout()
    fig.savefig(FIG / "song_lengths.png", dpi=140)
    plt.close(fig)

    p = model["params"]
    pmf = np.array([duration_pmf(p.exit_prob[s]) for s in range(N_SLOTS)])
    fig, ax = plt.subplots(figsize=(11, 2.8))
    im = ax.imshow(pmf.T, aspect="auto", cmap="viridis", origin="lower", extent=(-0.5, N_SLOTS - 0.5, 0.5, N_DUR + 0.5))
    names = [f"T{SLOT_THEME[s] + 1}.{s - FIRST_SLOT[SLOT_THEME[s]] + 1}\n{SLOT_UNITS[s]}" for s in range(N_SLOTS)]
    ax.set_xticks(range(N_SLOTS), names, fontsize=6)
    ax.set_ylabel("slot length (s)")
    ax.set_title("Learned slot length distributions")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "slot_lengths.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    w = 0.38
    for ax, h in zip(axes, HALVES, strict=True):
        th = halves[h]["themes"]
        x_ = np.arange(N_THEMES)
        ax.bar(x_ - w / 2, th["csv_seconds"], w, label="CSV")
        ax.bar(x_ + w / 2, th["decoded_seconds"], w, label="decoded")
        ax.set_xticks(x_, [f"T{k + 1}" for k in x_])
        ax.set_title(f"{h}: seconds per theme")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "theme_seconds.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- markdown


def _f(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def write_markdown(rep: dict) -> None:
    tr, te = rep["train"], rep["test"]
    base = rep["baselines"]
    syn = rep["synthetic"]
    p = rep["params"]
    lines = [
        "# Song-session HMM: model report",
        "",
        f"Recording `{rep['recording']}`: 10 h at 2 kHz. **Train** = first 5 h (0–{SPLIT_S} s), **test** = last 5 h. "
        "Features, initialization and Baum–Welch use the train half only. Architecture and choices: [plan.md](plan.md).",
        "",
        "The CSV reference covers every second of the recording with whole-second units, and some of its songs skip themes. "
        "It looks generated, so agreement with it is **agreement with the reference, not accuracy**. "
        "The synthetic section is the only score against a true path.",
        "",
        "## 1. Fit and generalization",
        "",
        "| log-likelihood / s | train | test |",
        "|---|---|---|",
        f"| song-session HMM | {_f(tr['loglik_per_s'])} | {_f(te['loglik_per_s'])} |",
        f"| one Gaussian for every second | {_f(base['one_gaussian']['train'])} | {_f(base['one_gaussian']['test'])} |",
        f"| 2-state HMM (background + one song Gaussian) | {_f(base['two_state']['train'])} | {_f(base['two_state']['test'])} |",
        "",
        f"Baum–Welch ran {len(rep['history']) - 1} iterations. ![](figures/loglik.png)",
        "",
        "## 2. Session detection",
        "",
        "| | train | test |",
        "|---|---|---|",
    ]
    for key, label in [
        ("n", "sessions"),
        ("song_fraction", "fraction of seconds in a session"),
        ("breath_fraction", "fraction in breath states (inside sessions)"),
        ("background_fraction", "fraction in background"),
        ("csv_song_seconds_decoded_as_song", "CSV-song seconds decoded as song"),
        ("mean_p_song", "mean P(song)"),
    ]:
        a, b = tr["sessions"][key], te["sessions"][key]
        lines.append(f"| {label} | {a if key == 'n' else _f(a)} | {b if key == 'n' else _f(b)} |")
    med = [f"{int(np.median(ls))} / {max(ls)}" if ls else "–" for ls in (rep[h]["sessions"]["lengths_s"] for h in HALVES)]
    lines.append(f"| median / max session length (s) | {med[0]} | {med[1]} |")
    lines += [
        "",
        "![](figures/timeline.png)",
        "",
        "30 minutes of the test half at 1 s resolution:",
        "",
        "![](figures/timeline_zoom_test.png)",
        "",
        "## 3. Song boundaries (T6 → T1)",
        "",
        "| | train | test |",
        "|---|---|---|",
        f"| decoded songs (incl. partial) | {tr['songs']['n_decoded']} | {te['songs']['n_decoded']} |",
        f"| complete songs (T1 … T6) | {tr['songs']['n_complete']} | {te['songs']['n_complete']} |",
        f"| CSV songs fully inside the half | {tr['songs']['n_csv_in_half']} | {te['songs']['n_csv_in_half']} |",
    ]
    for tol in TOLS:
        a, b = tr["songs"]["boundaries"][f"tol_{tol}s"], te["songs"]["boundaries"][f"tol_{tol}s"]
        lines.append(
            f"| boundary P / R / F1 at ±{tol} s vs CSV song starts | {_f(a['precision'], 2)} / {_f(a['recall'], 2)} / {_f(a['f1'], 2)} "
            f"({a['tp']}/{a['n_pred']} pred, {a['n_ref']} ref) | {_f(b['precision'], 2)} / {_f(b['recall'], 2)} / {_f(b['f1'], 2)} "
            f"({b['tp']}/{b['n_pred']} pred, {b['n_ref']} ref) |"
        )
    lines += [
        f"| median offset of nearest decoded boundary to each CSV start (s) | {_f(tr['songs']['median_abs_offset_nearest_s'], 0)} | {_f(te['songs']['median_abs_offset_nearest_s'], 0)} |",
        "",
        "![](figures/song_lengths.png)",
        "",
        "## 4. Theme labeling",
        "",
        "| | train | test |",
        "|---|---|---|",
        f"| per-second theme agreement with CSV (background counts as wrong) | {_f(tr['themes']['accuracy'])} | {_f(te['themes']['accuracy'])} |",
        f"| same, inside decoded sessions only | {_f(tr['themes']['accuracy_in_session'])} | {_f(te['themes']['accuracy_in_session'])} |",
        f"| mean posterior on the CSV theme | {_f(tr['themes']['posterior_on_csv_theme'])} | {_f(te['themes']['posterior_on_csv_theme'])} |",
        f"| always-majority-theme baseline | {_f(rep['majority_theme']['train'])} | {_f(rep['majority_theme']['test'])} |",
        "",
        "![](figures/confusion.png)",
        "",
        "![](figures/theme_seconds.png)",
        "",
        "Phrases per complete song:",
        "",
        "| theme | " + " | ".join(f"T{k + 1}" for k in range(N_THEMES)) + " |",
        "|---|" + "---|" * N_THEMES,
    ]
    for h in HALVES:
        th = rep[h]["themes"]
        lines.append(f"| decoded, {h} | " + " | ".join(_f(v, 1) for v in th["decoded_phrases_per_song"]) + " |")
        if th["csv_phrases_per_song"]:
            lines.append(f"| CSV, {h} | " + " | ".join(_f(v, 1) for v in th["csv_phrases_per_song"]) + " |")
    lines += [
        "",
        "## 5. Learned structure",
        "",
        "End-of-theme branch (from the last slot of each theme) and breath states. "
        f"`end` is fixed at {p['branch'][0][3]:.0e} and the background self-loop at {p['bg'][0]:.5f} (a prior, not learned).",
        "",
        "| theme | repeat | next | breath | end | breath: stay | resume | next |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in range(N_THEMES):
        b, q = p["branch"][k], p["pause"][k]
        lines.append(f"| T{k + 1} | {b[0]:.3f} | {b[1]:.3f} | {b[2]:.3f} | {b[3]:.0e} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} |")
    enter = np.asarray(p["bg"][1:])
    lines += [
        "",
        "Session entry (background → theme): " + ", ".join(f"T{k + 1} {v / enter.sum():.2f}" for k, v in enumerate(enter)) + ".",
        "",
        "![](figures/slot_lengths.png)",
        "",
        "## 6. Synthetic recovery",
        "",
        f"{syn['n_seconds']} s sampled from the fitted model ({syn['true_sessions']} sessions, {syn['true_song_boundaries']} song boundaries), "
        "re-initialized from the true slot path the same way the CSV seeds the real fit, refit, and decoded.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| song vs background, per second | {_f(syn['song_vs_background_accuracy'])} |",
        f"| theme accuracy (sung seconds) | {_f(syn['theme_accuracy'])} |",
        f"| slot accuracy (sung seconds) | {_f(syn['slot_accuracy'])} |",
    ]
    for tol in TOLS:
        a, b = syn["session_edges"][f"tol_{tol}s"], syn["song_boundaries"][f"tol_{tol}s"]
        lines.append(f"| session start/end F1 at ±{tol} s | {_f(a['f1'], 2)} ({a['tp']}/{a['n_ref']}) |")
        lines.append(f"| song boundary F1 at ±{tol} s | {_f(b['f1'], 2)} ({b['tp']}/{b['n_ref']}) |")
    lines.append("")
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- main


def main() -> None:
    feats = dict(np.load(WORK / "features.npz"))
    x = feats["x"]
    labels = load_labels(CSV_PATH, len(x))
    p = Params.load(MODEL_PATH)
    with np.load(MODEL_PATH) as m:
        history = m["history"].tolist()
    train, test = x[:SPLIT_S], x[SPLIT_S:]

    halves = {"train": half_metrics("train", train, labels, p, 0), "test": half_metrics("test", test, labels, p, SPLIT_S)}

    mean, var = train.mean(0), train.var(0)
    two_tr, two_te = two_state_hmm(train, test, LEVEL_COL)
    major = int(np.bincount(labels.theme[:SPLIT_S][labels.theme[:SPLIT_S] >= 0]).argmax())
    print("synthetic recovery ...", flush=True)
    synth = synthetic_recovery(p, SPLIT_S)

    rep = {
        "recording": "SOCAL_H_76_230426_193500_df100.x.x.wav",
        "split_s": SPLIT_S,
        "features": list(FEATURE_NAMES),
        "history": history,
        "baselines": {
            "one_gaussian": {"train": float(diag_gauss_ll(train, mean, var).mean()), "test": float(diag_gauss_ll(test, mean, var).mean())},
            "two_state": {"train": two_tr / len(train), "test": two_te / len(test)},
        },
        "majority_theme": {
            h: float((labels.theme[sl][labels.theme[sl] >= 0] == major).mean())
            for h, sl in (("train", slice(0, SPLIT_S)), ("test", slice(SPLIT_S, None)))
        },
        "params": {"branch": p.branch.tolist(), "pause": p.pause.tolist(), "bg": p.bg.tolist(), "start": p.start.tolist()},
        "synthetic": synth,
        **{h: {k: v for k, v in halves[h].items() if not k.startswith("_")} for h in HALVES},
    }
    figures({"history": history, "params": p}, feats, labels, halves, synth)
    (WORK / "report.json").write_text(json.dumps(rep, indent=2, default=lambda o: o.item()), encoding="utf-8")
    write_markdown(rep)
    print(f"wrote {HERE / 'report.md'}")


if __name__ == "__main__":
    main()
