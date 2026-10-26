"""Fit song_soft_hmm_t6 on the 193500 motif scores.

GPL is a separate long job. This module reads its whitened windows, builds
the panMP motifs, runs Baum–Welch, and writes ``analysis/HMM/song_soft_hmm_t6.npz``.
The graph comes from ``topology``: after theme 6 the next theme is theme 1,
and a training song may end in any theme.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .hmm import (
    SoftHMM,
    baum_welch,
    fit_metrics,
    parameter_sensitivity,
    total_loglik,
)
from .observations import (
    HZ,
    N_PROTOTYPES,
    energy_series_10hz,
    load_slot_labels,
    pan_prototypes,
    scores_per_second,
    song_slices,
    spectrogram_paths,
    zscore_columns,
)
from .topology import (
    N_SLOTS,
    REPEAT_INIT,
    SLOT_UNITS,
    exit_probs_from_duration_hist,
)

WAV_PATH = Path("SOCAL_H_76_230426_193500_df100.x.x_10kHz.wav")
REPO = Path(__file__).resolve().parents[2]
HMM_DIR = REPO / "analysis" / "HMM"
WORK = HMM_DIR / "work"
GPL_DIR = WORK / "gpl_193500"
CSV_PATH = (
    REPO
    / "backend"
    / "data"
    / "song_structure"
    / "SOCAL_H_76_230426_193500_df100.x.x.motif_segments.csv"
)
MODEL_PATH = HMM_DIR / "song_soft_hmm_t6.npz"
N_GPL_WINDOWS = 480


def n_seconds_in_wav(path: Path) -> int:
    import wave

    with wave.open(str(path), "rb") as handle:
        return int(handle.getnframes() // handle.getframerate())


def gpl_is_done(npy_dir: Path) -> bool:
    return len(spectrogram_paths(npy_dir)) >= N_GPL_WINDOWS


def initial_model(scores: np.ndarray, labels: dict[str, np.ndarray]) -> SoftHMM:
    k_dim = scores.shape[1]
    means = np.zeros((N_SLOTS, k_dim), dtype=np.float64)
    variances = np.ones((N_SLOTS, k_dim), dtype=np.float64)
    slot = labels["slot_per_second"]
    for index in range(N_SLOTS):
        chosen = scores[slot == index]
        if len(chosen) < 2:
            means[index] = scores.mean(axis=0)
            variances[index] = np.maximum(scores.var(axis=0), 1e-3)
        else:
            means[index] = chosen.mean(axis=0)
            variances[index] = np.maximum(chosen.var(axis=0), 1e-3)
    exit_prob = np.vstack(
        [exit_probs_from_duration_hist(labels["duration_hist"][index]) for index in range(N_SLOTS)]
    )
    start_prob = np.full(6, 1.0 / 6.0)
    return SoftHMM(means, variances, exit_prob, REPEAT_INIT.copy(), start_prob)


def build_scores(n_seconds: int) -> dict[str, np.ndarray | list]:
    cache = WORK / "motif_scores.npz"
    if cache.is_file():
        stored = np.load(cache, allow_pickle=True)
        prototypes = [np.asarray(item, dtype=np.float64) for item in stored["prototypes"]]
        print(f"loaded motif scores from {cache}", flush=True)
        return {
            "scores": stored["scores"],
            "score_mean": stored["score_mean"],
            "score_std": stored["score_std"],
            "prototypes": prototypes,
            "series": stored["series"],
        }

    npy_dir = GPL_DIR / "spectrogram_npy"
    if not gpl_is_done(npy_dir):
        raise FileNotFoundError(
            f"GPL has {len(spectrogram_paths(npy_dir))} / {N_GPL_WINDOWS} windows in {npy_dir}"
        )
    series_path = WORK / "series_10hz.npy"
    if series_path.is_file():
        series = np.load(series_path)
        print(f"loaded 10 Hz series {series.shape} from {series_path}", flush=True)
    else:
        print("building 10 Hz energy series", flush=True)
        series = energy_series_10hz(npy_dir, n_seconds)
        np.save(series_path, series)
    if len(series) != n_seconds * HZ:
        raise RuntimeError(f"expected {n_seconds * HZ} samples, got {len(series)}")
    prototypes = pan_prototypes(series, n_prototypes=N_PROTOTYPES)
    raw = scores_per_second(series, prototypes, n_seconds)
    scores, score_mean, score_std = zscore_columns(raw)
    np.savez(
        cache,
        scores=scores,
        score_mean=score_mean,
        score_std=score_std,
        prototypes=np.array(prototypes, dtype=object),
        series=series,
    )
    print(f"wrote {cache} with {len(prototypes)} motifs", flush=True)
    return {
        "scores": scores,
        "score_mean": score_mean,
        "score_std": score_std,
        "prototypes": prototypes,
        "series": series,
    }


def save_model(model: SoftHMM, bundle: dict, history: list[float], complete: float, extra: dict) -> None:
    np.savez(
        MODEL_PATH,
        means=model.means,
        vars=model.vars,
        exit_prob=model.exit_prob,
        repeat_prob=model.repeat_prob,
        start_prob=model.start_prob,
        prototypes=np.array(bundle["prototypes"], dtype=object),
        score_mean=bundle["score_mean"],
        score_std=bundle["score_std"],
        loglik_history=np.asarray(history, dtype=np.float64),
        complete_loglik=np.array(complete),
        slot_units=np.array(SLOT_UNITS),
        **extra,
    )


def labeled_phrase_counts() -> np.ndarray:
    """How many CSV phrases each theme has. A phrase is one pass through that theme's slots."""
    counts = np.zeros(6, dtype=np.float64)
    seen: set[tuple[int, int, int]] = set()
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(float(row["song"])), int(float(row["theme"])), int(float(row["phrase_count"])))
            if key in seen:
                continue
            seen.add(key)
            counts[key[1] - 1] += 1.0
    return counts


def load_model(path: Path = MODEL_PATH) -> tuple[SoftHMM, dict]:
    stored = np.load(path, allow_pickle=True)
    if "start_prob" in stored.files:
        start_prob = stored["start_prob"]
    else:
        start_prob = np.full(6, 1.0 / 6.0)
    model = SoftHMM(
        stored["means"],
        stored["vars"],
        stored["exit_prob"],
        stored["repeat_prob"],
        start_prob,
    )
    bundle = {
        "prototypes": [np.asarray(item, dtype=np.float64) for item in stored["prototypes"]],
        "score_mean": stored["score_mean"],
        "score_std": stored["score_std"],
    }
    return model, bundle


def _plot(history, metrics, sensitivity) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = WORK / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stale = fig_dir / "posterior_song.png"
    if stale.is_file():
        stale.unlink()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(1, len(history) + 1), history, marker="o")
    ax.set_xlabel("iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title("Baum–Welch")
    fig.tight_layout()
    fig.savefig(fig_dir / "loglik.png", dpi=120)
    plt.close(fig)

    rate = metrics["song_loglik"] / np.maximum(metrics["song_seconds"], 1.0)
    order = np.argsort(rate)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(rate)), rate[order], color="#3d5a80")
    ax.axhline(float(metrics["normal_per_second"]), color="#c44536", linewidth=1.0, label="standard normal")
    ax.set_xlabel("song, sorted by log-likelihood per second")
    ax.set_ylabel("nats per second")
    ax.set_title("Per-song likelihood")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "song_loglik.png", dpi=120)
    plt.close(fig)

    themes = np.arange(1, 7)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(themes - width / 2, metrics["theme_labeled_seconds"], width=width, label="labeled seconds")
    ax.bar(themes + width / 2, metrics["theme_expected_seconds"], width=width, label="posterior seconds")
    ax.set_xlabel("theme")
    ax.set_ylabel("seconds")
    ax.set_title("Theme occupancy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "theme_occupancy.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(len(sensitivity["branch_drop"])), sensitivity["branch_drop"])
    ax.set_xlabel("theme-end nudge (two per theme, + then -)")
    ax.set_ylabel("log-likelihood drop")
    ax.set_title("Branch sensitivity")
    fig.tight_layout()
    fig.savefig(fig_dir / "branch_sensitivity.png", dpi=120)
    plt.close(fig)


def _json_report(
    n_seconds: int,
    n_songs: int,
    n_motifs: int,
    history: list[float] | np.ndarray,
    complete: float,
    initial_ll: float,
    metrics: dict[str, np.ndarray],
    sensitivity: dict[str, np.ndarray],
) -> dict:
    history_a = np.asarray(history, dtype=np.float64)
    rate = metrics["song_loglik"] / np.maximum(metrics["song_seconds"], 1.0)
    total_ll = float(history_a[-1])
    return {
        "n_seconds": n_seconds,
        "n_songs": n_songs,
        "n_motifs": n_motifs,
        "loglik_history": history_a.tolist(),
        "loglik": total_ll,
        "loglik_per_second": total_ll / n_seconds,
        "loglik_gain": float(history_a[-1] - history_a[0]),
        "initial_loglik": initial_ll,
        "normal_per_second": float(metrics["normal_per_second"]),
        "posterior_weighted_likelihood": complete,
        "emission_complete": float(metrics["emission_complete"]),
        "transition_complete": float(metrics["transition_complete"]),
        "start_complete": None,
        "song_loglik_per_second_mean": float(rate.mean()),
        "song_loglik_per_second_median": float(np.median(rate)),
        "song_loglik_per_second_min": float(rate.min()),
        "song_loglik_per_second_max": float(rate.max()),
        "posterior_on_label": float(metrics["posterior_on_label"]),
        "posterior_on_label_by_theme": metrics["posterior_on_label_by_theme"].tolist(),
        "theme_expected_seconds": metrics["theme_expected_seconds"].tolist(),
        "theme_labeled_seconds": metrics["theme_labeled_seconds"].tolist(),
        "labeled_runs_per_song": (metrics["labeled_runs"] / n_songs).tolist(),
        "labeled_phrases_per_song": (labeled_phrase_counts() / n_songs).tolist(),
        "slot_occupancy_min": float(metrics["slot_occupancy"].min()),
        "slots_below_one_second": int(np.sum(metrics["slot_occupancy"] < 1.0)),
        "branch_drop_median": float(np.median(sensitivity["branch_drop"])),
        "mean_drop_median": float(np.median(sensitivity["mean_drop"])),
        "repeat_prob": None,
        "start_prob": None,
    }


def _finish(
    model: SoftHMM,
    bundle: dict,
    history: list[float] | np.ndarray,
    complete: float,
    initial_ll: float,
    sequences: list[np.ndarray],
    themes: list[np.ndarray],
    sensitivity: dict[str, np.ndarray],
    n_seconds: int,
) -> None:
    metrics = fit_metrics(sequences, themes, model)
    rate = metrics["song_loglik"] / np.maximum(metrics["song_seconds"], 1.0)
    print(
        f"log-likelihood per second {float(np.sum(metrics['song_loglik'])) / n_seconds:.3f}, "
        f"standard normal {float(metrics['normal_per_second']):.3f}",
        flush=True,
    )
    print(
        f"posterior mass on the labeled theme {float(metrics['posterior_on_label']):.3f}",
        flush=True,
    )
    print(
        f"per-song nats/second mean {rate.mean():.3f}, "
        f"median {np.median(rate):.3f}, min {rate.min():.3f}, max {rate.max():.3f}",
        flush=True,
    )
    _plot(history, metrics, sensitivity)
    save_model(
        model,
        bundle,
        list(np.asarray(history, dtype=np.float64)),
        complete,
        {
            "branch_drop": sensitivity["branch_drop"],
            "mean_drop": sensitivity["mean_drop"],
            "song_loglik": metrics["song_loglik"],
            "song_seconds": metrics["song_seconds"],
            "theme_expected_seconds": metrics["theme_expected_seconds"],
            "theme_labeled_seconds": metrics["theme_labeled_seconds"],
            "posterior_on_label_by_theme": metrics["posterior_on_label_by_theme"],
            "posterior_on_label": metrics["posterior_on_label"],
            "labeled_runs": metrics["labeled_runs"],
            "slot_occupancy": metrics["slot_occupancy"],
            "emission_complete": metrics["emission_complete"],
            "transition_complete": metrics["transition_complete"],
            "initial_loglik": np.array(initial_ll),
        },
    )
    report = _json_report(
        n_seconds,
        len(sequences),
        int(sequences[0].shape[1]),
        history,
        complete,
        initial_ll,
        metrics,
        sensitivity,
    )
    saved_ll = float(metrics["song_loglik"].sum())
    report["loglik"] = saved_ll
    report["loglik_per_second"] = saved_ll / n_seconds
    report["loglik_gain"] = saved_ll - initial_ll
    start_lp = np.log(np.maximum(model.start_prob, 1e-300))
    report["start_complete"] = float(np.dot(metrics["start_count"], start_lp))
    report["repeat_prob"] = model.repeat_prob.tolist()
    report["start_prob"] = model.start_prob.tolist()
    (WORK / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MODEL_PATH}", flush=True)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    n_seconds = n_seconds_in_wav(WAV_PATH)
    print(f"recording {WAV_PATH.name}: {n_seconds} s", flush=True)
    bundle = build_scores(n_seconds)
    scores = np.asarray(bundle["scores"])
    labels = load_slot_labels(CSV_PATH, n_seconds)
    slices = song_slices(labels, n_seconds)
    sequences = [scores[start:end] for _song, start, end in slices]
    print(f"{len(sequences)} songs, motif dimension {scores.shape[1]}", flush=True)

    themes = [labels["theme_per_second"][start:end] for _song, start, end in slices]
    start = initial_model(scores, labels)
    print("scoring the initialization", flush=True)
    initial_ll = total_loglik(sequences, start)
    model, history, complete = baum_welch(sequences, start, max_iter=20, tol=1e-4)
    for index, value in enumerate(history, start=1):
        print(f"iteration {index}: log-likelihood {value:.3f}", flush=True)
    print(f"posterior-weighted likelihood {complete:.3f}", flush=True)
    print("nudging parameters", flush=True)
    sensitivity = parameter_sensitivity(sequences, model)
    print(
        f"parameter sensitivity: median branch drop {np.median(sensitivity['branch_drop']):.3f}, "
        f"median mean drop {np.median(sensitivity['mean_drop']):.3f}",
        flush=True,
    )
    _finish(model, bundle, history, complete, initial_ll, sequences, themes, sensitivity, n_seconds)


def refresh_metrics() -> None:
    """Recompute fit metrics for the saved model. Does not run Baum–Welch again."""
    stored = np.load(MODEL_PATH, allow_pickle=True)
    model, bundle = load_model()
    scores = np.load(WORK / "motif_scores.npz")["scores"]
    labels = load_slot_labels(CSV_PATH, scores.shape[0])
    slices = song_slices(labels, scores.shape[0])
    sequences = [scores[start:end] for _song, start, end in slices]
    themes = [labels["theme_per_second"][start:end] for _song, start, end in slices]
    print("scoring the initialization", flush=True)
    initial_ll = total_loglik(sequences, initial_model(scores, labels))
    sensitivity = {"branch_drop": stored["branch_drop"], "mean_drop": stored["mean_drop"]}
    _finish(
        model,
        bundle,
        stored["loglik_history"],
        float(stored["complete_loglik"]),
        initial_ll,
        sequences,
        themes,
        sensitivity,
        int(scores.shape[0]),
    )


if __name__ == "__main__":
    main()
