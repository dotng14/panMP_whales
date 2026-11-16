"""JS boundary quality and scores vs marks."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from analysis.song_hmm.exp4.baselines import _js_divergence, peaks_from_novelty, token_histogram_novelty
from analysis.song_hmm.exp4.metrics_boundary import boundary_scores
from analysis.song_hmm.exp4.sequence import Session


def js_at_times_motifs(
    sessions: Sequence[Session],
    times: Sequence[float],
    *,
    W: float = 60.0,
    n_motifs: int | None = None,
) -> list[float]:
    events = [t for s in sessions for t in s.tokens if t.kind == "motif" and t.motif is not None]
    if not events:
        return [0.0] * len(times)
    if n_motifs is None:
        n_motifs = max(int(t.motif) for t in events) + 1  # type: ignore[arg-type]
    scores = []
    for mid in times:
        hl = np.zeros(n_motifs)
        hr = np.zeros(n_motifs)
        for t in events:
            if mid - W <= t.t0 < mid:
                hl[int(t.motif)] += 1  # type: ignore[arg-type]
            elif mid <= t.t0 < mid + W:
                hr[int(t.motif)] += 1  # type: ignore[arg-type]
        scores.append(_js_divergence(hl + 1e-6, hr + 1e-6))
    return scores


def js_at_times_features(
    X: NDArray[np.floating],
    feat_times: NDArray[np.floating],
    times: Sequence[float],
    *,
    W: float = 60.0,
    n_bins: int = 8,
) -> list[float]:
    """JS on histogram of first-PC-like scalar (mean over dims) quantized."""
    if len(X) == 0:
        return [0.0] * len(times)
    scalar = X.mean(axis=1)
    # quantize globally
    edges = np.quantile(scalar, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    codes = np.clip(np.digitize(scalar, edges[1:-1], right=False), 0, n_bins - 1)
    scores = []
    for mid in times:
        left = (feat_times >= mid - W) & (feat_times < mid)
        right = (feat_times >= mid) & (feat_times < mid + W)
        hl = np.bincount(codes[left], minlength=n_bins).astype(float) + 1e-6
        hr = np.bincount(codes[right], minlength=n_bins).astype(float) + 1e-6
        scores.append(_js_divergence(hl, hr))
    return scores


def summarize_js(pred_js: Sequence[float], random_js: Sequence[float]) -> dict:
    mean_p = float(np.mean(pred_js)) if pred_js else None
    mean_r = float(np.mean(random_js)) if random_js else None
    out = {
        "n_pred": len(pred_js),
        "n_random": len(random_js),
        "mean_js_pred": mean_p,
        "mean_js_random": mean_r,
        "median_js_pred": float(np.median(pred_js)) if pred_js else None,
        "median_js_random": float(np.median(random_js)) if random_js else None,
    }
    if mean_p is not None and mean_r is not None and mean_r > 0:
        out["mean_js_pred_over_random"] = mean_p / mean_r
        # Rough null: pred is not clearly above random if within ~10% of random mean
        out["looks_like_random"] = bool(mean_p < mean_r * 1.1)
    return out


def random_position_js_null(
    sessions: Sequence[Session],
    *,
    t_lo: float,
    t_hi: float,
    n_times: int,
    n_trials: int = 50,
    seed: int = 0,
    W: float = 60.0,
) -> dict:
    """JS at uniformly random times — null that theme boundaries should beat.

    Returns mean/median over trials of the *mean* JS at ``n_times`` random positions,
    plus one flat draw for direct compare to ``summarize_js``.
    """
    rng = np.random.default_rng(seed)
    trial_means: list[float] = []
    for _ in range(n_trials):
        times = list(rng.uniform(t_lo, t_hi, size=max(n_times, 1)))
        scores = js_at_times_motifs(sessions, times, W=W)
        trial_means.append(float(np.mean(scores)) if scores else 0.0)
    # one matched draw for pairwise summarize_js
    flat_times = list(rng.uniform(t_lo, t_hi, size=max(n_times, 1)))
    flat_js = js_at_times_motifs(sessions, flat_times, W=W)
    return {
        "n_times": n_times,
        "n_trials": n_trials,
        "t_lo": t_lo,
        "t_hi": t_hi,
        "mean_of_trial_means": float(np.mean(trial_means)) if trial_means else None,
        "std_of_trial_means": float(np.std(trial_means)) if trial_means else None,
        "p90_of_trial_means": float(np.percentile(trial_means, 90)) if trial_means else None,
        "flat_random_times": flat_times,
        "flat_random_js": flat_js,
    }


def js_novelty_baseline_motifs(sessions: Sequence[Session], **kwargs) -> list[float]:
    times, nov = token_histogram_novelty(sessions, **kwargs)
    return peaks_from_novelty(times, nov)


def score_boundaries(
    predicted: Sequence[float],
    marked: Sequence[float],
    *,
    hours: float,
) -> dict:
    return boundary_scores(list(predicted), list(marked), hours=hours)
