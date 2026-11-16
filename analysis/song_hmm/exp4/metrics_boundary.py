"""Boundary scores and simple input-step checks."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def match_boundaries(
    predicted: Sequence[float],
    marked: Sequence[float],
    tau: float,
) -> tuple[int, int, int, list[float]]:
    """Greedy nearest matching within ±tau.

    Returns hits, n_pred, n_marks, errors_for_hits.
    """
    preds = sorted(float(x) for x in predicted)
    marks = sorted(float(x) for x in marked)
    used = set()
    hits = 0
    errors: list[float] = []
    for p in preds:
        best_j, best_d = None, None
        for j, m in enumerate(marks):
            if j in used:
                continue
            d = abs(p - m)
            if d <= tau and (best_d is None or d < best_d):
                best_j, best_d = j, d
        if best_j is not None and best_d is not None:
            used.add(best_j)
            hits += 1
            errors.append(best_d)
    return hits, len(preds), len(marks), errors


def boundary_scores(
    predicted: Sequence[float],
    marked: Sequence[float],
    *,
    hours: float | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for tau in (5.0, 10.0, 30.0):
        hits, n_pred, n_marks, errors = match_boundaries(predicted, marked, tau)
        recall = hits / n_marks if n_marks else 0.0
        precision = hits / n_pred if n_pred else 0.0
        out[f"recall@{int(tau)}s"] = recall
        out[f"precision@{int(tau)}s"] = precision
        if tau == 10.0:
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            out["f1@10s"] = f1
        if tau == 30.0 and errors:
            out["median_err_s"] = float(np.median(errors))
            out["p90_err_s"] = float(np.quantile(errors, 0.9))
    hits30, n_pred, n_marks, _ = match_boundaries(predicted, marked, 30.0)
    false_alarms = max(n_pred - hits30, 0)
    if hours is None:
        if marked:
            hours = max((max(marked) - min(marked)) / 3600.0, 1e-6)
        else:
            hours = 1.0
    out["false_alarms_per_h"] = false_alarms / max(hours, 1e-6)
    out["n_pred"] = float(n_pred)
    out["n_marks"] = float(n_marks)
    return out


def agreement(a: Sequence[float], b: Sequence[float], tau: float = 5.0) -> float:
    """Symmetric boundary agreement (T3)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ab, _, na, _ = match_boundaries(a, b, tau)
    ba, _, nb, _ = match_boundaries(b, a, tau)
    return 0.5 * (ab / max(na, 1) + ba / max(nb, 1))
