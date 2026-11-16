"""Baselines B1 (token-histogram novelty) and B2 (phrase d1 threshold)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .phrases import PhraseFeatures, cut_phrases, phrase_feature_rows
from .sequence import Session, Token


def _js_divergence(p: NDArray[np.floating], q: NDArray[np.floating]) -> float:
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def token_histogram_novelty(
    sessions: Sequence[Session],
    *,
    win_s: float = 60.0,
    step_s: float = 5.0,
    n_motifs: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return (times, novelty) for B1 over concatenated motif tokens."""
    events: list[Token] = []
    for s in sessions:
        events.extend(t for t in s.tokens if t.kind == "motif" and t.motif is not None)
    if not events:
        return np.asarray([]), np.asarray([])
    if n_motifs is None:
        n_motifs = max(int(t.motif) for t in events) + 1  # type: ignore[arg-type]
    t0 = events[0].t0
    t1 = events[-1].t1
    times = np.arange(t0 + win_s, t1 - win_s + 1e-9, step_s)
    nov = np.zeros(len(times))
    for i, mid in enumerate(times):
        left = [t for t in events if mid - win_s <= t.t0 < mid]
        right = [t for t in events if mid <= t.t0 < mid + win_s]
        hl = np.zeros(n_motifs)
        hr = np.zeros(n_motifs)
        for t in left:
            hl[int(t.motif)] += 1  # type: ignore[arg-type]
        for t in right:
            hr[int(t.motif)] += 1  # type: ignore[arg-type]
        nov[i] = _js_divergence(hl + 1e-6, hr + 1e-6)
    return times, nov


def peaks_from_novelty(
    times: NDArray[np.floating],
    novelty: NDArray[np.floating],
    *,
    threshold: float | None = None,
    min_sep_s: float = 60.0,
) -> list[float]:
    if len(times) == 0:
        return []
    if threshold is None:
        threshold = float(np.quantile(novelty, 0.85))
    order = np.argsort(-novelty)
    chosen: list[float] = []
    for i in order:
        if novelty[i] < threshold:
            break
        t = float(times[i])
        if all(abs(t - c) >= min_sep_s for c in chosen):
            chosen.append(t)
    return sorted(chosen)


def b1_boundaries(sessions: Sequence[Session], **kwargs) -> list[float]:
    times, nov = token_histogram_novelty(sessions)
    return peaks_from_novelty(times, nov, **kwargs)


def b2_boundaries(
    sessions: Sequence[Session],
    *,
    threshold: float = 0.5,
) -> list[float]:
    bounds: list[float] = []
    for s in sessions:
        phrases = cut_phrases(s)
        feats = phrase_feature_rows(phrases)
        for ph, fe in zip(phrases, feats):
            if fe.session_start:
                continue
            if fe.d1 >= threshold:
                bounds.append(ph.t0)
    return bounds
