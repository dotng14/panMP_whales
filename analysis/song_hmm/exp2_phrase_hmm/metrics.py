"""Metrics shared by Experiment 1, Experiment 2 and the CSV. Definitions: metrics.md.

B  agreement with the CSV (a reference, not ground truth)
C  label-free structure, measured on the audio's phrase-window descriptors
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from analysis.song_hmm.report import match_events

from .decode import WINDOW_CENTRE_S, Decoded
from .topology import N_THEMES

BOUNDARY_TOLS = (15, 30, 60)
EDGE_GUARD_S = 15
MIN_LAG_WINDOWS = 6  # pairs closer than 12 s share audio; left out of coherence
NOVELTY_HALF = 15  # windows each side (30 s)
NOVELTY_TOL_WINDOWS = 5  # +-10 s


# ---------------------------------------------------------------- B: CSV agreement


def kappa(conf: np.ndarray) -> float:
    n = conf.sum()
    po = np.trace(conf) / n
    pe = (conf.sum(0) * conf.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def csv_agreement(dec: Decoded, csv_theme: np.ndarray, csv_starts: np.ndarray, majority: int) -> dict:
    n = len(dec.theme)
    lab = csv_theme >= 0
    y, p = csv_theme[lab], dec.theme[lab]
    conf = np.zeros((N_THEMES, N_THEMES + 1), dtype=np.int64)  # last column: background
    np.add.at(conf, (y, np.where(p >= 0, p, N_THEMES)), 1)
    recall = np.diag(conf[:, :N_THEMES]) / np.maximum(conf.sum(1), 1)
    ref = csv_starts[(csv_starts >= EDGE_GUARD_S) & (csv_starts < n - EDGE_GUARD_S)]
    complete = [s for s in dec.songs if s.complete]
    return {
        "theme_accuracy": float(np.mean(p == y)),
        "theme_balanced_accuracy": float(recall.mean()),
        "theme_kappa": kappa(np.vstack([conf, np.zeros((1, N_THEMES + 1))])),  # square: background as a 7th class
        "theme_recall": recall.tolist(),
        "majority_baseline": float(np.mean(y == majority)),
        "confusion": conf.tolist(),
        "boundaries": {f"tol_{t}s": match_events(dec.boundaries, ref, t) for t in BOUNDARY_TOLS},
        "n_songs": len(dec.songs),
        "n_complete_songs": len(complete),
        "complete_song_lengths_s": [s.end_s - s.start_s for s in complete],
        "session_fraction": float(dec.session.mean()),
        "n_sessions": len(dec.sessions),
    }


# ---------------------------------------------------------------- C: label-free


def unit_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.mean(0)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)


def window_values(per_second: np.ndarray, t_start: np.ndarray, offset: int) -> np.ndarray:
    sec = np.clip(np.floor(t_start - offset + WINDOW_CENTRE_S).astype(int), 0, len(per_second) - 1)
    return per_second[sec]


def _runs(lab: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, end, label) runs of equal labels."""
    cut = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1], True])
    return [(int(a), int(b), int(lab[a])) for a, b in zip(cut[:-1], cut[1:], strict=True)]


def _mean_sim(a: np.ndarray, b: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> tuple[float, int]:
    k = a @ b.T
    far = np.abs(ia[:, None] - ib[None, :]) >= MIN_LAG_WINDOWS
    return float(k[far].sum()), int(far.sum())


def theme_coherence(zn: np.ndarray, wtheme: np.ndarray) -> dict:
    """Mean cosine within decoded theme segments minus between adjacent segments (pairs >= 12 s apart)."""
    segs = [r for r in _runs(wtheme) if r[2] >= 0]
    w_sum = w_n = b_sum = b_n = 0.0
    idx = np.arange(len(zn))
    for j, (a, b, _) in enumerate(segs):
        s, n = _mean_sim(zn[a:b], zn[a:b], idx[a:b], idx[a:b])
        w_sum, w_n = w_sum + s, w_n + n
        if j + 1 < len(segs):
            a2, b2, _ = segs[j + 1]
            s, n = _mean_sim(zn[a:b], zn[a2:b2], idx[a:b], idx[a2:b2])
            b_sum, b_n = b_sum + s, b_n + n
    within = w_sum / max(w_n, 1)
    between = b_sum / max(b_n, 1)
    return {"within": within, "between_adjacent": between, "coherence": within - between, "n_segments": len(segs)}


def theme_identity(zn: np.ndarray, wtheme: np.ndarray, song_windows: list[tuple[int, int]]) -> dict:
    """Same theme in consecutive songs vs different themes in consecutive songs (cosine of theme means)."""
    means = []
    for a, b in song_windows:
        m = np.full((N_THEMES, zn.shape[1]), np.nan)
        for k in range(N_THEMES):
            rows = zn[a:b][wtheme[a:b] == k]
            if len(rows) >= 3:
                v = rows.mean(0)
                m[k] = v / max(np.linalg.norm(v), 1e-9)
        means.append(m)
    same, diff = [], []
    for m1, m2 in zip(means[:-1], means[1:], strict=False):
        sim = m1 @ m2.T
        for i in range(N_THEMES):
            for j in range(N_THEMES):
                if np.isfinite(sim[i, j]):
                    (same if i == j else diff).append(sim[i, j])
    s, d = (float(np.mean(v)) if v else float("nan") for v in (same, diff))
    return {"same_theme": s, "other_theme": d, "identity": s - d, "n_song_pairs": max(len(means) - 1, 0)}


def novelty_curve(zn: np.ndarray, half: int = NOVELTY_HALF) -> np.ndarray:
    """Checkerboard-kernel novelty along the diagonal of the cosine self-similarity matrix."""
    g = np.exp(-0.5 * (np.arange(-half, half) + 0.5) ** 2 / (half / 2) ** 2)
    sign = np.sign(np.arange(-half, half) + 0.5)
    kernel = np.outer(g * sign, g * sign)
    out = np.zeros(len(zn))
    for i in range(half, len(zn) - half):
        blk = zn[i - half : i + half]
        out[i] = np.sum(kernel * (blk @ blk.T))
    return out


def novelty_peaks(curve: np.ndarray) -> np.ndarray:
    peaks, _ = find_peaks(curve, distance=NOVELTY_TOL_WINDOWS * 2, prominence=0.5 * curve.std())
    return peaks


def boundary_novelty(wtheme: np.ndarray, peaks: np.ndarray) -> dict:
    change = np.flatnonzero((wtheme[1:] != wtheme[:-1]) & (wtheme[1:] >= 0) & (wtheme[:-1] >= 0)) + 1
    return match_events(change, peaks, NOVELTY_TOL_WINDOWS) | {"n_changes": int(len(change)), "n_peaks": int(len(peaks))}


def song_regularity(songs: list, train_songs: list) -> dict:
    lengths = np.array([s.end_s - s.start_s for s in songs if s.complete], dtype=float)
    tr = np.array([s.theme_seconds for s in train_songs if s.complete], dtype=float)
    te = np.array([s.theme_seconds for s in songs if s.complete], dtype=float)
    inside = float("nan")
    if len(tr) >= 4 and len(te):
        lo, hi = np.percentile(tr, 25, axis=0), np.percentile(tr, 75, axis=0)
        inside = float(np.mean(np.all((te >= lo) & (te <= hi), axis=1)))
    return {
        "n_complete": int(len(lengths)),
        "length_cv": float(lengths.std() / lengths.mean()) if len(lengths) > 1 else float("nan"),
        "median_length_s": float(np.median(lengths)) if len(lengths) else float("nan"),
        "all_themes_in_train_iqr": inside,
    }


def label_free(dec: Decoded, zn: np.ndarray, t_start: np.ndarray, offset: int, peaks: np.ndarray, train_dec: Decoded) -> dict:
    wtheme = window_values(dec.theme, t_start, offset)
    centres = t_start - offset + WINDOW_CENTRE_S
    song_windows = [(int(np.searchsorted(centres, s.start_s)), int(np.searchsorted(centres, s.end_s))) for s in dec.songs]
    runs = [b - a for a, b, lab in _runs(dec.theme) if lab >= 0]
    return {
        "longest_theme_run_s": int(max(runs)) if runs else 0,
        "coherence": theme_coherence(zn, wtheme),
        "identity": theme_identity(zn, wtheme, song_windows),
        "novelty": boundary_novelty(wtheme, peaks),
        "regularity": song_regularity(dec.songs, train_dec.songs),
    }
