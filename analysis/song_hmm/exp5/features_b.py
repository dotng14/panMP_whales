"""Approach B: continuous features from GPL spectrograms (train stats frozen)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from analysis.dict_learning.code.cli import load_windows


@dataclass
class FeaturePack:
    X: NDArray[np.floating]  # (N, D)
    times: NDArray[np.floating]  # (N,) center times in seconds
    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    hop_s: float
    n_bands: int


def _band_edges(n_freq: int, n_bands: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n_freq, n_bands + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_bands)]


def window_features(
    spec: NDArray[np.floating],
    *,
    t_start: float,
    sec_per_bin: float,
    hop_bins: int,
    n_bands: int = 16,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Log-band energy + centroid/bandwidth summaries every hop_bins columns."""
    F, T = spec.shape
    bands = _band_edges(F, n_bands)
    rows: list[NDArray[np.floating]] = []
    times: list[float] = []
    for c0 in range(0, max(T - hop_bins, 1), hop_bins):
        c1 = min(T, c0 + hop_bins)
        tile = spec[:, c0:c1]
        tile = np.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0)
        energy = np.maximum(tile, 1e-8)
        log_e = np.log(energy)
        feats = []
        for a, b in bands:
            if b <= a:
                feats.append(0.0)
            else:
                feats.append(float(log_e[a:b].mean()))
        # spectral centroid / bandwidth in bin units
        col_sum = energy.sum(axis=0) + 1e-8
        freqs = np.arange(F, dtype=float)[:, None]
        centroid = float(((freqs * energy).sum(axis=0) / col_sum).mean())
        bw = float(np.sqrt((((freqs - centroid) ** 2) * energy).sum(axis=0) / col_sum).mean())
        feats.extend([centroid / max(F, 1), bw / max(F, 1), float(np.log(col_sum.mean()))])
        rows.append(np.asarray(feats, dtype=float))
        times.append(t_start + 0.5 * (c0 + c1) * sec_per_bin)
    if not rows:
        return np.zeros((0, n_bands + 3)), np.zeros(0)
    return np.vstack(rows), np.asarray(times, dtype=float)


def build_features(
    gpl_dir: Path,
    *,
    window_offset: int,
    n_windows: int,
    hop_s: float = 1.0,
    n_bands: int = 16,
    mean: NDArray[np.floating] | None = None,
    std: NDArray[np.floating] | None = None,
) -> FeaturePack:
    windows = load_windows(gpl_dir, n_windows, window_offset)
    if not windows:
        raise ValueError("no GPL windows")
    sec_per_bin = float(windows[0].overlap) / float(windows[0].sample_freq)
    hop_bins = max(1, int(round(hop_s / sec_per_bin)))
    Xs: list[NDArray[np.floating]] = []
    Ts: list[NDArray[np.floating]] = []
    for w in windows:
        t_start = (int(w.start_sample) - 1) / float(w.sample_freq)
        x, t = window_features(
            np.asarray(w.data, dtype=float),
            t_start=t_start,
            sec_per_bin=sec_per_bin,
            hop_bins=hop_bins,
            n_bands=n_bands,
        )
        if len(x):
            Xs.append(x)
            Ts.append(t)
    X = np.vstack(Xs) if Xs else np.zeros((0, n_bands + 3))
    times = np.concatenate(Ts) if Ts else np.zeros(0)
    if mean is None:
        mean = X.mean(axis=0) if len(X) else np.zeros(X.shape[1] if X.ndim == 2 else 0)
    if std is None:
        std = X.std(axis=0) if len(X) else np.ones_like(mean)
        std = np.maximum(std, 1e-6)
    Xn = (X - mean) / std if len(X) else X
    return FeaturePack(X=Xn, times=times, mean=mean, std=std, hop_s=hop_s, n_bands=n_bands)


def save_features(path: Path, pack: FeaturePack, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        X=pack.X,
        times=pack.times,
        mean=pack.mean,
        std=pack.std,
        hop_s=np.asarray([pack.hop_s]),
        n_bands=np.asarray([pack.n_bands]),
    )
    Path(str(path) + ".meta.json").write_text(json.dumps(meta or {}, indent=2))


def load_features(path: Path) -> tuple[FeaturePack, dict]:
    path = Path(path)
    data = np.load(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text()) if Path(str(path) + ".meta.json").is_file() else {}
    pack = FeaturePack(
        X=data["X"],
        times=data["times"],
        mean=data["mean"],
        std=data["std"],
        hop_s=float(data["hop_s"][0]),
        n_bands=int(data["n_bands"][0]),
    )
    return pack, meta
