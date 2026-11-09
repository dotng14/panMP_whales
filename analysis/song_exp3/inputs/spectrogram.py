"""Processing options for Experiment 3 (see pipeline.md).

S: spectrogram resolution        N: noise normalisation
  S1 512-sample window, 0.05 s     N1 rolling median floor, clipped at 0 (Experiment 2)
  S2 256-sample window, 0.025 s    N2 none: one fixed offset per band
  S3 512 below 150 Hz, 256 above,  N3 rolling 10th-percentile floor, clipped at 0
     both every 0.025 s            N4 PCEN (per-band adaptive gain)

All options remove the DC offset first and time-stamp each frame at its centre.
Arrays are ln band power (N1-N3) or PCEN values (N4), shape (frames, bands).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.song_hmm_phrase import features as F

from .pcen import pcen

SR = 2000
S_OPTIONS = {"S1": (512, 100), "S2": (256, 50), "S3": ("multi", 50)}
N_OPTIONS = ("N1", "N2", "N3", "N4")
S3_SPLIT_HZ = 150.0


@dataclass
class Spec:
    values: np.ndarray  # (frames, bands) float32
    fps: int
    edges: np.ndarray  # (bands, 2) Hz
    centre_s: float  # time of frame 0's centre, relative to the stretch start


def dc_removed(audio, chunk_s: int = 3600) -> np.ndarray:
    """Whole recording as float32 with its mean removed (computed in chunks)."""
    n = audio.shape[0]
    total = 0.0
    for i in range(0, n, chunk_s * SR):
        total += float(np.asarray(audio[i : i + chunk_s * SR], np.float64).sum())
    mean = total / n
    out = np.empty(n, np.float32)
    for i in range(0, n, chunk_s * SR):
        out[i : i + chunk_s * SR] = np.asarray(audio[i : i + chunk_s * SR], np.float32) - mean
    return out


def spectrogram(x: np.ndarray, s_code: str, seconds: int) -> Spec:
    """ln band power of a DC-removed stretch ``x`` (2 kHz), ``seconds`` long."""
    nfft, hop = S_OPTIONS[s_code]
    fps = SR // hop
    n = seconds * fps
    if nfft != "multi":
        _, edges = F.band_matrix(SR, nfft)
        return Spec(F.band_spectrogram(x, SR, n, nfft=nfft, hop=hop), fps, edges, nfft / 2 / SR)
    # S3: 512-sample frames for bands below 150 Hz, 256-sample frames above; centres aligned
    _, e512 = F.band_matrix(SR, 512)
    _, e256 = F.band_matrix(SR, 256)
    lo = F.band_spectrogram(x, SR, n, nfft=512, hop=hop)[:, e512[:, 1] <= S3_SPLIT_HZ]
    shifted = np.concatenate([x[128:], np.zeros(128, x.dtype)])  # 256-frame centre = 512-frame centre
    hi = F.band_spectrogram(shifted, SR, n, nfft=256, hop=hop)[:, e256[:, 0] >= S3_SPLIT_HZ]
    edges = np.vstack([e512[e512[:, 1] <= S3_SPLIT_HZ], e256[e256[:, 0] >= S3_SPLIT_HZ]])
    return Spec(np.hstack([lo, hi]).astype(np.float32), fps, edges, 256 / SR)


def normalise(s: Spec, n_code: str) -> np.ndarray:
    """Apply a noise normalisation to one half (statistics never cross the split)."""
    v = s.values
    if n_code == "N1":
        return F.whiten(v, s.fps)
    if n_code == "N2":
        return (v - np.median(v, axis=0)).astype(np.float32)
    if n_code == "N3":
        return F.whiten(v, s.fps, pct=10.0)
    if n_code == "N4":
        energy = np.exp(v.astype(np.float64))
        energy /= np.median(energy, axis=0)  # typical level 1 in every band, so PCEN's offset means the same everywhere
        return pcen(energy, s.fps).astype(np.float32)
    raise ValueError(n_code)
