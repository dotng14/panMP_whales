"""Place the 35-motif dictionary on whitened windows of a 10 kHz recording.

The rule is the one in `02_place_dictionary.md`, applied to the dictionary
rows that a 10 kHz file can represent (below 4.5 kHz). Components are made
orthonormal again after the rows are cut. One pursuit covers 30 windows
(37.5 minutes), the length of the recordings the dictionary was scored on.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np

import spectro

DICTIONARY = Path(
    "/home/yfreund/whale_processing/result/LeiData/"
    "SOCAL_H_76_230427_014845.x.limited_overlap_k1_w40_pc5_tau04_m8x32_refine2"
)
N_ROWS = 736  # dictionary rows 0..735, 12 Hz to 4.5 kHz
K = 1
EPS = 0.01
TAU = 0.4
CHUNK_WINDOWS = 30
NFFT = 1024


@dataclass
class Dictionary:
    ids: list[int]
    widths: np.ndarray
    mean_energy: cp.ndarray  # (n_motifs,)
    mean_coeff: cp.ndarray  # (n_motifs, n_components)
    spectra: cp.ndarray  # (NFFT/2+1, n_motifs * (1 + n_components), N_ROWS), conjugated
    n_components: int


def load_dictionary(n_rows: int = N_ROWS) -> Dictionary:
    ids = sorted(int(Path(p).stem.split("_")[1]) for p in glob.glob(str(DICTIONARY / "atoms" / "template_*.npy")))
    widths, mean_energy, mean_coeff, stack = [], [], [], []
    n_components = 0
    for motif_id in ids:
        components = np.load(DICTIONARY / "atoms" / f"template_{motif_id:03d}.npy")[:, :n_rows, :].astype(np.float64)
        mean = np.load(DICTIONARY / "atoms" / f"mean_{motif_id:03d}.npy")[:n_rows, :].astype(np.float64)
        n_components, _rows, width = components.shape
        flat, _ = np.linalg.qr(components.reshape(n_components, -1).T)
        basis = flat.T.reshape(n_components, n_rows, width)
        widths.append(width)
        mean_energy.append(float(np.sum(mean * mean)))
        mean_coeff.append([float(np.sum(mean * b)) for b in basis])
        for template in (mean, *basis):
            padded = np.zeros((n_rows, NFFT))
            padded[:, :width] = template
            stack.append(np.conj(np.fft.rfft(padded, axis=1)))
    spectra = np.transpose(np.stack(stack), (2, 0, 1))  # (omega, template, row)
    return Dictionary(
        ids=ids,
        widths=np.array(widths),
        mean_energy=cp.asarray(mean_energy),
        mean_coeff=cp.asarray(mean_coeff),
        spectra=cp.asarray(spectra),
        n_components=n_components,
    )


def explain(window: np.ndarray, dictionary: Dictionary) -> tuple[np.ndarray, np.ndarray]:
    """Explained fraction and explained energy, shape (n_motifs, n_time). Invalid starts are -inf."""
    spec = cp.asarray(window)
    n_time = spec.shape[1]
    transform = cp.fft.rfft(spec, n=NFFT, axis=1)  # (row, omega)
    product = cp.matmul(dictionary.spectra, transform.T[:, :, None])[:, :, 0]  # (omega, template)
    dots = cp.fft.irfft(product.T, n=NFFT, axis=1)[:, :n_time]
    per = 1 + dictionary.n_components
    dots = dots.reshape(len(dictionary.ids), per, n_time)
    column = cp.concatenate([cp.zeros(1), cp.cumsum(cp.sum(spec * spec, axis=0))])
    fraction = cp.full((len(dictionary.ids), n_time), -cp.inf)
    explained = cp.zeros((len(dictionary.ids), n_time))
    for index, width in enumerate(dictionary.widths):
        n_valid = n_time - int(width) + 1
        energy = column[int(width) : int(width) + n_valid] - column[:n_valid]
        centered = energy + dictionary.mean_energy[index] - 2.0 * dots[index, 0, :n_valid]
        projected = cp.sum((dots[index, 1:, :n_valid] - dictionary.mean_coeff[index][:, None]) ** 2, axis=0)
        residual = cp.maximum(centered - projected, 0.0)
        gain = energy - residual
        explained[index, :n_valid] = gain
        fraction[index, :n_valid] = cp.where(energy > 0.0, gain / cp.maximum(energy, 1e-300), -cp.inf)
    return cp.asnumpy(fraction), cp.asnumpy(explained)


def greedy(fraction: np.ndarray, explained: np.ndarray, widths: np.ndarray) -> list[tuple[float, float, int, int]]:
    """Placements of one window in the order pursuit accepts them: (fraction, energy, motif index, start)."""
    score = fraction.copy()
    n_time = score.shape[1]
    out = []
    while True:
        flat = int(np.argmax(score))
        motif, start = divmod(flat, n_time)
        best = score[motif, start]
        if not np.isfinite(best):
            return out
        out.append((float(best), float(explained[motif, start]), motif, start))
        end = start + int(widths[motif])
        for other, width in enumerate(widths):
            score[other, max(0, start - int(width) + 1) : end] = -np.inf


def place_chunk(windows: list[np.ndarray], dictionary: Dictionary) -> list[tuple[int, int, int, float]]:
    """Kept placements of one pursuit: (window index, motif id, start column, unexplained)."""
    candidates = []
    for index, window in enumerate(windows):
        fraction, explained = explain(window, dictionary)
        for score, energy, motif, start in greedy(fraction, explained, dictionary.widths):
            candidates.append((score, energy, index, motif, start))
    candidates.sort(key=lambda row: -row[0])
    kept = []
    first_energy = None
    for score, energy, index, motif, start in candidates:
        if first_energy is None:
            first_energy = energy
        elif energy < EPS * first_energy:
            break
        if 1.0 - score <= TAU:
            kept.append((index, dictionary.ids[motif], start, 1.0 - score))
    return kept


def place_span(wav: Path, start_sec: float, n_windows: int, dictionary: Dictionary):
    """Rows (motif id, start seconds from the file start, end seconds, unexplained)."""
    rows = []
    for first in range(0, n_windows, CHUNK_WINDOWS):
        count = min(CHUNK_WINDOWS, n_windows - first)
        chunk_start = start_sec + first * spectro.WINDOW_SEC
        audio = spectro.read_seconds(wav, chunk_start, count * spectro.WINDOW_SEC)
        size = spectro.WINDOW_SEC * spectro.SOURCE_RATE
        windows = [
            spectro.window_spectrogram(audio[i * size : (i + 1) * size], N_ROWS)
            for i in range(audio.size // size)
        ]
        for index, motif_id, column, unexplained in place_chunk(windows, dictionary):
            begin = chunk_start + index * spectro.WINDOW_SEC + column * spectro.HOP_SEC
            width = int(dictionary.widths[dictionary.ids.index(motif_id)])
            rows.append((motif_id, begin, begin + width * spectro.HOP_SEC, unexplained))
    rows.sort(key=lambda row: row[1])
    return rows
