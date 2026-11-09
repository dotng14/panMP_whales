"""Coverage, over-coverage, and per-motif specificity of a dictionary."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.fft import irfft, rfft
from scipy.fft import next_fast_len

from analysis.dict_learning.code.core import Matching, coverage_from_matchings
from analysis.dict_learning.code.io import discover_windows

DEFAULT_DICTIONARY_ID = 276
DEFAULT_WAV = Path("/home/yfreund/whale_processing/LeiData/SOCAL_H_76_230427_014845.x.wav")
DEFAULT_CATALOG = Path("/home/yfreund/whale_processing/result/dict_learning_catalog.sqlite")

_WINDOW = "spectrogram"


def _connect(catalog: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)


def _whitened_dir(catalog: Path, wav: Path) -> Path:
    """GPL output directory for this wave file."""
    wanted = wav.expanduser().resolve()
    conn = _connect(catalog)
    try:
        rows = conn.execute(
            """
            SELECT w.abs_path, g.npy_dir
            FROM wav_files w
            JOIN gpl_segmentations g ON g.wav_id = w.wav_id
            """
        ).fetchall()
    finally:
        conn.close()
    for abs_path, npy_dir in rows:
        if Path(abs_path).expanduser().resolve() == wanted:
            return Path(npy_dir)
    raise FileNotFoundError(f"no GPL whitened spectrogram is registered for {wanted}")


def _stitch(npy_dir: Path) -> tuple[np.ndarray, dict[str, int]]:
    windows = discover_windows(npy_dir)
    windows.sort(key=lambda window: (window.start_sample, window.window_id))
    if not windows:
        raise FileNotFoundError(f"no whitened spectrogram windows in {npy_dir}")
    offset: dict[str, int] = {}
    column = 0
    parts: list[np.ndarray] = []
    for window in windows:
        offset[window.window_id] = column
        data = np.asarray(window.data, dtype=np.float64)
        parts.append(data)
        column += int(data.shape[1])
    return np.concatenate(parts, axis=1), offset


def _dictionary_rows(catalog: Path, dictionary_id: int) -> list[tuple[int, int, str]]:
    conn = _connect(catalog)
    try:
        rows = conn.execute(
            """
            SELECT t.template_id, t.width, t.npy_path
            FROM dictionary_templates dt
            JOIN templates t ON t.template_pk = dt.template_pk
            WHERE dt.dictionary_id = ?
            ORDER BY dt.position
            """,
            (int(dictionary_id),),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise KeyError(f"dictionary {dictionary_id} has no motifs")
    return [(int(template_id), int(width), str(path)) for template_id, width, path in rows]


def _placements(
    catalog: Path,
    dictionary_id: int,
    offset: Mapping[str, int],
) -> list[tuple[int, int, int]]:
    conn = _connect(catalog)
    try:
        rows = conn.execute(
            """
            SELECT m.window_id, m.t_start, t.template_id, t.width
            FROM matchings m
            JOIN templates t ON t.template_pk = m.template_pk
            WHERE m.dictionary_id = ?
            """,
            (int(dictionary_id),),
        ).fetchall()
    finally:
        conn.close()
    placements: list[tuple[int, int, int]] = []
    for window_id, t_start, template_id, width in rows:
        if window_id not in offset:
            continue
        placements.append((int(template_id), offset[window_id] + int(t_start), int(width)))
    return placements


def _dot_products(spectrogram: np.ndarray, motif: np.ndarray) -> np.ndarray:
    """Dot product of ``motif`` with each in-range start column. Other columns are NaN."""
    frequency, n_time = spectrogram.shape
    if motif.shape[0] != frequency:
        raise ValueError(
            f"motif has {motif.shape[0]} frequency rows, spectrogram has {frequency}"
        )
    width = int(motif.shape[1])
    dots = np.full(n_time, np.nan, dtype=np.float64)
    if n_time < width:
        return dots
    n_valid = n_time - width + 1
    nfft = next_fast_len(n_time + width - 1)
    full = irfft(
        rfft(spectrogram, n=nfft, axis=1) * rfft(np.ascontiguousarray(motif[:, ::-1]), n=nfft, axis=1),
        n=nfft,
        axis=1,
    )
    dots[:n_valid] = full[:, width - 1 : width - 1 + n_valid].sum(axis=0)
    return dots


def _local_maxima(dots: np.ndarray) -> np.ndarray:
    """Columns strictly larger than both neighbors. Ends are not maxima."""
    values = np.asarray(dots, dtype=np.float64)
    if values.size < 3:
        return np.zeros(0, dtype=np.int64)
    middle = values[1:-1]
    keep = (
        np.isfinite(middle)
        & np.isfinite(values[:-2])
        & np.isfinite(values[2:])
        & (middle > values[:-2])
        & (middle > values[2:])
    )
    return np.flatnonzero(keep).astype(np.int64) + 1


def _centered_convolve(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve and keep the length of ``signal``. ``kernel`` has odd length."""
    full = np.convolve(signal, kernel, mode="full")
    start = (kernel.size - 1) // 2
    return full[start : start + signal.size]


def _gaussian_smooth(dots: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve ``dots`` with a Gaussian of standard deviation ``sigma``.

    Missing samples are left out of the kernel and the remaining weights are
    renormalized, so a NaN does not spread into the finite columns.
    """
    values = np.asarray(dots, dtype=np.float64)
    if sigma < 0.0:
        raise ValueError(f"k must be >= 0, got {sigma}")
    if sigma == 0.0:
        return values.copy()
    finite = np.isfinite(values)
    radius = max(1, int(np.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    numer = _centered_convolve(np.where(finite, values, 0.0), kernel)
    denom = _centered_convolve(finite.astype(np.float64), kernel)
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    good = finite & (denom > 0.0)
    smoothed[good] = numer[good] / denom[good]
    return smoothed


def _peak_ratios(dots: np.ndarray, sigma: float) -> np.ndarray:
    """``dot(t*) / smoothed(t*)`` at each local maximum."""
    smoothed = _gaussian_smooth(dots, sigma)
    peaks = _local_maxima(dots)
    ratios = np.full(peaks.shape, np.nan, dtype=np.float64)
    if peaks.size == 0:
        return ratios
    original = dots[peaks]
    baseline = smoothed[peaks]
    valid = np.isfinite(original) & np.isfinite(baseline) & (baseline != 0.0)
    ratios[valid] = original[valid] / baseline[valid]
    return ratios


def _filter_ratios(ratios: np.ndarray, m: float) -> tuple[np.ndarray, float]:
    """Drop ratios smaller than the largest ratio divided by ``m``.

    The largest ratio is kept even when it is negative, because dividing a
    negative maximum by ``m > 1`` would otherwise put that maximum below the cutoff.
    """
    if m <= 0.0:
        raise ValueError(f"m must be positive, got {m}")
    finite = np.asarray(ratios, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return finite, float("nan")
    largest = float(np.max(finite))
    kept = finite[finite >= largest / float(m)]
    if kept.size == 0 or not np.any(kept == largest):
        kept = np.append(kept, largest)
    return kept, float(np.median(kept))


def _sigma(
    motif_id: int,
    width: int,
    k: int | float | Mapping[int, int | float] | None,
) -> float:
    if k is None:
        return float(width // 2)
    if isinstance(k, Mapping):
        return float(k[motif_id])
    return float(k)


def _measures(
    placements: Sequence[tuple[int, int, int]],
    widths: Mapping[int, int],
    dots_by_motif: Sequence[tuple[int, np.ndarray]],
    n_time: int,
    k: int | float | Mapping[int, int | float] | None,
    m: float,
) -> dict[str, object]:
    matchings = [
        Matching(template_id=motif_id, window_id=_WINDOW, t_start=start, amplitude=1.0)
        for motif_id, start, _width in placements
    ]
    stack = {_WINDOW: np.zeros((1, n_time), dtype=np.float32)}
    _windows_hit, _windows, n_covered, n_total = coverage_from_matchings(
        stack, matchings, template_widths=dict(widths)
    )
    placed = 0
    for _motif_id, start, width in placements:
        lo = max(0, int(start))
        hi = min(n_time, int(start) + int(width))
        if hi > lo:
            placed += hi - lo
    specificity: dict[int, np.ndarray] = {}
    specificity_median: dict[int, float] = {}
    sigmas: dict[int, float] = {}
    for motif_id, dots in dots_by_motif:
        sigma = _sigma(motif_id, int(widths[motif_id]), k)
        sigmas[motif_id] = sigma
        kept, median = _filter_ratios(_peak_ratios(dots, sigma), m)
        specificity[motif_id] = kept
        specificity_median[motif_id] = median
    return {
        "coverage": (n_covered / n_total) if n_total else 0.0,
        "over_coverage": (placed / n_covered) if n_covered else float("nan"),
        "n_bins_covered": int(n_covered),
        "n_bins_total": int(n_total),
        "placed_length": int(placed),
        "k": sigmas,
        "m": float(m),
        "specificity": specificity,
        "specificity_median": specificity_median,
    }


def evaluate_dictionary(
    dictionary_id: int = DEFAULT_DICTIONARY_ID,
    wav: Path | str = DEFAULT_WAV,
    k: int | float | Mapping[int, int | float] | None = None,
    m: float = 2.0,
    *,
    catalog: Path | str = DEFAULT_CATALOG,
) -> dict[str, object]:
    """Evaluate ``dictionary_id`` on the GPL whitened spectrogram of ``wav``.

    ``k`` is the standard deviation, in spectrogram columns, of the Gaussian
    applied to each motif's dot product. ``None`` uses half the width of each
    motif. At a local maximum ``t*`` the specificity ratio is the original dot
    product divided by the smoothed dot product. ``m`` removes ratios smaller
    than the largest ratio divided by ``m``, and the largest ratio is kept.
    Coverage is the fraction of columns covered by at least one kept placement.
    Over-coverage is the total placed length divided by the covered length.
    The specificity summary is the median of the filtered ratios.
    """
    catalog = Path(catalog)
    wav = Path(wav)
    spectrogram, offset = _stitch(_whitened_dir(catalog, wav))
    n_time = int(spectrogram.shape[1])
    rows = _dictionary_rows(catalog, dictionary_id)
    widths = {motif_id: width for motif_id, width, _path in rows}
    placements = _placements(catalog, dictionary_id, offset)
    dots_by_motif = [
        (motif_id, _dot_products(spectrogram, np.load(path)))
        for motif_id, _width, path in rows
    ]
    result = _measures(placements, widths, dots_by_motif, n_time, k, m)
    result["dictionary_id"] = int(dictionary_id)
    result["wav"] = str(wav.expanduser().resolve())
    return result
