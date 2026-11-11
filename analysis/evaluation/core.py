"""Coverage, residuals, gaps, and per-motif specificity of a dictionary."""

from __future__ import annotations

import sqlite3
from collections import Counter
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


def _load_basis(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Component stack ``(k, F, W)`` and the sibling mean when it matches.

    A rank-1 file is returned as one component and no mean.
    """
    motif = np.load(path)
    if motif.ndim == 2:
        return motif.reshape(1, *motif.shape), None
    if motif.ndim != 3:
        raise ValueError(f"motif must be (F, W) or (k, F, W), got shape {motif.shape}")
    mean_path = Path(path).with_name("mean_" + Path(path).name.removeprefix("template_"))
    if mean_path.is_file():
        mean = np.load(mean_path)
        if mean.ndim == 2 and mean.shape == motif.shape[1:]:
            return motif, mean
    return motif, None


def _load_motif(path: str) -> np.ndarray:
    """Rank-1 image used for the dot product.

    A ``(k, F, W)`` atom uses the sibling ``mean_*.npy`` when that file matches
    the component shape. Otherwise the first component is used.
    """
    basis, mean = _load_basis(path)
    if mean is not None:
        return mean
    return np.asarray(basis[0])


def _patch_residuals(
    patches: np.ndarray, basis: np.ndarray, mean: np.ndarray | None
) -> np.ndarray:
    """Unexplained fraction of each patch after the motif reconstruction.

    With a mean, the reconstruction is the mean plus the components. Without
    one, it is the projection onto the components. ``patches`` has shape
    ``(N, F, W)``.
    """
    energy = np.sum(patches * patches, axis=(1, 2))
    if mean is None:
        coeffs = np.tensordot(patches, basis, axes=([1, 2], [1, 2]))
        numer = energy - np.sum(coeffs * coeffs, axis=1)
    else:
        delta = patches - mean
        coeffs = np.tensordot(delta, basis, axes=([1, 2], [1, 2]))
        numer = np.sum(delta * delta, axis=(1, 2)) - np.sum(coeffs * coeffs, axis=1)
    residual = np.ones(patches.shape[0], dtype=np.float64)
    positive = energy > 0.0
    residual[positive] = np.maximum(numer[positive], 0.0) / energy[positive]
    return residual


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


def _shift_scale_dots(dots: np.ndarray, sigma: float) -> np.ndarray:
    """Subtract the slow mean and divide by the slow standard deviation.

    ``mean(t)`` is ``dots`` smoothed by a Gaussian of standard deviation
    ``sigma``. ``std(t)`` is the standard deviation of ``dots`` in that same
    window. Weights past the ends are dropped and the remaining weights are
    renormalized. A time whose standard deviation is 0 is left unchanged.
    """
    values = np.asarray(dots, dtype=np.float64)
    mean = _gaussian_smooth(values, sigma)
    squared = np.where(np.isfinite(values), values * values, np.nan)
    second = _gaussian_smooth(squared, sigma)
    variance = second - mean * mean
    std = np.sqrt(np.maximum(variance, 0.0))
    shifted = np.array(values, copy=True)
    good = np.isfinite(values) & np.isfinite(mean) & np.isfinite(std) & (std > 0.0)
    shifted[good] = (values[good] - mean[good]) / std[good]
    return shifted


def _peak_ratios(dots: np.ndarray, sigma: float, shift_sigma: float) -> np.ndarray:
    """Peak-to-baseline ratios of the shifted and scaled dot products."""
    shifted = _shift_scale_dots(dots, shift_sigma)
    smoothed = _gaussian_smooth(shifted, sigma)
    peaks = _local_maxima(shifted)
    ratios = np.full(peaks.shape, np.nan, dtype=np.float64)
    if peaks.size == 0:
        return ratios
    original = shifted[peaks]
    baseline = smoothed[peaks]
    valid = np.isfinite(original) & np.isfinite(baseline) & (baseline != 0.0)
    ratios[valid] = original[valid] / baseline[valid]
    return ratios


def _specificity_p90(ratios: np.ndarray) -> tuple[np.ndarray, float]:
    """Finite peak ratios and their 90th percentile."""
    finite = np.asarray(ratios, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return finite, float("nan")
    return finite, float(np.percentile(finite, 90))


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
    shift_sigma: float,
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
    cover = np.zeros(n_time, dtype=np.int32)
    for _motif_id, start, width in placements:
        lo = max(0, int(start))
        hi = min(n_time, int(start) + int(width))
        if hi > lo:
            cover[lo:hi] += 1
    specificity: dict[int, np.ndarray] = {}
    specificity_p90: dict[int, float] = {}
    sigmas: dict[int, float] = {}
    for motif_id, dots in dots_by_motif:
        sigma = _sigma(motif_id, int(widths[motif_id]), k)
        sigmas[motif_id] = sigma
        ratios, percentile = _specificity_p90(_peak_ratios(dots, sigma, shift_sigma))
        specificity[motif_id] = ratios
        specificity_p90[motif_id] = percentile
    return {
        "coverage": (n_covered / n_total) if n_total else 0.0,
        "over_coverage": (placed / n_covered) if n_covered else float("nan"),
        "max_count": int(cover.max()) if n_time else 0,
        "n_bins_covered": int(n_covered),
        "n_bins_total": int(n_total),
        "placed_length": int(placed),
        "k": sigmas,
        "shift_sigma": float(shift_sigma),
        "specificity": specificity,
        "specificity_p90": specificity_p90,
    }


def evaluate_dictionary(
    dictionary_id: int = DEFAULT_DICTIONARY_ID,
    wav: Path | str = DEFAULT_WAV,
    k: int | float | Mapping[int, int | float] | None = None,
    *,
    catalog: Path | str = DEFAULT_CATALOG,
) -> dict[str, object]:
    """Evaluate ``dictionary_id`` on the GPL whitened spectrogram of ``wav``.

    ``k`` is the standard deviation, in spectrogram columns, of the Gaussian
    applied to each motif's shifted and scaled dot product. ``None`` uses half
    the width of each motif. The dot-product sequence is first shifted and
    scaled by a one-minute Gaussian. At a local maximum ``t*`` the specificity
    ratio is that shifted value divided by its short smoothing. The specificity
    summary is the 90th percentile of those ratios.
    Coverage is the fraction of columns covered by at least one kept placement.
    Over-coverage is the total placed length divided by the covered length.
    The residual of a placement is the unexplained fraction after that motif's
    reconstruction. The gap is the runner-up residual minus that residual.
    The coverage ratio is the largest per-motif column count divided by the
    smallest. The placement ratio is the same comparison of placement counts.
    """
    catalog = Path(catalog)
    wav = Path(wav)
    npy_dir = _whitened_dir(catalog, wav)
    spectrogram, offset = _stitch(npy_dir)
    windows = discover_windows(npy_dir)
    sec_per_bin = float(windows[0].overlap) / float(windows[0].sample_freq)
    shift_sigma = 60.0 / sec_per_bin
    n_time = int(spectrogram.shape[1])
    rows = _dictionary_rows(catalog, dictionary_id)
    widths = {motif_id: width for motif_id, width, _path in rows}
    placements = _placements(catalog, dictionary_id, offset)
    dots_by_motif = [
        (motif_id, _dot_products(spectrogram, _load_motif(path)))
        for motif_id, _width, path in rows
    ]
    result = _measures(placements, widths, dots_by_motif, n_time, k, shift_sigma)
    models = {motif_id: _load_basis(path) for motif_id, _width, path in rows}
    result.update(_residual_summary(spectrogram, offset, placements, models))
    result["dictionary_id"] = int(dictionary_id)
    result["wav"] = str(wav.expanduser().resolve())
    return result


def _window_bounds(offset: Mapping[str, int], n_time: int) -> tuple[np.ndarray, np.ndarray]:
    """Global ``[start, end)`` of each window, in stitch order."""
    starts = np.array(sorted(offset.values()), dtype=np.int64)
    ends = np.empty_like(starts)
    if starts.size:
        ends[:-1] = starts[1:]
        ends[-1] = int(n_time)
    return starts, ends


def _residual_summary(
    spectrogram: np.ndarray,
    offset: Mapping[str, int],
    placements: Sequence[tuple[int, int, int]],
    models: Mapping[int, tuple[np.ndarray, np.ndarray | None]],
) -> dict[str, object]:
    """Mean residual and runner-up gap of each motif, plus the two ratios.

    A candidate is scored on a patch of its own width that starts at the
    placement and stays inside that placement's window. The runner-up is the
    other motif with the smallest residual. The gap is that residual minus
    the assigned motif's residual.
    """
    motif_ids = list(models)
    index = {motif_id: i for i, motif_id in enumerate(motif_ids)}
    starts, ends = _window_bounds(offset, int(spectrogram.shape[1]))
    n_placements = len(placements)
    residuals = np.full((n_placements, len(motif_ids)), np.nan, dtype=np.float64)
    by_window: dict[int, list[int]] = {}
    assigned = np.empty(n_placements, dtype=np.int64)
    for placement_index, (motif_id, start, _width) in enumerate(placements):
        assigned[placement_index] = index[motif_id]
        window = int(np.searchsorted(ends, int(start), side="right"))
        if 0 <= window < len(ends) and int(start) >= int(starts[window]):
            by_window.setdefault(window, []).append(placement_index)
    for motif_id in motif_ids:
        basis, mean = models[motif_id]
        width = int(basis.shape[-1])
        column = index[motif_id]
        for window, members in by_window.items():
            room = int(ends[window])
            ok = [i for i in members if int(placements[i][1]) + width <= room]
            if not ok:
                continue
            patches = np.stack(
                [
                    spectrogram[:, int(placements[i][1]) : int(placements[i][1]) + width]
                    for i in ok
                ]
            )
            residuals[np.array(ok), column] = _patch_residuals(patches, basis, mean)

    columns: dict[int, set[int]] = {motif_id: set() for motif_id in motif_ids}
    counts: Counter[int] = Counter()
    own_values: dict[int, list[float]] = {motif_id: [] for motif_id in motif_ids}
    runner_values: dict[int, list[float]] = {motif_id: [] for motif_id in motif_ids}
    gaps: dict[int, list[float]] = {motif_id: [] for motif_id in motif_ids}
    runners: dict[int, list[int]] = {motif_id: [] for motif_id in motif_ids}
    for placement_index, (motif_id, start, width) in enumerate(placements):
        counts[motif_id] += 1
        columns[motif_id].update(range(int(start), int(start) + int(width)))
        row = residuals[placement_index].copy()
        own = float(row[assigned[placement_index]])
        row[assigned[placement_index]] = np.nan
        if not np.isfinite(own) or not np.any(np.isfinite(row)):
            continue
        choice = int(np.nanargmin(row))
        own_values[motif_id].append(float(own))
        runner_values[motif_id].append(float(row[choice]))
        gaps[motif_id].append(float(row[choice] - own))
        runners[motif_id].append(motif_ids[choice])

    motifs: dict[int, dict[str, object]] = {}
    gap_means: list[float] = []
    for motif_id in motif_ids:
        basis, _mean = models[motif_id]
        votes = Counter(runners[motif_id])
        mean_gap = float(np.mean(gaps[motif_id])) if gaps[motif_id] else float("nan")
        if gaps[motif_id]:
            gap_means.append(mean_gap)
        motifs[motif_id] = {
            "n_components": int(basis.shape[0]),
            "n_placements": int(counts[motif_id]),
            "n_columns": len(columns[motif_id]),
            "residual": float(np.mean(own_values[motif_id])) if own_values[motif_id] else float("nan"),
            "residual_median": (
                float(np.median(own_values[motif_id])) if own_values[motif_id] else float("nan")
            ),
            "runner_residual": (
                float(np.mean(runner_values[motif_id])) if runner_values[motif_id] else float("nan")
            ),
            "gap": mean_gap,
            "gap_median": float(np.median(gaps[motif_id])) if gaps[motif_id] else float("nan"),
            "runner_id": votes.most_common(1)[0][0] if votes else None,
        }
    column_counts = [len(covered) for covered in columns.values()]
    placement_counts = [int(counts[motif_id]) for motif_id in motif_ids]
    return {
        "n_motifs": len(motif_ids),
        "coverage_ratio": _count_ratio(column_counts),
        "placement_ratio": _count_ratio(placement_counts),
        "min_gap": float(min(gap_means)) if gap_means else float("nan"),
        "motifs": motifs,
    }


def _count_ratio(counts: Sequence[int]) -> float:
    """Largest count divided by the smallest. Undefined when a count is zero."""
    if not counts or min(counts) <= 0:
        return float("nan")
    return float(max(counts) / min(counts))
