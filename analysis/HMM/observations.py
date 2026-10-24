"""GPL windows to a 10 Hz series, panMP prototypes, and 1-second motif scores.

Labels from the motif CSV cut the recording into songs and name the slot
that each second belongs to. They are not the observation. A labeled song
may end in any theme. The return from theme 6 to theme 1 is an edge in
``topology``, not a cut made here.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .topology import FIRST_SLOT, N_SLOTS, THEME_N_SLOTS

HZ = 10
MOTIF_LENGTHS = (10, 20, 30, 40, 50)
N_PROTOTYPES = 12
_WAV_DATETIME_RE = re.compile(r"(?<!\d)(\d{6,8})_(\d{6})(?!\d)")


def recording_start_unix(stem: str) -> float:
    for match in _WAV_DATETIME_RE.finditer(stem):
        date_s, time_s = match.group(1), match.group(2)
        if len(date_s) == 8:
            year, month, day = int(date_s[0:4]), int(date_s[4:6]), int(date_s[6:8])
        elif len(date_s) == 6:
            year, month, day = 2000 + int(date_s[0:2]), int(date_s[2:4]), int(date_s[4:6])
        else:
            continue
        hour, minute, second = int(time_s[0:2]), int(time_s[2:4]), int(time_s[4:6])
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp()
    raise ValueError(f"cannot parse recording start from {stem!r}")


def load_slot_labels(csv_path: Path, n_seconds: int) -> dict[str, np.ndarray]:
    """Per-second slot and theme, song bounds, and a 5-bin duration histogram per slot.

    ``theme_per_second`` is 0 where no unit covers the midpoint of that second,
    otherwise the 1-based theme. ``slot_per_second`` uses -1 for the same gap.
    """
    stem = csv_path.name.replace(".motif_segments.csv", "")
    origin = recording_start_unix(stem)
    groups: dict[tuple[int, int, int], list[tuple[float, float]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            start = float(row["start_unix"]) - origin
            end = float(row["end_unix"]) - origin
            if end <= start:
                continue
            key = (int(float(row["song"])), int(float(row["theme"])), int(float(row["phrase_count"])))
            groups[key].append((start, end))

    starts: list[float] = []
    ends: list[float] = []
    slots: list[int] = []
    hist = np.zeros((N_SLOTS, 5), dtype=np.float64)
    song_span: dict[int, list[float]] = {}
    for (song, theme, _phrase_count), pieces in groups.items():
        pieces.sort()
        n_slots = THEME_N_SLOTS[theme - 1]
        origin_slot = int(FIRST_SLOT[theme - 1])
        for position, (start, end) in enumerate(pieces):
            if position >= n_slots:
                continue
            slot = origin_slot + position
            starts.append(start)
            ends.append(end)
            slots.append(slot)
            length = int(np.clip(np.rint(end - start), 1, 5))
            hist[slot, length - 1] += 1.0
            span = song_span.get(song)
            if span is None:
                song_span[song] = [start, end]
            else:
                span[0] = min(span[0], start)
                span[1] = max(span[1], end)

    order = np.argsort(np.asarray(starts))
    start_a = np.asarray(starts, dtype=np.float64)[order]
    end_a = np.asarray(ends, dtype=np.float64)[order]
    slot_a = np.asarray(slots, dtype=np.int32)[order]

    slot_per_second = np.full(n_seconds, -1, dtype=np.int32)
    theme_per_second = np.zeros(n_seconds, dtype=np.int32)
    from .topology import SLOT_THEME

    for second in range(n_seconds):
        mid = second + 0.5
        idx = int(np.searchsorted(start_a, mid, side="right") - 1)
        if idx < 0 or mid >= end_a[idx]:
            continue
        slot_per_second[second] = slot_a[idx]
        theme_per_second[second] = int(SLOT_THEME[slot_a[idx]]) + 1

    songs = np.array(sorted(song_span), dtype=np.int32)
    bounds = np.array([song_span[int(s)] for s in songs], dtype=np.float64)
    return {
        "slot_per_second": slot_per_second,
        "theme_per_second": theme_per_second,
        "song_id": songs,
        "song_bounds": bounds,
        "duration_hist": hist,
    }


def spectrogram_paths(npy_dir: Path) -> list[Path]:
    paths = [
        path
        for path in npy_dir.glob("spwhiten_*.npy")
        if not path.name.endswith("_display.npy")
    ]
    return sorted(paths, key=lambda path: path.name)


def energy_series_10hz(npy_dir: Path, n_seconds: int) -> np.ndarray:
    """Sum each whitened window over frequency and average into 0.1 s bins."""
    n_bins = n_seconds * HZ
    total = np.zeros(n_bins, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.float64)
    paths = spectrogram_paths(npy_dir)
    if not paths:
        raise FileNotFoundError(f"no whitened spectrograms in {npy_dir}")
    for path in paths:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        spec = np.load(path)
        energy = np.asarray(spec, dtype=np.float64).sum(axis=0)
        hop = int(meta["overlap"])
        sample_rate = float(meta["sample_freq"])
        fftl = int(meta["fftl"])
        start_sample = int(meta["start_sample"]) - 1
        for frame, value in enumerate(energy):
            center = (start_sample + frame * hop + fftl / 2.0) / sample_rate
            bin_index = int(center * HZ)
            if 0 <= bin_index < n_bins:
                total[bin_index] += float(value)
                count[bin_index] += 1.0
    series = np.zeros(n_bins, dtype=np.float64)
    filled = count > 0
    series[filled] = total[filled] / count[filled]
    if not np.all(filled):
        known = np.flatnonzero(filled)
        if known.size == 0:
            raise RuntimeError("10 Hz series has no energy")
        missing = np.flatnonzero(~filled)
        series[missing] = np.interp(missing, known, series[known])
    return series


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if int(finite.sum()) < 2:
        out[finite] = 0.0
        return out
    std = float(np.std(values[finite], ddof=1))
    if std == 0.0:
        out[finite] = 0.0
        return out
    out[finite] = (values[finite] - float(np.mean(values[finite]))) / std
    return out


def _mpx():
    mp_dir = Path(__file__).resolve().parents[1] / "matrix_profile"
    if str(mp_dir) not in sys.path:
        sys.path.insert(0, str(mp_dir))
    from mpx_fast import mpx_fast

    return mpx_fast


def pan_prototypes(
    series: np.ndarray,
    *,
    lengths: tuple[int, ...] = MOTIF_LENGTHS,
    n_prototypes: int = N_PROTOTYPES,
    nthreads: int | None = None,
) -> list[np.ndarray]:
    """Strongest pan-MP peaks, with an exclusion zone of each motif's length."""
    import os

    series = np.asarray(series, dtype=np.float64).ravel()
    mpx_fast = _mpx()
    if nthreads is None:
        nthreads = max(1, (os.cpu_count() or 1))
    surfaces: list[np.ndarray] = []
    neighbors: list[np.ndarray] = []
    for length in lengths:
        print(f"panMP motif length {length} ({length / HZ:.0f} s)", flush=True)
        distance, index = mpx_fast(series, max(length - 2, 4), length, nthreads)
        score = -_zscore(distance / np.sqrt(length))
        column = np.full(len(series), -np.inf)
        good = np.isfinite(score)
        column[: len(score)][good] = score[good]
        neighbor = np.full(len(series), -1, dtype=np.int64)
        finite = np.isfinite(index)
        neighbor[: len(index)][finite] = np.rint(index[finite]).astype(np.int64)
        surfaces.append(column)
        neighbors.append(neighbor)
    surface = np.column_stack(surfaces)
    prototypes: list[np.ndarray] = []
    for _ in range(n_prototypes):
        flat = int(np.argmax(surface))
        if not np.isfinite(surface.ravel()[flat]) or surface.ravel()[flat] == -np.inf:
            break
        start, length_i = np.unravel_index(flat, surface.shape)
        start = int(start)
        length = int(lengths[int(length_i)])
        prototypes.append(series[start : start + length].copy())
        _exclude(surface, start, length)
        neighbor = int(neighbors[int(length_i)][start])
        if neighbor >= 0:
            _exclude(surface, neighbor, length)
    if not prototypes:
        raise RuntimeError("panMP found no motif prototypes")
    return prototypes


def _exclude(surface: np.ndarray, center: int, length: int) -> None:
    lo = max(0, center - length + 1)
    hi = min(surface.shape[0], center + length)
    surface[lo:hi, :] = -np.inf


def sliding_znorm_corr(series: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64).ravel()
    prototype = np.asarray(prototype, dtype=np.float64).ravel()
    width = len(prototype)
    centered = prototype - prototype.mean()
    scale = float(np.sqrt(np.dot(centered, centered)))
    if scale == 0.0:
        return np.zeros(len(series) - width + 1, dtype=np.float64)
    centered = centered / scale
    cumulative = np.cumsum(np.r_[0.0, series])
    cumulative_sq = np.cumsum(np.r_[0.0, series * series])
    window_sum = cumulative[width:] - cumulative[:-width]
    window_sq = cumulative_sq[width:] - cumulative_sq[:-width]
    var = np.maximum(window_sq / width - (window_sum / width) ** 2, 0.0)
    std = np.sqrt(var)
    dot = np.correlate(series, centered, mode="valid")
    corr = np.zeros_like(dot)
    ok = std > 1e-8
    corr[ok] = dot[ok] / std[ok]
    return corr


def scores_per_second(series: np.ndarray, prototypes: list[np.ndarray], n_seconds: int) -> np.ndarray:
    """Mean z-normalized correlation of each prototype inside each second."""
    scores = np.zeros((n_seconds, len(prototypes)), dtype=np.float64)
    for index, prototype in enumerate(prototypes):
        corr = sliding_znorm_corr(series, prototype)
        full = np.full(n_seconds * HZ, np.nan, dtype=np.float64)
        full[: min(len(corr), full.size)] = corr[: full.size]
        second = np.nanmean(full.reshape(n_seconds, HZ), axis=1)
        second = np.where(np.isfinite(second), second, 0.0)
        scores[:, index] = second
    return scores


def zscore_columns(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = scores.mean(axis=0)
    std = scores.std(axis=0)
    std = np.maximum(std, 1e-6)
    return (scores - mean) / std, mean, std


def apply_zscore(scores: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (scores - mean) / np.maximum(std, 1e-6)


def song_slices(labels: dict[str, np.ndarray], n_seconds: int) -> list[tuple[int, int, int]]:
    """(song id, start second inclusive, end second exclusive) in time order.

    The CSV span is the whole sequence. It may stop in any theme. Theme 6
    returning to theme 1 is handled by the transition graph, not by extending
    the slice.
    """
    slices = []
    for song, (start, end) in zip(labels["song_id"], labels["song_bounds"], strict=True):
        i0 = max(0, int(np.floor(start)))
        i1 = min(n_seconds, int(np.ceil(end)))
        if i1 - i0 >= 2:
            slices.append((int(song), i0, i1))
    return slices
