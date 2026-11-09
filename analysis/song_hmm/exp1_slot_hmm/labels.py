"""Read the motif-label CSV into per-second slot and theme, song spans, and unit lengths.

The CSV seeds the model and is the reference in the report. It is not
treated as ground truth.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .topology import FIRST_SLOT, N_DUR, N_SLOTS, SLOT_THEME, THEME_N_SLOTS

_STAMP = re.compile(r"(?<!\d)(\d{6,8})_(\d{6})(?!\d)")


def recording_start_unix(name: str) -> float:
    """Start time encoded in a file name such as ``SOCAL_H_76_230426_193500_...``."""
    for match in _STAMP.finditer(name):
        date_s, time_s = match.groups()
        if len(date_s) == 6:
            year, month, day = 2000 + int(date_s[:2]), int(date_s[2:4]), int(date_s[4:6])
        elif len(date_s) == 8:
            year, month, day = int(date_s[:4]), int(date_s[4:6]), int(date_s[6:8])
        else:
            continue
        stamp = datetime(year, month, day, int(time_s[:2]), int(time_s[2:4]), int(time_s[4:6]), tzinfo=timezone.utc)
        return stamp.timestamp()
    raise ValueError(f"no recording start in {name!r}")


@dataclass
class Labels:
    slot: np.ndarray  # (T,) slot index per second, -1 where no unit covers the second's midpoint
    theme: np.ndarray  # (T,) 0-based theme per second, -1 where no unit
    song_bounds: np.ndarray  # (n_songs, 2) start, end in seconds, time order
    units: np.ndarray  # (n_units, 3) start s, end s, slot; time order
    phrases: np.ndarray  # (n_songs, 6) phrase count per theme per song


def load_labels(csv_path: Path, n_seconds: int) -> Labels:
    origin = recording_start_unix(csv_path.name)
    phrases: dict[tuple[int, int, int], list[tuple[float, float]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            start = float(row["start_unix"]) - origin
            end = float(row["end_unix"]) - origin
            if end > start:
                key = (int(float(row["song"])), int(float(row["theme"])) - 1, int(float(row["phrase_count"])))
                phrases[key].append((start, end))

    units: list[tuple[float, float, int]] = []
    spans: dict[int, list[float]] = {}
    counts: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(len(THEME_N_SLOTS), dtype=np.int64))
    for (song, theme, _), pieces in phrases.items():
        pieces.sort()
        counts[song][theme] += 1
        for position, (start, end) in enumerate(pieces[: THEME_N_SLOTS[theme]]):
            slot = int(FIRST_SLOT[theme]) + position
            units.append((start, end, slot))
            span = spans.setdefault(song, [start, end])
            span[0], span[1] = min(span[0], start), max(span[1], end)

    units.sort()
    starts = np.array([u[0] for u in units])
    ends = np.array([u[1] for u in units])
    unit_slot = np.array([u[2] for u in units], dtype=np.int32)
    mid = np.arange(n_seconds) + 0.5
    idx = np.searchsorted(starts, mid, side="right") - 1
    covered = (idx >= 0) & (mid < ends[np.clip(idx, 0, None)])
    slot = np.where(covered, unit_slot[np.clip(idx, 0, None)], -1).astype(np.int32)
    theme = np.where(slot >= 0, SLOT_THEME[np.clip(slot, 0, None)], -1).astype(np.int32)

    songs = sorted(spans, key=lambda s: spans[s][0])
    return Labels(
        slot=slot,
        theme=theme,
        song_bounds=np.array([spans[s] for s in songs], dtype=np.float64),
        units=np.column_stack([starts, ends, unit_slot]),
        phrases=np.array([counts[s] for s in songs]),
    )


def length_hist(labels: Labels, until_s: float) -> np.ndarray:
    """Unit lengths (1..5 s bins) per slot for units that end by ``until_s``."""
    hist = np.zeros((N_SLOTS, N_DUR))
    for start, end, slot in labels.units:
        if end <= until_s:
            hist[int(slot), int(np.clip(round(end - start), 1, N_DUR)) - 1] += 1
    return hist
