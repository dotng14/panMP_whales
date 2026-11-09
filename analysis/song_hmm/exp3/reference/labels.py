"""Parse Audacity label exports of the hand-labelled reference clips.

An Audacity export is tab-separated ``start<TAB>end<TAB>label`` in seconds from
the clip start; point labels have ``start == end``. Lines starting with ``\``
(spectral-selection frequency lines) are ignored. Accepted labels:

* ``T1`` .. ``T6`` -- a theme span
* ``S``            -- a song start (point label; for a span its start is used)
* ``BG``           -- a non-song gap

Times are shifted by the clip offset in ``clips.json`` to recording time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REFERENCE_DIR = Path(__file__).resolve().parent
LABELS_DIR = REFERENCE_DIR / "labels"
CLIPS_JSON = REFERENCE_DIR.parents[0] / "work" / "reference" / "clips.json"
_THEME = re.compile(r"^T([1-6])$")


@dataclass
class ClipLabels:
    name: str
    offset_s: float
    duration_s: int
    seconds: np.ndarray  # (duration_s,) recording time of each second's start
    theme: np.ndarray  # (duration_s,) int32, 0..5, -1 = background / unlabelled
    theme_spans: np.ndarray  # (n, 3) recording-time start, end, theme 0..5
    song_starts: np.ndarray  # (k,) recording time
    bg_spans: np.ndarray  # (m, 2) recording-time start, end


def read_audacity(path: Path) -> list[tuple[float, float, str]]:
    """Rows ``(start, end, label)`` of an Audacity label export, label upper-cased."""
    rows = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.startswith("\\"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"{path}:{lineno}: expected start<TAB>end<TAB>label, got {line!r}")
        start, end = float(parts[0]), float(parts[1])
        label = parts[2].strip().upper() if len(parts) > 2 else ""
        if end < start:
            raise ValueError(f"{path}:{lineno}: end before start")
        rows.append((start, end, label))
    return rows


def parse_rows(
    rows: list[tuple[float, float, str]], name: str, offset_s: float, duration_s: int
) -> ClipLabels:
    themes, starts, bgs = [], [], []
    for start, end, label in rows:
        m = _THEME.match(label)
        if m:
            if end <= start:
                raise ValueError(f"{name}: theme label {label} at {start} s has no length")
            themes.append((start, end, int(m.group(1)) - 1))
        elif label == "S":
            starts.append(start)
        elif label == "BG":
            bgs.append((start, end))
        else:
            raise ValueError(f"{name}: unknown label {label!r} at {start} s")

    themes.sort()
    theme = np.full(duration_s, -1, dtype=np.int32)
    mid = np.arange(duration_s) + 0.5
    for start, end, th in themes:  # later spans win on overlap
        theme[(mid >= start) & (mid < end)] = th

    spans = np.array(themes, dtype=np.float64).reshape(-1, 3)
    spans[:, :2] += offset_s
    return ClipLabels(
        name=name,
        offset_s=float(offset_s),
        duration_s=int(duration_s),
        seconds=offset_s + np.arange(duration_s, dtype=np.float64),
        theme=theme,
        theme_spans=spans,
        song_starts=np.sort(np.array(starts, dtype=np.float64)) + offset_s,
        bg_spans=np.array(sorted(bgs), dtype=np.float64).reshape(-1, 2) + offset_s,
    )


def load_clip(name: str, labels_path: Path | None = None, clips_json: Path = CLIPS_JSON) -> ClipLabels:
    """Labels of clip ``name`` (e.g. ``clip1``) in recording time."""
    clip = json.loads(Path(clips_json).read_text(encoding="utf-8"))["clips"][name]
    labels_path = labels_path or LABELS_DIR / f"{name}_labels.txt"
    return parse_rows(read_audacity(labels_path), name, clip["offset_s"], int(clip["duration_s"]))


def load_all(labels_dir: Path = LABELS_DIR, clips_json: Path = CLIPS_JSON) -> dict[str, ClipLabels]:
    """All clips in ``clips.json`` that have a ``<clip>_labels.txt`` in ``labels_dir``."""
    names = json.loads(Path(clips_json).read_text(encoding="utf-8"))["clips"]
    return {
        n: load_clip(n, labels_dir / f"{n}_labels.txt", clips_json)
        for n in names
        if (labels_dir / f"{n}_labels.txt").exists()
    }


def recording_theme(clips: dict[str, ClipLabels], n_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-second theme over the whole recording (-1 outside labels) and a mask of hand-labelled seconds."""
    theme = np.full(n_seconds, -1, dtype=np.int32)
    labelled = np.zeros(n_seconds, dtype=bool)
    for c in clips.values():
        idx = c.seconds.astype(np.int64)
        ok = (idx >= 0) & (idx < n_seconds)
        theme[idx[ok]] = c.theme[ok]
        labelled[idx[ok]] = True
    return theme, labelled
