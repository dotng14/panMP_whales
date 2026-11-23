"""Load motif / song / theme / phrase structure for the viewer.

A CSV is picked from the catalog, which keeps a Parquet copy of its segments
(``load_song_structure_parquet``; ``database.song_segments`` has the layouts),
or, when the viewer asks without one, looked up by WAV stem under
``backend/data/song_structure/<wav_stem>.motif_segments.csv``.

Segment times are unix seconds; the WAV's start comes from a date-time in its stem,
or, for a stem without one (``wav23``), from ``wav_starts.json`` in the same
folder, which maps stem to ``YYMMDD_HHMMSS``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import BACKEND_DIR
from database import song_segments

SONG_STRUCTURE_DIR = BACKEND_DIR / "data" / "song_structure"
WAV_STARTS_FILE = SONG_STRUCTURE_DIR / "wav_starts.json"

# SOCAL stems often use YYMMDD_HHMMSS; some libraries use YYYYMMDD_HHMMSS.
_WAV_DATETIME_RE = re.compile(r"(?<!\d)(\d{6,8})_(\d{6})(?!\d)")


def wav_start_unix(stem: str) -> float | None:
    """Recording-start unix seconds from the stem's date-time, else from ``wav_starts.json``."""
    start = _parse_datetime(stem)
    if start is None and WAV_STARTS_FILE.is_file():
        given = json.loads(WAV_STARTS_FILE.read_text()).get(stem)
        if isinstance(given, str):
            start = _parse_datetime(given)
    return start


def _parse_datetime(text: str) -> float | None:
    for m in _WAV_DATETIME_RE.finditer(text):
        date_s, time_s = m.group(1), m.group(2)
        try:
            if len(date_s) == 8:
                y, mo, d = int(date_s[0:4]), int(date_s[4:6]), int(date_s[6:8])
            elif len(date_s) == 6:
                y, mo, d = 2000 + int(date_s[0:2]), int(date_s[2:4]), int(date_s[4:6])
            else:
                continue
            hh, mm, ss = int(time_s[0:2]), int(time_s[2:4]), int(time_s[4:6])
            return datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def song_structure_csv_path(stem: str) -> Path:
    return SONG_STRUCTURE_DIR / f"{stem}.motif_segments.csv"


def load_song_structure(stem: str) -> dict[str, Any] | None:
    """Return units + song spans for ``stem``, or None if the CSV is absent."""
    path = song_structure_csv_path(stem)
    if not path.is_file():
        return None
    return load_song_structure_csv(path, stem)


def load_song_structure_csv(path: Path, stem: str) -> dict[str, Any]:
    """The viewer payload from a CSV in either layout."""
    return _payload(_read(song_segments.read_csv, path), stem)


def load_song_structure_parquet(path: Path, stem: str) -> dict[str, Any]:
    """The viewer payload from the catalog's Parquet copy of a CSV."""
    return _payload(song_segments.read_parquet(path), stem)


def _read(reader, path: Path):
    try:
        return reader(path)
    except song_segments.SegmentsError as exc:
        raise ValueError(str(exc)) from exc


def _payload(table, stem: str) -> dict[str, Any]:
    """Units (motifs), songs and themes, in seconds from the WAV's start (``wav_start_unix``).

    ``themes`` is None when the themes are given per motif (wide layout).
    """
    start_unix0 = wav_start_unix(stem)
    if start_unix0 is None:
        raise ValueError(f"cannot parse recording start time from stem {stem!r}")
    units: list[dict[str, Any]] = []
    songs: list[dict[str, Any]] = []
    themes: list[dict[str, Any]] = []
    for r in table.to_pylist():
        start_sec, end_sec = r["start_unix"] - start_unix0, r["end_unix"] - start_unix0
        if r["kind"] == "motif":
            units.append(
                {
                    "motif": r["label"],
                    "song": r["song"],
                    "theme": r["theme"],
                    "phraseId": r["phrase_id"],
                    "phraseCount": r["phrase_count"],
                    "startSec": start_sec,
                    "endSec": end_sec,
                }
            )
        elif r["kind"] == "song":
            songs.append({"song": r["song"], "startSec": start_sec, "endSec": end_sec})
        elif r["kind"] == "theme":
            themes.append({"theme": r["theme"], "startSec": start_sec, "endSec": end_sec})
    return {"wavStartUnix": start_unix0, "units": units, "songs": songs, "themes": themes or None}
