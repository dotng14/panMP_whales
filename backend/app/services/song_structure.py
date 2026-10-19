"""Load motif / song / theme / phrase structure from a repo-local CSV.

CSV files live under ``backend/data/song_structure/<wav_stem>.motif_segments.csv``
and are looked up by WAV stem only (no machine-local paths).
"""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import BACKEND_DIR

SONG_STRUCTURE_DIR = BACKEND_DIR / "data" / "song_structure"

# SOCAL stems often use YYMMDD_HHMMSS; some libraries use YYYYMMDD_HHMMSS.
_WAV_DATETIME_RE = re.compile(r"(?<!\d)(\d{6,8})_(\d{6})(?!\d)")


def wav_start_unix(stem: str) -> float | None:
    """Parse recording-start unix seconds from a WAV stem, if present."""
    for m in _WAV_DATETIME_RE.finditer(stem):
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

    start_unix0 = wav_start_unix(stem)
    if start_unix0 is None:
        raise ValueError(f"cannot parse recording start time from stem {stem!r}")

    units: list[dict[str, Any]] = []
    song_bounds: dict[int, dict[str, float]] = {}

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {
            "motif",
            "song",
            "theme",
            "phrase_id",
            "phrase_count",
            "start_unix",
            "end_unix",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"motif CSV missing required columns; got {reader.fieldnames!r}"
            )
        for row in reader:
            start_sec = float(row["start_unix"]) - start_unix0
            end_sec = float(row["end_unix"]) - start_unix0
            if end_sec <= start_sec:
                continue
            song = int(float(row["song"]))
            theme = int(float(row["theme"]))
            phrase_id = int(float(row["phrase_id"]))
            phrase_count = int(float(row["phrase_count"]))
            units.append(
                {
                    "motif": str(row["motif"]).strip(),
                    "song": song,
                    "theme": theme,
                    "phraseId": phrase_id,
                    "phraseCount": phrase_count,
                    "startSec": start_sec,
                    "endSec": end_sec,
                }
            )
            bounds = song_bounds.get(song)
            if bounds is None:
                song_bounds[song] = {"startSec": start_sec, "endSec": end_sec}
            else:
                bounds["startSec"] = min(bounds["startSec"], start_sec)
                bounds["endSec"] = max(bounds["endSec"], end_sec)

    songs = [
        {"song": song, "startSec": bounds["startSec"], "endSec": bounds["endSec"]}
        for song, bounds in sorted(song_bounds.items())
    ]
    return {
        "wavStartUnix": start_unix0,
        "units": units,
        "songs": songs,
    }
