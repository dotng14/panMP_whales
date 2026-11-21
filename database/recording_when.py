"""Universal recording identity: site letter + wall-clock start time.

Primary key for the picker is ``site`` + ``YYYY-MM-DD HH:MM:SS`` (or a prefix),
not SOCAL filename layout. Legacy stems still parse via ``parse_wav_name``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .wav_name import parse_wav_name

# Full or partial wall-clock forms accepted from paste / When field.
_ISO_FULL = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?:[ T](?P<time>\d{2}(?::\d{2}(?::\d{2})?)?)?)?$"
)
_SITE_PREFIX = re.compile(
    r"^(?:site\s*)?(?P<site>[A-Za-z])\s+(?P<rest>.+)$",
    re.IGNORECASE,
)
_YYMMDD = re.compile(r"^\d{6}$")
_HHMMSS = re.compile(r"^\d{6}$")


@dataclass(frozen=True)
class RecordingWhen:
    """Universal lookup key (site letter + start, possibly partial)."""

    site: str | None = None
    start: str | None = None  # ``YYYY-MM-DD`` … ``YYYY-MM-DD HH:MM:SS``


def yymmdd_hhmmss_to_start(date: str | None, time: str | None = None) -> str | None:
    """Map stem ``date``/``time`` (``YYMMDD`` / ``HHMMSS``) to wall-clock text."""
    if not date or not _YYMMDD.match(date):
        return None
    century = 2000
    y = century + int(date[:2])
    mo, d = int(date[2:4]), int(date[4:6])
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    base = f"{y:04d}-{mo:02d}-{d:02d}"
    if not time:
        return base
    if not _HHMMSS.match(time):
        return base
    hh, mm, ss = int(time[:2]), int(time[2:4]), int(time[4:6])
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        return base
    return f"{base} {hh:02d}:{mm:02d}:{ss:02d}"


def normalize_start_prefix(raw: str) -> str | None:
    """Normalize a When string to a comparable ``start_time`` prefix, or None."""
    text = raw.strip().replace("T", " ", 1)
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    m = _ISO_FULL.match(text)
    if not m:
        return None
    date = m.group("date")
    time = m.group("time")
    if not time:
        return date
    parts = time.split(":")
    if len(parts) == 1:
        return f"{date} {parts[0]}"
    if len(parts) == 2:
        return f"{date} {parts[0]}:{parts[1]}"
    return f"{date} {parts[0]}:{parts[1]}:{parts[2]}"


def parse_recording_when(text: str, *, default_site: str = "H") -> RecordingWhen:
    """Parse paste / When input into site + start.

    Accepts:
    - ``2023-04-26 19:35:00`` / ``2023-04-26T19:35:00`` / partials
    - ``H 2023-04-26 19:35:00``
    - Legacy SOCAL stem / partial → site + start from ``date``/``time``
    """
    raw = (text or "").strip()
    if not raw:
        return RecordingWhen(site=default_site)

    site = default_site
    rest = raw
    sm = _SITE_PREFIX.match(raw)
    if sm and normalize_start_prefix(sm.group("rest")) is not None:
        site = sm.group("site").upper()
        rest = sm.group("rest").strip()

    start = normalize_start_prefix(rest)
    if start is not None:
        return RecordingWhen(site=site, start=start)

    parsed = parse_wav_name(raw)
    if parsed is not None:
        start = yymmdd_hhmmss_to_start(parsed.date, parsed.time)
        return RecordingWhen(
            site=(parsed.site or default_site),
            start=start,
        )

    return RecordingWhen(site=default_site, start=None)


def start_filter_sql(start: str) -> tuple[str, str]:
    """Return ``(sql_fragment_with_?, arg)`` for prefix or exact ``start_time`` match.

    Full second → equality; shorter → ``LIKE prefix%``.
    """
    normalized = normalize_start_prefix(start) or start.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", normalized):
        return "start_time = ?", normalized
    return "start_time LIKE ?", f"{normalized}%"
