"""Parse HARP / Triton-style recording filenames into structured components.

v1 registers SOCAL only. Append another parser to ``PARSERS`` for future
families (e.g. CHNMS) without changing call sites.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

COMPONENT_KEYS = (
    "family",
    "site",
    "deployment",
    "disk",
    "date",
    "time",
    "df",
    "channel",
    "resample",
)


@dataclass(frozen=True)
class RecordingComponents:
    family: str | None = None
    site: str | None = None
    deployment: str | None = None
    disk: str | None = None
    date: str | None = None
    time: str | None = None
    df: str | None = None  # "20", "100", or "native" when explicitly no df tag
    channel: str | None = None
    resample: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)

    def filled(self) -> dict[str, str]:
        return {k: v for k, v in self.to_dict().items() if v is not None and v != ""}


def normalize_wav_basename(name: str) -> str:
    """Strip directories and a trailing ``.wav`` (any case)."""
    stem = Path(str(name).strip()).name
    if stem.lower().endswith(".wav"):
        stem = stem[:-4]
    return stem


_MODERN_HEAD = re.compile(
    r"^SOCAL_H_(?P<deployment>\d+)(?P<he>_HE)?",
    re.IGNORECASE,
)
_OLD_HEAD = re.compile(
    r"^SOCAL(?P<deployment>\d+)H",
    re.IGNORECASE,
)
_DISK_EXPLICIT = re.compile(
    r"(?:^|_)(?:DL(?P<dl>\d+)|disk(?P<disk>\d+)|d(?P<d>\d+))(?=_|\.|$)",
    re.IGNORECASE,
)
# Old form SOCAL38H_01_YYMMDD_HHMMSS — bare disk number before the datetime.
_DISK_BARE = re.compile(r"^_(?P<bare>\d{1,2})_(?=\d{6}_\d{6})")
_DATETIME = re.compile(r"(?<!\d)(?P<date>\d{6})_(?P<time>\d{6})(?!\d)")
# Full ``df20`` / ``.df100``; short ``.d100`` (dot-d only — not ``_d01`` disk ids).
_DF = re.compile(r"(?:^|[._])df(?P<df>\d+)(?=[._]|$)", re.IGNORECASE)
_DF_SHORT = re.compile(r"\.d(?P<df>\d+)(?=\.|$)", re.IGNORECASE)
_CHANNEL = re.compile(r"(?<![A-Za-z0-9])(?P<channel>x(?:\.x)?)(?=(?:_\d+kHz)?$)", re.IGNORECASE)
_RESAMPLE = re.compile(r"_(?P<resample>\d+kHz)$", re.IGNORECASE)


def _parse_socal(stem: str) -> RecordingComponents | None:
    if not stem.upper().startswith("SOCAL"):
        return None

    site: str | None = None
    deployment: str | None = None
    rest = stem

    m = _MODERN_HEAD.match(stem)
    if m:
        site = "H"
        deployment = m.group("deployment")
        rest = stem[m.end() :]
    else:
        m = _OLD_HEAD.match(stem)
        if m:
            site = "H"
            deployment = m.group("deployment")
            rest = stem[m.end() :]
        elif stem.upper() in {"SOCAL", "SOCAL_H"}:
            return RecordingComponents(family="SOCAL", site="H" if "_" in stem.upper() else None)
        else:
            # "SOCAL_" or other incomplete prefix still counts as SOCAL family.
            return RecordingComponents(family="SOCAL")

    disk: str | None = None
    dm = _DISK_EXPLICIT.search(rest)
    if dm:
        if dm.group("dl") is not None:
            disk = f"DL{dm.group('dl')}"
        elif dm.group("disk") is not None:
            disk = f"disk{dm.group('disk')}"
        else:
            disk = f"d{dm.group('d')}"
    else:
        bm = _DISK_BARE.match(rest)
        if bm:
            disk = bm.group("bare")

    date: str | None = None
    time: str | None = None
    dt = _DATETIME.search(stem)
    if dt:
        date = dt.group("date")
        time = dt.group("time")
    else:
        # Partial paste: SOCAL_H_76_230426 (date without time).
        alone = re.search(r"(?<!\d)(\d{6})(?!\d)", rest)
        if alone:
            date = alone.group(1)

    channel: str | None = None
    ch = _CHANNEL.search(stem)
    if ch:
        channel = ch.group("channel").lower()

    resample: str | None = None
    rs = _RESAMPLE.search(stem)
    if rs:
        resample = rs.group("resample")

    df: str | None = None
    dfm = _DF.search(stem) or _DF_SHORT.search(stem)
    if dfm:
        df = dfm.group("df")
    elif channel is not None and date is not None and time is not None:
        # Channel present and no df tag → native sample rate.
        df = "native"

    return RecordingComponents(
        family="SOCAL",
        site=site,
        deployment=deployment,
        disk=disk,
        date=date,
        time=time,
        df=df,
        channel=channel,
        resample=resample,
    )


Parser = Callable[[str], RecordingComponents | None]

# Future families: append parsers here (e.g. parse_chnms).
PARSERS: list[Parser] = [_parse_socal]


def parse_wav_name(name: str) -> RecordingComponents | None:
    """Return components for the first matching family parser, or None."""
    stem = normalize_wav_basename(name)
    if not stem:
        return None
    for parser in PARSERS:
        got = parser(stem)
        if got is not None:
            return got
    return None


def components_from_row(row: dict) -> RecordingComponents:
    """Build components from a ``wav_files`` row (or any mapping with those keys)."""
    kwargs = {f.name: row.get(f.name) for f in fields(RecordingComponents)}
    return RecordingComponents(**kwargs)
