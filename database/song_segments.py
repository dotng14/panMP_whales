"""Song-structure segments: read either CSV layout, store and read them as Parquet.

A song-structure CSV comes in one of two layouts:

- **wide**: one row per motif with its song / theme / phrase
  (``motif, song, theme, phrase_id, phrase_count, start_unix, end_unix``);
- **long**: one row per motif, theme or song span
  (``type, segment index, start_unix, end_unix``).

Both become one table, one row per segment (``SCHEMA``):

- ``kind``: ``motif``, ``theme`` or ``song``;
- ``label``: the motif name, or the theme or song number as text;
- ``song`` / ``theme``: for a motif, the song and theme it starts in (null
  outside every span); for a theme or song span, its own number;
- ``phrase_id`` / ``phrase_count``: wide layout only, null otherwise;
- ``start_unix`` / ``end_unix``: unix seconds.

A wide CSV has no theme spans (themes are per motif); its song spans are the
extent of each song's motifs. Rows are sorted by kind, then start.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

WIDE_COLUMNS = frozenset({"motif", "song", "theme", "phrase_id", "phrase_count", "start_unix", "end_unix"})
LONG_COLUMNS = frozenset({"type", "segment index", "start_unix", "end_unix"})
KINDS = ("motif", "theme", "song")

SCHEMA = pa.schema(
    [
        ("kind", pa.dictionary(pa.int8(), pa.string())),
        ("label", pa.string()),
        ("song", pa.int32()),
        ("theme", pa.int32()),
        ("phrase_id", pa.int32()),
        ("phrase_count", pa.int32()),
        ("start_unix", pa.float64()),
        ("end_unix", pa.float64()),
    ]
)


class SegmentsError(ValueError):
    """A CSV in neither layout."""


def read_csv(path: Path | str) -> pa.Table:
    """The segments of a song-structure CSV in either layout; rows with ``end <= start`` are dropped."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = set(reader.fieldnames or ())
        if LONG_COLUMNS <= columns:
            rows = _from_long(reader)
        elif WIDE_COLUMNS <= columns:
            rows = _from_wide(reader)
        else:
            # Name what is missing from the layout the header looks like.
            missing = (LONG_COLUMNS if "type" in columns else WIDE_COLUMNS) - columns
            raise SegmentsError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    rows.sort(key=lambda r: (KINDS.index(r["kind"]), r["start_unix"]))
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def write_parquet(table: pa.Table, path: Path | str) -> Path:
    """Write ``table`` (zstd) via a temporary file, so a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    return path


def read_parquet(path: Path | str) -> pa.Table:
    return pq.read_table(path, schema=SCHEMA)


def counts(table: pa.Table) -> tuple[int, int]:
    """``(motifs, songs)``."""
    kinds = table.column("kind").to_pylist()
    return kinds.count("motif"), kinds.count("song")


def _row(kind: str, label: str, start: float, end: float, **extra: Any) -> dict[str, Any]:
    row = {"kind": kind, "label": label, "song": None, "theme": None, "phrase_id": None, "phrase_count": None}
    row.update(extra, start_unix=start, end_unix=end)
    return row


def _int(text: str) -> int:
    return int(float(text))


def _from_wide(reader: csv.DictReader) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    song_bounds: dict[int, list[float]] = {}
    for r in reader:
        start, end = float(r["start_unix"]), float(r["end_unix"])
        if end <= start:
            continue
        song = _int(r["song"])
        rows.append(
            _row(
                "motif", str(r["motif"]).strip(), start, end,
                song=song, theme=_int(r["theme"]),
                phrase_id=_int(r["phrase_id"]), phrase_count=_int(r["phrase_count"]),
            )
        )
        bounds = song_bounds.setdefault(song, [start, end])
        bounds[0], bounds[1] = min(bounds[0], start), max(bounds[1], end)
    rows += [_row("song", str(s), b[0], b[1], song=s) for s, b in song_bounds.items()]
    return rows


def _from_long(reader: csv.DictReader) -> list[dict[str, Any]]:
    motifs: list[tuple[str, float, float]] = []
    spans: dict[str, list[tuple[int, float, float]]] = {"theme": [], "song": []}
    for r in reader:
        start, end = float(r["start_unix"]), float(r["end_unix"])
        if end <= start:
            continue
        kind, index = r["type"].strip().lower(), str(r["segment index"]).strip()
        if kind == "motif":
            motifs.append((index, start, end))
        elif kind in spans:
            spans[kind].append((_int(index), start, end))

    def containing(kind: str, t: float) -> int | None:
        return next((n for n, s, e in spans[kind] if s <= t < e), None)

    rows = [
        _row("motif", m, s, e, song=containing("song", s), theme=containing("theme", s))
        for m, s, e in motifs
    ]
    rows += [_row("theme", str(n), s, e, theme=n) for n, s, e in spans["theme"]]
    rows += [_row("song", str(n), s, e, song=n) for n, s, e in spans["song"]]
    return rows
