"""SQLite catalog for WAV files, GPL segmentations, dictionaries, MP runs and CSVs.

Large arrays stay on disk as ``.npy``; this module stores metadata and paths.
Writers are the dict_learning CLIs (``--catalog``), ``python -m database`` and
the backend's rename / feedback routes; the backend reads.

Typical use::

    from database import catalog

    with catalog.connect("catalog.sqlite") as conn:
        catalog.register_run_dir(conn, "runs/build_a")          # randomized-build output
        catalog.register_run_dir(conn, "runs/build_a_mp")       # matching-pursuit output
        catalog.upsert_annotation_csv(conn, "x.motif_segments.csv")
        conn.commit()

    with catalog.connect("catalog.sqlite", read_only=True) as conn:
        catalog.list_analyses_for_stem(conn, "SOCAL_H_76_230426_193500_df100.x.x")
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

DICT_SOURCES = ("seed", "refinement", "matching", "edit")
TOOL_NAMES = ("randomized_build", "matching_pursuit")
FEEDBACK_LABELS = ("good", "bad")
MOTIF_CSV_SUFFIX = ".motif_segments.csv"
MOTIF_CSV_COLUMNS = frozenset(
    {"motif", "song", "theme", "phrase_id", "phrase_count", "start_unix", "end_unix"}
)
# The randomized-build knobs kept on the dictionary row (the rest stay in summary.json).
_BUILD_PARAMS = (
    "m_per_outer", "min_coverage", "max_outer", "refine_iters", "n_components", "n_min",
    "epsilon", "min_template_width", "max_template_width", "eps_step_fraction",
    "max_relative_distance", "post_refine_max_relative_distance", "seed",
    "n_windows", "window_offset",
)
_MP_PARAMS = ("n_windows", "window_offset", "n_atoms")

# WAV stems carry the recording start as YYMMDD_HHMMSS or YYYYMMDD_HHMMSS.
_WAV_DATETIME_RE = re.compile(r"(?<!\d)(\d{6,8})_(\d{6})(?!\d)")
# GPL window ids: spwhiten_<stem>_w<index>[_s<start>_e<end>].
_WINDOW_ID_RE = re.compile(r"^spwhiten_(?P<stem>.+)_w0*(?P<index>\d+)(?:_s\d+_e\d+)?$")


class CatalogError(Exception):
    """A folder or file that cannot be registered (bad layout or missing parent)."""


# ── helpers ──────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _json_or_none(value: Any) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True)


def window_id_parts(window_id: str) -> tuple[str, int] | None:
    """``spwhiten_<stem>_w000003_s…_e…`` → ``(stem, 3)``; None for other ids."""
    match = _WINDOW_ID_RE.match(window_id)
    if match is None:
        return None
    return match.group("stem"), int(match.group("index"))


def wav_start_datetime(stem: str) -> datetime | None:
    """Recording start parsed from a WAV stem, if it has one."""
    for match in _WAV_DATETIME_RE.finditer(stem):
        date_s, time_s = match.groups()
        if len(date_s) == 8:
            y, mo, d = int(date_s[:4]), int(date_s[4:6]), int(date_s[6:8])
        elif len(date_s) == 6:
            y, mo, d = 2000 + int(date_s[:2]), int(date_s[2:4]), int(date_s[4:6])
        else:
            continue
        try:
            return datetime(
                y, mo, d, int(time_s[:2]), int(time_s[2:4]), int(time_s[4:6]), tzinfo=timezone.utc
            )
        except ValueError:
            continue
    return None


def _window_abs_times(
    wav_start: datetime | None, start_sample: Any, end_sample: Any, sample_freq: Any
) -> tuple[str | None, str | None]:
    """Absolute ISO times of a window: WAV start + (1-based sample - 1) / sample_freq."""
    if wav_start is None or not sample_freq or start_sample is None or end_sample is None:
        return None, None
    freq = float(sample_freq)
    start = wav_start + timedelta(seconds=(float(start_sample) - 1) / freq)
    end = wav_start + timedelta(seconds=float(end_sample) / freq)
    return start.isoformat(), end.isoformat()


def _normalize_user_label(raw: str | None) -> str | None:
    """Empty or whitespace labels become NULL (unnamed)."""
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def is_named(label: Any) -> bool:
    """True when a user alias is set (the automatic ``name`` does not count)."""
    return label is not None and bool(str(label).strip())


def csv_stem(path: Path | str) -> str:
    """WAV stem of a song-structure CSV: ``<stem>.motif_segments.csv`` → ``<stem>``."""
    name = Path(path).name
    if name.endswith(MOTIF_CSV_SUFFIX):
        return name[: -len(MOTIF_CSV_SUFFIX)]
    return Path(name).stem


def connect(db_path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the catalog, applying the schema on write connections.

    ``read_only=True`` opens the file through a ``mode=ro`` URI, so it is safe
    to query while another process writes. It raises ``FileNotFoundError`` when
    the file does not exist yet.
    """
    path = Path(db_path)
    if read_only:
        if not path.exists():
            raise FileNotFoundError(f"catalog db not found: {path}")
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())
    return conn


def _one(conn: sqlite3.Connection, sql: str, args: Sequence[Any] = ()) -> dict[str, Any] | None:
    row = conn.execute(sql, args).fetchone()
    return dict(row) if row else None


def _all(conn: sqlite3.Connection, sql: str, args: Sequence[Any] = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


# ── WAV files ────────────────────────────────────────────────────────────────


def upsert_wav_file(
    conn: sqlite3.Connection, *, library_root: Path | str, abs_path: Path | str
) -> int:
    """Insert or update a WAV row; returns its ``wav_id``."""
    root = Path(library_root).resolve()
    wav = Path(abs_path).resolve()
    rel_path = str(wav.relative_to(root)) if wav.is_relative_to(root) else wav.name
    stat = wav.stat() if wav.exists() else None
    conn.execute(
        """
        INSERT INTO wav_files (library_root, rel_path, abs_path, stem, size_bytes, mtime, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(abs_path) DO UPDATE SET
            library_root = excluded.library_root,
            rel_path     = excluded.rel_path,
            stem         = excluded.stem,
            size_bytes   = excluded.size_bytes,
            mtime        = excluded.mtime,
            indexed_at   = excluded.indexed_at
        """,
        (
            str(root),
            rel_path,
            str(wav),
            wav.stem,
            stat.st_size if stat else None,
            _mtime_iso(wav) if stat else None,
            _now(),
        ),
    )
    return int(_one(conn, "SELECT wav_id FROM wav_files WHERE abs_path = ?", (str(wav),))["wav_id"])


def list_wav_files(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _all(conn, "SELECT * FROM wav_files ORDER BY stem ASC, wav_id ASC")


# ── GPL segmentations and windows ────────────────────────────────────────────


def upsert_gpl_segmentation(
    conn: sqlite3.Connection,
    *,
    wav_id: int | None,
    output_dir: Path | str,
) -> int:
    """Insert or update one GPL output folder; returns its ``segmentation_id``.

    ``npy_dir``, the detection CSV and ``panel.bin`` are found inside
    ``output_dir``. A known ``wav_id`` is never replaced by NULL.
    """
    out = Path(output_dir).resolve()
    npy_dir = out / "spectrogram_npy"
    csvs = sorted(out.glob("CSV_*.csv"))
    panel = out / "panel.bin"
    conn.execute(
        """
        INSERT INTO gpl_segmentations (wav_id, output_dir, npy_dir, csv_path, panel_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(output_dir) DO UPDATE SET
            wav_id     = COALESCE(excluded.wav_id, gpl_segmentations.wav_id),
            npy_dir    = excluded.npy_dir,
            csv_path   = excluded.csv_path,
            panel_path = COALESCE(excluded.panel_path, gpl_segmentations.panel_path)
        """,
        (
            wav_id,
            str(out),
            str(npy_dir) if npy_dir.is_dir() else None,
            str(csvs[0]) if csvs else None,
            str(panel) if panel.is_file() else None,
            _now(),
        ),
    )
    row = _one(conn, "SELECT segmentation_id FROM gpl_segmentations WHERE output_dir = ?", (str(out),))
    return int(row["segmentation_id"])


def list_gpl_segmentations_for_wav(conn: sqlite3.Connection, wav_id: int) -> list[dict[str, Any]]:
    return _all(
        conn,
        """
        SELECT g.*, (SELECT COUNT(*) FROM windows w WHERE w.segmentation_id = g.segmentation_id) AS n_windows
        FROM gpl_segmentations g WHERE g.wav_id = ? ORDER BY g.segmentation_id ASC
        """,
        (wav_id,),
    )


def upsert_windows_from_dir(
    conn: sqlite3.Connection, *, segmentation_id: int, npy_dir: Path | str
) -> int:
    """Index window metadata from the ``*.json`` sidecars; returns the count.

    Arrays are not loaded. ``*_display.npy`` files and arrays without a sidecar
    are skipped, as the dict_learning loaders do.
    """
    wav = _one(
        conn,
        """
        SELECT w.wav_id, w.stem FROM gpl_segmentations g
        JOIN wav_files w ON w.wav_id = g.wav_id WHERE g.segmentation_id = ?
        """,
        (segmentation_id,),
    )
    wav_id = wav["wav_id"] if wav else None
    wav_start = wav_start_datetime(wav["stem"]) if wav else None
    count = 0
    for npy_path in sorted(Path(npy_dir).glob("*.npy")):
        json_path = npy_path.with_suffix(".json")
        if npy_path.name.endswith("_display.npy") or not json_path.is_file():
            continue
        meta = json.loads(json_path.read_text())
        window_id = npy_path.stem
        parts = window_id_parts(window_id)
        abs_start, abs_end = _window_abs_times(
            wav_start, meta.get("start_sample"), meta.get("end_sample"), meta.get("sample_freq")
        )
        conn.execute(
            """
            INSERT INTO windows (
                window_id, segmentation_id, wav_id, window_index, npy_path, json_path,
                start_sample, end_sample, abs_start_time, abs_end_time,
                overlap, fftl, sample_freq, start_freq, end_freq, n_freq, n_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(window_id) DO UPDATE SET
                segmentation_id = excluded.segmentation_id,
                wav_id          = COALESCE(excluded.wav_id, windows.wav_id),
                window_index    = excluded.window_index,
                npy_path        = excluded.npy_path,
                json_path       = excluded.json_path,
                start_sample    = excluded.start_sample,
                end_sample      = excluded.end_sample,
                abs_start_time  = COALESCE(excluded.abs_start_time, windows.abs_start_time),
                abs_end_time    = COALESCE(excluded.abs_end_time, windows.abs_end_time),
                overlap         = excluded.overlap,
                fftl            = excluded.fftl,
                sample_freq     = excluded.sample_freq,
                start_freq      = excluded.start_freq,
                end_freq        = excluded.end_freq,
                n_freq          = excluded.n_freq,
                n_time          = excluded.n_time
            """,
            (
                window_id,
                segmentation_id,
                wav_id,
                parts[1] if parts else None,
                str(npy_path.resolve()),
                str(json_path.resolve()),
                meta.get("start_sample"),
                meta.get("end_sample"),
                abs_start,
                abs_end,
                meta.get("overlap"),
                meta.get("fftl"),
                meta.get("sample_freq"),
                meta.get("start_freq"),
                meta.get("end_freq"),
                meta.get("rows"),
                meta.get("cols"),
            ),
        )
        count += 1
    return count


def list_windows_for_segmentation(
    conn: sqlite3.Connection, segmentation_id: int
) -> list[dict[str, Any]]:
    return _all(
        conn,
        "SELECT * FROM windows WHERE segmentation_id = ? ORDER BY start_sample ASC, window_id ASC",
        (segmentation_id,),
    )


def index_gpl_dir(conn: sqlite3.Connection, gpl_dir: Path | str, *, wav_id: int | None = None) -> int:
    """Index a GPL output folder and its windows; returns the ``segmentation_id``.

    Without ``wav_id``, the WAV is looked up by the stem in the window ids
    (only when exactly one indexed WAV has that stem).
    """
    out = Path(gpl_dir).resolve()
    npy_dir = out / "spectrogram_npy"
    if not npy_dir.is_dir():
        npy_dir = out
    if wav_id is None:
        stem = _stem_from_npy_dir(npy_dir)
        if stem is not None:
            rows = _all(conn, "SELECT wav_id FROM wav_files WHERE stem = ?", (stem,))
            if len(rows) == 1:
                wav_id = int(rows[0]["wav_id"])
    segmentation_id = upsert_gpl_segmentation(conn, wav_id=wav_id, output_dir=out)
    upsert_windows_from_dir(conn, segmentation_id=segmentation_id, npy_dir=npy_dir)
    return segmentation_id


def _stem_from_npy_dir(npy_dir: Path) -> str | None:
    for json_path in sorted(npy_dir.glob("spwhiten_*.json")):
        parts = window_id_parts(json_path.stem)
        if parts is not None:
            return parts[0]
    return None


# ── learning runs, templates, tool runs ──────────────────────────────────────


def list_learning_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _all(conn, "SELECT * FROM learning_runs ORDER BY run_id DESC")


def get_learning_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    return _one(conn, "SELECT * FROM learning_runs WHERE run_id = ?", (run_id,))


def list_learning_round_logs(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    return _all(
        conn, "SELECT * FROM learning_round_logs WHERE run_id = ? ORDER BY round_index ASC", (run_id,)
    )


def list_templates(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    return _all(conn, "SELECT * FROM templates WHERE run_id = ? ORDER BY template_id ASC", (run_id,))


def get_template(conn: sqlite3.Connection, run_id: int, template_id: int) -> dict[str, Any] | None:
    return _one(
        conn, "SELECT * FROM templates WHERE run_id = ? AND template_id = ?", (run_id, template_id)
    )


def list_tool_runs(
    conn: sqlite3.Connection, *, learning_run_id: int | None = None
) -> list[dict[str, Any]]:
    if learning_run_id is None:
        return _all(conn, "SELECT * FROM tool_runs ORDER BY tool_run_id ASC")
    return _all(
        conn,
        "SELECT * FROM tool_runs WHERE learning_run_id = ? ORDER BY tool_run_id ASC",
        (learning_run_id,),
    )


def get_tool_run(conn: sqlite3.Connection, tool_run_id: int) -> dict[str, Any] | None:
    return _one(conn, "SELECT * FROM tool_runs WHERE tool_run_id = ?", (tool_run_id,))


def _upsert_tool_run(conn: sqlite3.Connection, *, tool: str, out_dir: str, **fields: Any) -> int:
    if tool not in TOOL_NAMES:  # no-pipeline-cover: registration passes a fixed tool name
        raise ValueError(f"tool must be one of {TOOL_NAMES}, got {tool!r}")
    columns = ["tool", "out_dir", *fields, "created_at"]
    values = [tool, out_dir, *fields.values(), _now()]
    updates = ", ".join(f"{c} = excluded.{c}" for c in ["tool", *fields])
    conn.execute(
        f"""
        INSERT INTO tool_runs ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})
        ON CONFLICT(out_dir) DO UPDATE SET {updates}
        """,
        values,
    )
    return int(_one(conn, "SELECT tool_run_id FROM tool_runs WHERE out_dir = ?", (out_dir,))["tool_run_id"])


# ── dictionaries ─────────────────────────────────────────────────────────────

_DICTIONARY_COUNTS = """
    (SELECT COUNT(*) FROM dictionary_templates dt WHERE dt.dictionary_id = d.dictionary_id) AS n_templates,
    (SELECT COUNT(*) FROM dictionary_windows dw WHERE dw.dictionary_id = d.dictionary_id) AS n_windows,
    (SELECT COUNT(*) FROM matchings m WHERE m.dictionary_id = d.dictionary_id) AS n_matchings
"""


def _upsert_dictionary(conn: sqlite3.Connection, *, out_dir: str, **fields: Any) -> int:
    """Insert or update the dictionary registered from ``out_dir``; returns its id.

    Re-registering a folder keeps its ``dictionary_id`` and user label.
    """
    source = fields["source"]
    if source not in DICT_SOURCES:  # no-pipeline-cover: registration passes a fixed source
        raise ValueError(f"source must be one of {DICT_SOURCES}, got {source!r}")
    if (source == "seed") != (fields.get("parent_dictionary_id") is None):  # no-pipeline-cover: as above
        raise ValueError("seed dictionaries have no parent; refinement/matching/edit need one")
    existing = _one(conn, "SELECT dictionary_id FROM dictionaries WHERE out_dir = ?", (out_dir,))
    if existing is None:
        columns = ["out_dir", *fields]
        conn.execute(
            f"INSERT INTO dictionaries ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
            [out_dir, *fields.values()],
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    dictionary_id = int(existing["dictionary_id"])
    assignments = ", ".join(f"{c} = ?" for c in fields)
    conn.execute(
        f"UPDATE dictionaries SET {assignments} WHERE dictionary_id = ?",
        [*fields.values(), dictionary_id],
    )
    return dictionary_id


def set_dictionary_templates(
    conn: sqlite3.Connection, dictionary_id: int, template_pks: Sequence[int]
) -> int:
    """Replace a dictionary's members (list order = position); keeps labels of kept members."""
    labels = {
        int(r["template_pk"]): r["label"]
        for r in _all(
            conn,
            "SELECT template_pk, label FROM dictionary_templates WHERE dictionary_id = ?",
            (dictionary_id,),
        )
    }
    conn.execute("DELETE FROM dictionary_templates WHERE dictionary_id = ?", (dictionary_id,))
    conn.executemany(
        "INSERT INTO dictionary_templates (dictionary_id, template_pk, position, label) VALUES (?, ?, ?, ?)",
        [(dictionary_id, int(pk), pos, labels.get(int(pk))) for pos, pk in enumerate(template_pks)],
    )
    return len(template_pks)


def set_dictionary_windows(
    conn: sqlite3.Connection, dictionary_id: int, window_ids: Sequence[str]
) -> int:
    """Replace the set of windows a dictionary was matched over; returns the count."""
    conn.execute("DELETE FROM dictionary_windows WHERE dictionary_id = ?", (dictionary_id,))
    unique = list(dict.fromkeys(w for w in window_ids if w))
    conn.executemany(
        "INSERT INTO dictionary_windows (dictionary_id, window_id) VALUES (?, ?)",
        [(dictionary_id, w) for w in unique],
    )
    return len(unique)


def list_dictionaries(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """All dictionaries, newest first, with template, window and placement counts."""
    return _all(conn, f"SELECT d.*, {_DICTIONARY_COUNTS} FROM dictionaries d ORDER BY d.dictionary_id DESC")


def list_dictionaries_for_stem(conn: sqlite3.Connection, stem: str) -> list[dict[str, Any]]:
    """Dictionaries of one WAV (by ``wav_stem``, or through an indexed segmentation)."""
    return _all(
        conn,
        f"""
        SELECT d.*, {_DICTIONARY_COUNTS} FROM dictionaries d
        LEFT JOIN gpl_segmentations g ON g.segmentation_id = d.segmentation_id
        LEFT JOIN wav_files w ON w.wav_id = g.wav_id
        WHERE d.wav_stem = ? OR w.stem = ?
        ORDER BY d.dictionary_id DESC
        """,
        (stem, stem),
    )


def get_dictionary(conn: sqlite3.Connection, dictionary_id: int) -> dict[str, Any] | None:
    return _one(
        conn, f"SELECT d.*, {_DICTIONARY_COUNTS} FROM dictionaries d WHERE d.dictionary_id = ?", (dictionary_id,)
    )


def set_dictionary_label(
    conn: sqlite3.Connection, dictionary_id: int, label: str | None
) -> dict[str, Any]:
    """Set or clear the user alias on a dictionary; returns the updated row."""
    cur = conn.execute(
        "UPDATE dictionaries SET label = ? WHERE dictionary_id = ?",
        (_normalize_user_label(label), dictionary_id),
    )
    if cur.rowcount == 0:
        raise KeyError(f"dictionary {dictionary_id} not found")
    return get_dictionary(conn, dictionary_id)  # type: ignore[return-value]


def list_dictionary_templates(conn: sqlite3.Connection, dictionary_id: int) -> list[dict[str, Any]]:
    """A dictionary's atoms in position order, with the membership label."""
    return _all(
        conn,
        """
        SELECT dt.position, dt.label AS label, t.*
        FROM dictionary_templates dt JOIN templates t ON t.template_pk = dt.template_pk
        WHERE dt.dictionary_id = ? ORDER BY dt.position ASC
        """,
        (dictionary_id,),
    )


def set_dictionary_template_label(
    conn: sqlite3.Connection, dictionary_id: int, template_id: int, label: str | None
) -> dict[str, Any]:
    """Set or clear the user alias of atom ``template_id`` within one dictionary."""
    member = next(
        (m for m in list_dictionary_templates(conn, dictionary_id) if int(m["template_id"]) == template_id),
        None,
    )
    if member is None:
        raise KeyError(f"dictionary {dictionary_id} has no atom {template_id}")
    stored = _normalize_user_label(label)
    conn.execute(
        "UPDATE dictionary_templates SET label = ? WHERE dictionary_id = ? AND template_pk = ?",
        (stored, dictionary_id, int(member["template_pk"])),
    )
    member["label"] = stored
    return member


def list_dictionary_windows(conn: sqlite3.Connection, dictionary_id: int) -> list[dict[str, Any]]:
    """The windows a dictionary's MP ran over; un-indexed ids come back with NULL metadata."""
    return _all(
        conn,
        """
        SELECT dw.window_id, w.segmentation_id, w.wav_id, w.window_index,
               w.abs_start_time, w.abs_end_time, w.start_sample, w.end_sample, w.npy_path
        FROM dictionary_windows dw LEFT JOIN windows w ON w.window_id = dw.window_id
        WHERE dw.dictionary_id = ?
        ORDER BY w.start_sample, dw.window_id
        """,
        (dictionary_id,),
    )


def get_dictionary_lineage(conn: sqlite3.Connection, dictionary_id: int) -> list[dict[str, Any]]:
    """The ancestry chain, root seed first and this dictionary last."""
    chain: list[dict[str, Any]] = []
    current: int | None = dictionary_id
    while current is not None:
        row = get_dictionary(conn, current)
        if row is None:  # no-pipeline-cover: the API checks the id first, and parents cascade
            break
        chain.append(row)
        current = row["parent_dictionary_id"]
    chain.reverse()
    return chain


# ── matchings and feedback ───────────────────────────────────────────────────


def _feedback_clause(feedback: str | None, args: list[Any]) -> str | None:
    if feedback is None:
        return None
    if feedback == "unlabeled":
        return "f.label IS NULL"
    if feedback not in FEEDBACK_LABELS:  # no-pipeline-cover: the API validates the filter
        raise ValueError(f"feedback filter must be good, bad or unlabeled, got {feedback!r}")
    args.append(feedback)
    return "f.label = ?"


def list_matchings(
    conn: sqlite3.Connection,
    dictionary_id: int,
    *,
    template_pk: int | None = None,
    window_id: str | None = None,
    feedback: str | None = None,
) -> list[dict[str, Any]]:
    """Placements of one dictionary in time order, each with its atom id and feedback."""
    clauses = ["m.dictionary_id = ?"]
    args: list[Any] = [dictionary_id]
    if template_pk is not None:
        clauses.append("m.template_pk = ?")
        args.append(template_pk)
    if window_id is not None:
        clauses.append("m.window_id = ?")
        args.append(window_id)
    fb = _feedback_clause(feedback, args)
    if fb is not None:
        clauses.append(fb)
    return _all(
        conn,
        f"""
        SELECT m.*, t.template_id, dt.label AS template_label,
               f.label AS feedback, f.note AS feedback_note, f.updated_at AS feedback_at
        FROM matchings m
        JOIN templates t ON t.template_pk = m.template_pk
        LEFT JOIN dictionary_templates dt
            ON dt.dictionary_id = m.dictionary_id AND dt.template_pk = m.template_pk
        LEFT JOIN matching_feedback f ON f.matching_id = m.id
        WHERE {' AND '.join(clauses)}
        ORDER BY m.start_sec IS NULL, m.start_sec, m.id
        """,
        args,
    )


def get_matching(conn: sqlite3.Connection, matching_id: int) -> dict[str, Any] | None:
    return _one(
        conn,
        """
        SELECT m.*, f.label AS feedback, f.note AS feedback_note, f.updated_at AS feedback_at
        FROM matchings m LEFT JOIN matching_feedback f ON f.matching_id = m.id
        WHERE m.id = ?
        """,
        (matching_id,),
    )


def set_matching_feedback(
    conn: sqlite3.Connection, matching_id: int, label: str, *, note: str | None = None
) -> dict[str, Any]:
    """Record or replace the good/bad verdict on one placement."""
    if label not in FEEDBACK_LABELS:  # no-pipeline-cover: the API validates the label
        raise ValueError(f"label must be one of {FEEDBACK_LABELS}, got {label!r}")
    if get_matching(conn, matching_id) is None:
        raise KeyError(f"matching {matching_id} not found")
    now = _now()
    conn.execute(
        """
        INSERT INTO matching_feedback (matching_id, label, note, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(matching_id) DO UPDATE SET
            label = excluded.label, note = excluded.note, updated_at = excluded.updated_at
        """,
        (matching_id, label, note, now, now),
    )
    return get_matching(conn, matching_id)  # type: ignore[return-value]


def delete_matching_feedback(conn: sqlite3.Connection, matching_id: int) -> bool:
    """Clear the verdict on one placement; returns whether one existed."""
    return conn.execute("DELETE FROM matching_feedback WHERE matching_id = ?", (matching_id,)).rowcount > 0


def _window_timing(conn: sqlite3.Connection, window_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    if not window_ids:
        return {}
    unique = list(dict.fromkeys(window_ids))
    placeholders = ", ".join("?" * len(unique))
    rows = _all(
        conn,
        f"SELECT window_id, start_sample, sample_freq, overlap FROM windows WHERE window_id IN ({placeholders})",
        unique,
    )
    return {r["window_id"]: r for r in rows}


def _replace_matchings(
    conn: sqlite3.Connection,
    dictionary_id: int,
    rows: Sequence[dict[str, Any]],
    template_pks: dict[int, int],
    widths: dict[int, int],
) -> int:
    """Replace a dictionary's placements with ``matchings.json`` rows."""
    conn.execute("DELETE FROM matchings WHERE dictionary_id = ?", (dictionary_id,))
    timing = _window_timing(conn, [r["window_id"] for r in rows])
    values = []
    for row in rows:
        template_id = int(row["template_id"])
        if template_id not in template_pks:
            raise CatalogError(f"matchings.json places atom {template_id}, which the dictionary lacks")
        start_sec = end_sec = None
        window = timing.get(row["window_id"])
        if window and window["sample_freq"] and window["overlap"] and window["start_sample"] is not None:
            hop_sec = float(window["overlap"]) / float(window["sample_freq"])
            start_sec = (float(window["start_sample"]) - 1) / float(window["sample_freq"]) + row["t_start"] * hop_sec
            end_sec = start_sec + widths.get(template_id, 1) * hop_sec
        values.append(
            (
                dictionary_id,
                template_pks[template_id],
                row["window_id"],
                int(row["t_start"]),
                float(row["amplitude"]),
                row.get("energy"),
                _json_or_none(row.get("coeffs")),
                start_sec,
                end_sec,
            )
        )
    conn.executemany(
        """
        INSERT INTO matchings (dictionary_id, template_pk, window_id, t_start, amplitude,
                               energy, coeffs_json, start_sec, end_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    return len(values)


# ── registering CLI output folders ───────────────────────────────────────────


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise CatalogError(f"missing {path.name} in {path.parent}") from None
    except json.JSONDecodeError as exc:
        raise CatalogError(f"{path} is not valid JSON: {exc}") from None


def _resolve_recorded_path(raw: str | None, out_dir: Path) -> Path | None:
    """A path written into ``summary.json``: absolute, or relative to the cwd or the run folder."""
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (out_dir / path).resolve()


def _window_ids(summary: dict[str, Any], matchings: Sequence[dict[str, Any]]) -> list[str]:
    return list(summary.get("window_ids") or dict.fromkeys(m["window_id"] for m in matchings))


def _wav_stem(conn: sqlite3.Connection, segmentation_id: int | None, window_ids: Sequence[str]) -> str | None:
    if segmentation_id is not None:
        row = _one(
            conn,
            """
            SELECT w.stem FROM gpl_segmentations g JOIN wav_files w ON w.wav_id = g.wav_id
            WHERE g.segmentation_id = ?
            """,
            (segmentation_id,),
        )
        if row is not None:
            return row["stem"]
    for window_id in window_ids:
        parts = window_id_parts(window_id)
        if parts is not None:
            return parts[0]
    return None


def register_run_dir(
    conn: sqlite3.Connection, out_dir: Path | str, *, label: str | None = None
) -> dict[str, Any]:
    """Register a ``randomized-build`` or ``matching-pursuit`` output folder.

    Idempotent: registering the same folder again updates its rows in place and
    keeps ids and user labels. The GPL folder named in ``summary.json`` is
    indexed too, so placements get times in seconds. A matching-pursuit run
    registers its dictionary folder first when that is not in the catalog yet.

    Returns ``{"kind", "dictionary_id", "run_id", "tool_run_id"}``. The caller
    commits.
    """
    out = Path(out_dir).resolve()
    if not out.is_dir():
        raise CatalogError(f"run folder not found: {out}")
    summary = _read_json(out / "summary.json")
    command = summary.get("command")
    if command == "randomized-build":
        return _register_randomized_build(conn, out, summary, label)
    if command == "matching-pursuit":
        return _register_matching_pursuit(conn, out, summary, label)
    raise CatalogError(f"{out / 'summary.json'} is not from randomized-build or matching-pursuit")


def _index_recorded_gpl_dir(conn: sqlite3.Connection, summary: dict[str, Any], out: Path) -> int | None:
    gpl_dir = _resolve_recorded_path(summary.get("gpl_dir"), out)
    if gpl_dir is None or not gpl_dir.is_dir():
        return None
    return index_gpl_dir(conn, gpl_dir)


def _register_randomized_build(
    conn: sqlite3.Connection, out: Path, summary: dict[str, Any], label: str | None
) -> dict[str, Any]:
    atoms = summary.get("atoms")
    if atoms is None:
        raise CatalogError(f"{out / 'summary.json'} has no atom list")
    matchings = _read_json(out / "matchings.json")
    logs_path = out / "outer_logs.jsonl"
    logs = [json.loads(line) for line in logs_path.read_text().splitlines() if line.strip()] if logs_path.is_file() else []
    segmentation_id = _index_recorded_gpl_dir(conn, summary, out)
    window_ids = _window_ids(summary, matchings)
    created_at = _mtime_iso(out / "summary.json")
    params = {k: summary.get(k) for k in _BUILD_PARAMS}

    conn.execute(
        """
        INSERT INTO learning_runs (
            label, out_dir, gpl_dir, n_windows, max_outer, n_outer_completed, n_atoms, n_matchings,
            energy_before, energy_after, time_coverage_fraction, params_json, status, stop_reason,
            created_at, registered_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?)
        ON CONFLICT(out_dir) DO UPDATE SET
            gpl_dir = excluded.gpl_dir, n_windows = excluded.n_windows,
            max_outer = excluded.max_outer, n_outer_completed = excluded.n_outer_completed,
            n_atoms = excluded.n_atoms, n_matchings = excluded.n_matchings,
            energy_before = excluded.energy_before, energy_after = excluded.energy_after,
            time_coverage_fraction = excluded.time_coverage_fraction,
            params_json = excluded.params_json, stop_reason = excluded.stop_reason,
            created_at = excluded.created_at, registered_at = excluded.registered_at,
            label = COALESCE(excluded.label, learning_runs.label)
        """,
        (
            label,
            str(out),
            summary.get("gpl_dir"),
            summary.get("n_windows"),
            summary.get("max_outer"),
            summary.get("n_outer_completed"),
            summary.get("n_atoms"),
            summary.get("n_matchings"),
            summary.get("energy_before"),
            summary.get("energy_after"),
            summary.get("time_coverage_fraction"),
            _json_or_none(params),
            summary.get("stop_reason"),
            created_at,
            _now(),
        ),
    )
    run_id = int(_one(conn, "SELECT run_id FROM learning_runs WHERE out_dir = ?", (str(out),))["run_id"])

    conn.execute("DELETE FROM learning_round_logs WHERE run_id = ?", (run_id,))
    log_columns = (
        "pool_size", "n_sampled", "n_atoms_this_pass", "n_matchings_this_pass", "n_pruned",
        "n_distance_filtered", "n_atoms", "n_matchings", "time_coverage_fraction",
        "window_coverage_fraction", "n_bins_covered", "n_bins_total", "n_windows_matched",
        "n_windows_total", "energy_before", "energy_after",
    )
    conn.executemany(
        f"""
        INSERT INTO learning_round_logs (run_id, round_index, {', '.join(log_columns)})
        VALUES ({', '.join('?' * (len(log_columns) + 2))})
        """,
        [(run_id, int(log["outer"]), *(log.get(c) for c in log_columns)) for log in logs],
    )

    tool_run_id = _upsert_tool_run(
        conn,
        tool="randomized_build",
        out_dir=str(out),
        segmentation_id=segmentation_id,
        learning_run_id=run_id,
        params_json=_json_or_none(params),
        energy_before=summary.get("energy_before"),
        energy_after=summary.get("energy_after"),
        stop_reason=summary.get("stop_reason"),
    )

    template_pks: dict[int, int] = {}
    widths: dict[int, int] = {}
    for atom in atoms:
        path = out / atom["path"]
        if not path.is_file():
            raise CatalogError(f"atom file missing: {path}")
        template_id = int(atom["template_id"])
        conn.execute(
            """
            INSERT INTO templates (run_id, tool_run_id, template_id, npy_path, n_freq, width,
                                   n_components, n_patches, source_segment_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, template_id) WHERE run_id IS NOT NULL DO UPDATE SET
                tool_run_id = excluded.tool_run_id, npy_path = excluded.npy_path,
                n_freq = excluded.n_freq, width = excluded.width,
                n_components = excluded.n_components, n_patches = excluded.n_patches,
                source_segment_id = excluded.source_segment_id
            """,
            (
                run_id,
                tool_run_id,
                template_id,
                str(path),
                atom.get("n_freq"),
                atom.get("width"),
                atom.get("n_components"),
                atom.get("n_patches"),
                atom.get("source_segment_id"),
            ),
        )
        template_pks[template_id] = int(get_template(conn, run_id, template_id)["template_pk"])
        widths[template_id] = int(atom.get("width") or 1)
    stale = [
        t["template_pk"] for t in list_templates(conn, run_id) if int(t["template_id"]) not in template_pks
    ]
    conn.executemany("DELETE FROM templates WHERE template_pk = ?", [(pk,) for pk in stale])

    dictionary_id = _upsert_dictionary(
        conn,
        out_dir=str(out),
        name=out.name,
        source="seed",
        parent_dictionary_id=None,
        learning_run_id=run_id,
        segmentation_id=segmentation_id,
        wav_stem=_wav_stem(conn, segmentation_id, window_ids),
        params_json=_json_or_none(params),
        energy_before=summary.get("energy_before"),
        energy_after=summary.get("energy_after"),
        time_coverage_fraction=summary.get("time_coverage_fraction"),
        stop_reason=summary.get("stop_reason"),
        created_at=created_at,
    )
    if label is not None:
        set_dictionary_label(conn, dictionary_id, label)
    set_dictionary_templates(conn, dictionary_id, list(template_pks.values()))
    set_dictionary_windows(conn, dictionary_id, window_ids)
    _replace_matchings(conn, dictionary_id, matchings, template_pks, widths)
    conn.execute(
        "UPDATE tool_runs SET output_dictionary_id = ? WHERE tool_run_id = ?", (dictionary_id, tool_run_id)
    )
    return {"kind": "randomized-build", "dictionary_id": dictionary_id, "run_id": run_id, "tool_run_id": tool_run_id}


def _register_matching_pursuit(
    conn: sqlite3.Connection, out: Path, summary: dict[str, Any], label: str | None
) -> dict[str, Any]:
    dictionary_dir = _resolve_recorded_path(summary.get("dictionary"), out)
    if dictionary_dir is None:
        raise CatalogError(f"{out / 'summary.json'} does not name its dictionary folder")
    parent = _one(conn, "SELECT dictionary_id FROM dictionaries WHERE out_dir = ?", (str(dictionary_dir),))
    if parent is None:
        if not (dictionary_dir / "summary.json").is_file():
            raise CatalogError(f"dictionary folder {dictionary_dir} is not in the catalog and has no summary.json")
        parent_id = register_run_dir(conn, dictionary_dir)["dictionary_id"]
    else:
        parent_id = int(parent["dictionary_id"])
    members = list_dictionary_templates(conn, parent_id)
    template_pks = {int(m["template_id"]): int(m["template_pk"]) for m in members}
    widths = {int(m["template_id"]): int(m["width"] or 1) for m in members}

    matchings = _read_json(out / "matchings.json")
    segmentation_id = _index_recorded_gpl_dir(conn, summary, out)
    window_ids = _window_ids(summary, matchings)
    params = {k: summary.get(k) for k in _MP_PARAMS}
    dictionary_id = _upsert_dictionary(
        conn,
        out_dir=str(out),
        name=out.name,
        source="matching",
        parent_dictionary_id=parent_id,
        learning_run_id=None,
        segmentation_id=segmentation_id,
        wav_stem=_wav_stem(conn, segmentation_id, window_ids),
        params_json=_json_or_none(params),
        energy_before=summary.get("energy_before"),
        energy_after=summary.get("energy_after"),
        time_coverage_fraction=summary.get("time_coverage_fraction"),
        stop_reason=summary.get("stop_reason"),
        created_at=_mtime_iso(out / "summary.json"),
    )
    if label is not None:
        set_dictionary_label(conn, dictionary_id, label)
    set_dictionary_templates(conn, dictionary_id, [int(m["template_pk"]) for m in members])
    set_dictionary_windows(conn, dictionary_id, window_ids)
    _replace_matchings(conn, dictionary_id, matchings, template_pks, widths)
    tool_run_id = _upsert_tool_run(
        conn,
        tool="matching_pursuit",
        out_dir=str(out),
        input_dictionary_id=parent_id,
        output_dictionary_id=dictionary_id,
        segmentation_id=segmentation_id,
        params_json=_json_or_none(params),
        energy_before=summary.get("energy_before"),
        energy_after=summary.get("energy_after"),
        stop_reason=summary.get("stop_reason"),
    )
    return {"kind": "matching-pursuit", "dictionary_id": dictionary_id, "run_id": None, "tool_run_id": tool_run_id}


# ── song-structure CSVs ──────────────────────────────────────────────────────


def read_motif_csv_counts(path: Path | str) -> tuple[int, int]:
    """``(rows, songs)`` of a song-structure CSV; raises CatalogError on missing columns."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = MOTIF_CSV_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise CatalogError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        n_rows = 0
        songs: set[str] = set()
        for row in reader:
            n_rows += 1
            songs.add(row["song"])
    return n_rows, len(songs)


def upsert_annotation_csv(conn: sqlite3.Connection, path: Path | str, *, label: str | None = None) -> int:
    """Index one song-structure CSV; returns its ``csv_id``. Re-indexing keeps the id and label."""
    csv_path = Path(path).resolve()
    if not csv_path.is_file():  # no-pipeline-cover: the CLI and indexer pass existing files
        raise CatalogError(f"CSV not found: {csv_path}")
    n_rows, n_songs = read_motif_csv_counts(csv_path)
    conn.execute(
        """
        INSERT INTO annotation_csvs (csv_path, name, label, wav_stem, n_rows, n_songs,
                                     size_bytes, created_at, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(csv_path) DO UPDATE SET
            name = excluded.name, wav_stem = excluded.wav_stem, n_rows = excluded.n_rows,
            n_songs = excluded.n_songs, size_bytes = excluded.size_bytes,
            created_at = excluded.created_at, indexed_at = excluded.indexed_at,
            label = COALESCE(excluded.label, annotation_csvs.label)
        """,
        (
            str(csv_path),
            csv_path.name,
            _normalize_user_label(label),
            csv_stem(csv_path),
            n_rows,
            n_songs,
            csv_path.stat().st_size,
            _mtime_iso(csv_path),
            _now(),
        ),
    )
    return int(_one(conn, "SELECT csv_id FROM annotation_csvs WHERE csv_path = ?", (str(csv_path),))["csv_id"])


def list_annotation_csvs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _all(conn, "SELECT * FROM annotation_csvs ORDER BY csv_id DESC")


def list_annotation_csvs_for_stem(conn: sqlite3.Connection, stem: str) -> list[dict[str, Any]]:
    return _all(conn, "SELECT * FROM annotation_csvs WHERE wav_stem = ? ORDER BY csv_id DESC", (stem,))


def get_annotation_csv(conn: sqlite3.Connection, csv_id: int) -> dict[str, Any] | None:
    return _one(conn, "SELECT * FROM annotation_csvs WHERE csv_id = ?", (csv_id,))


def set_annotation_csv_label(conn: sqlite3.Connection, csv_id: int, label: str | None) -> dict[str, Any]:
    cur = conn.execute(
        "UPDATE annotation_csvs SET label = ? WHERE csv_id = ?", (_normalize_user_label(label), csv_id)
    )
    if cur.rowcount == 0:
        raise KeyError(f"CSV {csv_id} not found")
    return get_annotation_csv(conn, csv_id)  # type: ignore[return-value]


def prune_missing_annotation_csvs(conn: sqlite3.Connection) -> int:
    """Drop CSV rows whose file is gone; returns how many."""
    gone = [r["csv_id"] for r in list_annotation_csvs(conn) if not Path(r["csv_path"]).is_file()]
    conn.executemany("DELETE FROM annotation_csvs WHERE csv_id = ?", [(i,) for i in gone])
    return len(gone)


# ── the picker ───────────────────────────────────────────────────────────────


def list_analyses_for_stem(conn: sqlite3.Connection, stem: str) -> dict[str, list[dict[str, Any]]]:
    """Everything the picker lists for one WAV: its dictionaries and its CSVs."""
    return {
        "dictionaries": list_dictionaries_for_stem(conn, stem),
        "csvs": list_annotation_csvs_for_stem(conn, stem),
    }
