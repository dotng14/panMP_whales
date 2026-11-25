from __future__ import annotations

import base64
import json
import logging
import os
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.config import CATALOG_DB, GPL_RESULT_DIRNAME, WAV_LIBRARY_ROOTS
from app.services.gpl_cache import is_gpl_cached
from app.services.wav_name import COMPONENT_KEYS
from database import catalog
from database.index_existing import index_wav_root

logger = logging.getLogger(__name__)
_index_lock = threading.Lock()
_catalog_ready = False


# ── fileId codec ──────────────────────────────────────────────────────────────

def encode_file_id(root_idx: int, rel: str) -> str:
    payload = json.dumps({"root": root_idx, "rel": rel}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode()


def resolve_file_id(file_id: str) -> Path:
    try:
        payload = json.loads(base64.urlsafe_b64decode(file_id + "=="))
        root_idx: int = int(payload["root"])
        rel: str = payload["rel"]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid fileId")

    if root_idx < 0 or root_idx >= len(WAV_LIBRARY_ROOTS):
        raise HTTPException(status_code=400, detail="Invalid root index in fileId")

    root = WAV_LIBRARY_ROOTS[root_idx].resolve()
    candidate = (root / rel).resolve()

    if not candidate.is_relative_to(root):
        raise HTTPException(status_code=403, detail="Path traversal detected")

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return candidate


# ── directory listing ─────────────────────────────────────────────────────────

def _entry_sort_key(entry: dict, sort: str) -> object:
    if sort == "mtime":
        return entry.get("mtime") or ""
    if sort == "size":
        return entry.get("sizeBytes") or 0
    return entry["name"].lower()


def list_entries(
    root_idx: int,
    rel_path: str,
    q: str,
    sort: str,
    order: str,
    offset: int,
    limit: int,
    recursive: bool,
) -> dict:
    if root_idx < 0 or root_idx >= len(WAV_LIBRARY_ROOTS):
        raise HTTPException(status_code=400, detail="Invalid root index")

    root = WAV_LIBRARY_ROOTS[root_idx].resolve()
    target = (root / rel_path).resolve() if rel_path else root

    if not target.is_relative_to(root):
        raise HTTPException(status_code=403, detail="Path traversal detected")
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    entries: list[dict] = []

    if q:
        # Recursive filename search
        q_lower = q.lower()
        glob = target.rglob("*.wav") if recursive else target.glob("*.wav")
        for p in glob:
            if q_lower in p.name.lower():
                stat = p.stat()
                rel = str(p.relative_to(root))
                entries.append({
                    "type": "file",
                    "name": p.name,
                    "fileId": encode_file_id(root_idx, rel),
                    "sizeBytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "cached": is_gpl_cached(p),
                })
    else:
        # Single-level listing: dirs first, then .wav files
        try:
            children = sorted(target.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")

        for p in children:
            if p.is_dir() and not p.name.startswith("."):
                entries.append({
                    "type": "dir",
                    "name": p.name,
                    "path": str(p.relative_to(root)),
                })
            elif p.is_file() and p.suffix.lower() == ".wav":
                stat = p.stat()
                rel = str(p.relative_to(root))
                entries.append({
                    "type": "file",
                    "name": p.name,
                    "fileId": encode_file_id(root_idx, rel),
                    "sizeBytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "cached": is_gpl_cached(p),
                })

    # Sort (dirs always before files when not searching)
    files = [e for e in entries if e["type"] == "file"]
    dirs = [e for e in entries if e["type"] == "dir"]
    files.sort(key=lambda e: _entry_sort_key(e, sort), reverse=(order == "desc"))
    dirs.sort(key=lambda e: e["name"].lower(), reverse=(order == "desc"))
    ordered = dirs + files if not q else files  # search returns only files

    total = len(ordered)
    page = ordered[offset: offset + limit]

    parent_path = str(Path(rel_path).parent) if rel_path and rel_path != "." else ""
    if parent_path == ".":
        parent_path = ""

    return {
        "root": root_idx,
        "path": rel_path,
        "parentPath": parent_path,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": page,
    }


# ── component facets / match ─────────────────────────────────────────────────


def _is_external_exfat_mount(path_s: str) -> bool:
    """External exFAT mounts (``/mnt/sda``, ``/mnt/sdb``, …) can hang on metadata I/O."""
    return path_s.startswith("/mnt/sd")


def _norm_root(root: Path | str) -> str:
    """Normalize a library root without ``stat``/``resolve`` on ``/mnt/sd*``."""
    p = Path(root)
    s = str(p)
    if p.is_absolute() and _is_external_exfat_mount(s):
        return os.path.normpath(s)
    return str(p.resolve())


def _root_index_for(library_root: str) -> int | None:
    target = _norm_root(library_root)
    for i, root in enumerate(WAV_LIBRARY_ROOTS):
        if _norm_root(root) == target:
            return i
    return None


def _resolved_root_strings() -> list[str]:
    return [_norm_root(r) for r in WAV_LIBRARY_ROOTS]


def ensure_wav_component_index(conn: Any) -> None:
    """Index library roots when empty or missing parsed components.

    Small roots index in-process. External ``/mnt/sd*`` Site H mounts are not
    auto-scanned (use ``backend/scripts/index_siteh.py``).
    """
    for root in WAV_LIBRARY_ROOTS:
        # Resolve / skip Site H before any exists/is_dir — those can hang on /mnt/sd*.
        try:
            root_s = str(root if root.is_absolute() else root.resolve())
        except OSError:
            logger.warning("Skipping library root (resolve failed): %s", root)
            continue
        if _is_external_exfat_mount(root_s):
            logger.info("Skipping auto-index for %s (run backend/scripts/index_siteh.py)", root_s)
            continue
        try:
            if not root.is_dir():
                continue
            root_s = str(root.resolve())
        except OSError:
            logger.warning("Skipping library root (stat failed): %s", root)
            continue
        total = int(
            conn.execute(
                "SELECT COUNT(*) FROM wav_files WHERE library_root = ?", (root_s,)
            ).fetchone()[0]
        )
        unparsed = int(
            conn.execute(
                "SELECT COUNT(*) FROM wav_files WHERE library_root = ? AND family IS NULL",
                (root_s,),
            ).fetchone()[0]
        )
        if total > 0 and unparsed == 0:
            continue
        index_wav_root(conn, root, result_dirname=GPL_RESULT_DIRNAME)


def ensure_catalog_ready() -> None:
    """One-time write-side ensure so parallel facets/match can stay read-only."""
    global _catalog_ready
    if _catalog_ready:
        return
    with _index_lock:
        if _catalog_ready:
            return
        with closing(catalog.connect(CATALOG_DB)) as conn:
            ensure_wav_component_index(conn)
            conn.commit()
        _catalog_ready = True


def open_catalog_read():
    """Read-only catalog connection after a one-time ensure (safe for parallel GETs)."""
    ensure_catalog_ready()
    return closing(catalog.connect(CATALOG_DB, read_only=True))


def match_recordings(
    conn: Any,
    *,
    filters: dict[str, str | None],
    q: str = "",
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Paginated WAV matches with ``fileId`` + GPL cache flag."""
    roots = _resolved_root_strings()
    rows, total = catalog.match_wav_files(
        conn,
        filters=filters,
        q=q,
        library_roots=roots,
        offset=offset,
        limit=limit,
    )
    entries: list[dict] = []
    for row in rows:
        root_idx = _root_index_for(row["library_root"])
        if root_idx is None:
            continue
        abs_path = Path(row["abs_path"])
        # For /mnt/sd* roots, is_gpl_cached checks only the local SSD fallback
        # cache dir (zero I/O on the external mount); never call abs_path.is_file().
        if _is_external_exfat_mount(str(abs_path)):
            try:
                cached = is_gpl_cached(abs_path)
            except OSError:
                cached = False
        else:
            try:
                cached = is_gpl_cached(abs_path) if abs_path.is_file() else False
            except OSError:
                cached = False
        entries.append({
            "type": "file",
            "name": abs_path.name,
            "fileId": encode_file_id(root_idx, row["rel_path"]),
            "sizeBytes": row.get("size_bytes"),
            "mtime": row.get("mtime"),
            "cached": cached,
            "stem": row.get("stem"),
            "startTime": row.get("start_time"),
            "components": {k: row.get(k) for k in COMPONENT_KEYS},
        })
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
    }


def recording_facets(
    conn: Any,
    *,
    filters: dict[str, str | None],
    q: str = "",
) -> dict:
    facets = catalog.wav_facets(
        conn,
        filters=filters,
        q=q,
        library_roots=_resolved_root_strings(),
        keys=COMPONENT_KEYS,
    )
    # Drop empty-string family sentinel used for non-SOCAL examined rows.
    if "family" in facets:
        facets["family"] = [v for v in facets["family"] if v.get("value") not in ("", None)]
    return {"facets": facets}
