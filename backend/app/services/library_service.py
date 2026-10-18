from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from app.config import WAV_LIBRARY_ROOTS
from app.services.gpl_cache import is_gpl_cached


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
