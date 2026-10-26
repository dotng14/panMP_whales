"""Populate the catalog from what is already on disk.

- **WAV roots**: every ``*.wav`` → ``wav_files``; its GPL cache
  ``<root>/<GPL_RESULT_DIRNAME>/<rel>.gpl/output`` (as the backend's
  ``gpl_cache.py`` writes it) → ``gpl_segmentations`` + ``windows``.
- **Run roots**: every folder with a ``summary.json`` from ``randomized-build``
  or ``matching-pursuit`` → dictionaries, atoms, placements. Builds go first so
  matching-pursuit runs find their parent.
- **CSV roots**: every ``*.motif_segments.csv`` → ``annotation_csvs``. Rows
  whose file is gone are dropped.

Used by ``python -m database index`` and ``POST /api/catalog/reindex``.
Folders or files that cannot be registered are skipped and reported.
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Any

from . import catalog

RUN_COMMANDS = ("randomized-build", "matching-pursuit")


def gpl_output_dir(wav_path: Path, root: Path, result_dirname: str) -> Path:
    """Mirror ``backend/app/services/gpl_cache.py``: ``cache_dir_for(wav) / "output"``."""
    rel = wav_path.resolve().relative_to(root)
    return root / result_dirname / rel.with_suffix(".gpl") / "output"


def index_wav_root(conn, root: Path, *, result_dirname: str) -> dict[str, int]:
    root = root.resolve()
    counts = {"wavs": 0, "gpl_segmentations": 0, "windows": 0}
    if not root.is_dir():
        return counts
    result_root = root / result_dirname
    for wav_path in sorted(root.rglob("*.wav")):
        if wav_path.is_relative_to(result_root):
            continue
        wav_id = catalog.upsert_wav_file(conn, library_root=root, abs_path=wav_path)
        counts["wavs"] += 1
        output_dir = gpl_output_dir(wav_path, root, result_dirname)
        if not (output_dir / "spectrogram_npy").is_dir():
            continue
        segmentation_id = catalog.index_gpl_dir(conn, output_dir, wav_id=wav_id)
        counts["gpl_segmentations"] += 1
        counts["windows"] += len(catalog.list_windows_for_segmentation(conn, segmentation_id))
    return counts


def _run_dirs(runs_root: Path) -> list[Path]:
    """Run folders under ``runs_root``, builds before matching-pursuit runs."""
    found: list[tuple[int, Path]] = []
    for summary_path in sorted(runs_root.rglob("summary.json")):
        try:
            command = json.loads(summary_path.read_text()).get("command")
        except (json.JSONDecodeError, AttributeError):
            continue
        if command in RUN_COMMANDS:
            found.append((RUN_COMMANDS.index(command), summary_path.parent))
    return [path for _, path in sorted(found, key=lambda item: item[0])]


def index_runs_root(conn, runs_root: Path, skipped: list[str]) -> int:
    count = 0
    for run_dir in _run_dirs(runs_root.resolve()) if runs_root.is_dir() else []:
        try:
            with conn:  # one transaction per folder, so a bad folder leaves no half rows
                catalog.register_run_dir(conn, run_dir)
            count += 1
        except catalog.CatalogError as exc:
            skipped.append(f"{run_dir}: {exc}")
    return count


def index_csv_root(conn, csv_root: Path, skipped: list[str]) -> int:
    count = 0
    paths = sorted(csv_root.resolve().rglob(f"*{catalog.MOTIF_CSV_SUFFIX}")) if csv_root.is_dir() else []
    for path in paths:
        try:
            catalog.upsert_annotation_csv(conn, path)
            count += 1
        except catalog.CatalogError as exc:
            skipped.append(f"{path}: {exc}")
    return count


def reindex(
    db_path: Path | str,
    *,
    wav_roots: list[Path] | None = None,
    runs_roots: list[Path] | None = None,
    csv_roots: list[Path] | None = None,
    result_dirname: str = "result",
) -> dict[str, Any]:
    """Index every root and commit; returns counts and the skipped items."""
    totals: dict[str, Any] = {
        "wavs": 0, "gpl_segmentations": 0, "windows": 0, "runs": 0, "csvs": 0, "csvs_pruned": 0,
    }
    skipped: list[str] = []
    with closing(catalog.connect(db_path)) as conn:
        for root in wav_roots or []:
            for key, value in index_wav_root(conn, Path(root), result_dirname=result_dirname).items():
                totals[key] += value
        conn.commit()
        for root in runs_roots or []:
            totals["runs"] += index_runs_root(conn, Path(root), skipped)
        for root in csv_roots or []:
            totals["csvs"] += index_csv_root(conn, Path(root), skipped)
        totals["csvs_pruned"] = catalog.prune_missing_annotation_csvs(conn)
        conn.commit()
    totals["skipped"] = skipped
    return totals
