"""Backend adapter over the ``database`` catalog package.

Reads open the catalog read-only (safe while a CLI run registers into the same
file); renames, feedback and reindex open it for writing. Paths come from
``app.config`` at call time.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app import config
from database import catalog, index_existing


def _read_conn():
    try:
        return catalog.connect(config.CATALOG_DB, read_only=True)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Catalog not built yet ({config.CATALOG_DB}); POST /api/catalog/reindex first.",
        ) from None


def _write_conn():
    return catalog.connect(config.CATALOG_DB)


def _found(row: dict[str, Any] | None, what: str) -> dict[str, Any]:
    if row is None:
        raise HTTPException(status_code=404, detail=f"{what} not found")
    return row


# ── picker rows ──────────────────────────────────────────────────────────────


def _label_fields(row: dict[str, Any]) -> dict[str, Any]:
    label = row.get("label")
    named = catalog.is_named(label)
    return {
        "label": str(label).strip() if named else None,
        "named": named,
        "displayName": str(label).strip() if named else row["name"],
    }


def dictionary_for_ui(row: dict[str, Any]) -> dict[str, Any]:
    """A catalog dictionary row in the camelCase shape the picker uses."""
    return {
        "kind": "dictionary",
        "id": int(row["dictionary_id"]),
        "name": row["name"],
        **_label_fields(row),
        "description": row.get("description"),
        "source": row["source"],
        "parentId": row.get("parent_dictionary_id"),
        "createdAt": row.get("created_at"),
        "nTemplates": row.get("n_templates") or 0,
        "nWindows": row.get("n_windows") or 0,
        "nMatchings": row.get("n_matchings") or 0,
        "timeCoverage": row.get("time_coverage_fraction"),
    }


def csv_for_ui(row: dict[str, Any]) -> dict[str, Any]:
    """A catalog CSV row in the camelCase shape the picker uses."""
    return {
        "kind": "csv",
        "id": int(row["csv_id"]),
        "name": row["name"],
        **_label_fields(row),
        "createdAt": row.get("created_at"),
        "nRows": row.get("n_rows") or 0,
        "nSongs": row.get("n_songs") or 0,
    }


def list_analyses_for_wav(wav_path: Path) -> dict[str, Any]:
    """The picker's list for one WAV: dictionaries and CSVs with its stem.

    An empty list (not a 404) when the catalog does not exist yet.
    """
    stem = wav_path.stem
    try:
        conn = catalog.connect(config.CATALOG_DB, read_only=True)
    except FileNotFoundError:
        return {"stem": stem, "items": []}
    with closing(conn):
        found = catalog.list_analyses_for_stem(conn, stem)
    items = [dictionary_for_ui(r) for r in found["dictionaries"]] + [csv_for_ui(r) for r in found["csvs"]]
    return {"stem": stem, "items": items}


# ── reads ────────────────────────────────────────────────────────────────────


def list_wavs() -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        return catalog.list_wav_files(conn)


def list_gpl_segmentations_for_wav(wav_id: int) -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        return catalog.list_gpl_segmentations_for_wav(conn, wav_id)


def list_runs() -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        return catalog.list_learning_runs(conn)


def list_templates(run_id: int) -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        _found(catalog.get_learning_run(conn, run_id), "Learning run")
        return catalog.list_templates(conn, run_id)


def list_round_logs(run_id: int) -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        _found(catalog.get_learning_run(conn, run_id), "Learning run")
        return catalog.list_learning_round_logs(conn, run_id)


def get_template(run_id: int, template_id: int) -> dict[str, Any]:
    with closing(_read_conn()) as conn:
        return _found(catalog.get_template(conn, run_id, template_id), "Template")


def get_template_npy(run_id: int, template_id: int) -> tuple[Path, str]:
    """The atom's ``.npy`` file and a download name."""
    path = Path(get_template(run_id, template_id)["npy_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Template file missing: {path}")
    return path, f"run{run_id}_template{template_id}.npy"


def list_dictionaries() -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        return catalog.list_dictionaries(conn)


def get_dictionary(dictionary_id: int) -> dict[str, Any]:
    with closing(_read_conn()) as conn:
        return _found(catalog.get_dictionary(conn, dictionary_id), "Dictionary")


def list_dictionary_templates(dictionary_id: int) -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        _found(catalog.get_dictionary(conn, dictionary_id), "Dictionary")
        return catalog.list_dictionary_templates(conn, dictionary_id)


def list_dictionary_windows(dictionary_id: int) -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        _found(catalog.get_dictionary(conn, dictionary_id), "Dictionary")
        return catalog.list_dictionary_windows(conn, dictionary_id)


def get_dictionary_lineage(dictionary_id: int) -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        _found(catalog.get_dictionary(conn, dictionary_id), "Dictionary")
        return catalog.get_dictionary_lineage(conn, dictionary_id)


def list_dictionary_matchings(
    dictionary_id: int,
    *,
    template_pk: int | None = None,
    window_id: str | None = None,
    feedback: str | None = None,
) -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        _found(catalog.get_dictionary(conn, dictionary_id), "Dictionary")
        return catalog.list_matchings(
            conn, dictionary_id, template_pk=template_pk, window_id=window_id, feedback=feedback
        )


def list_dictionary_placements(dictionary_id: int) -> dict[str, Any]:
    """Placements with times, for the viewer's atom lane."""
    with closing(_read_conn()) as conn:
        row = _found(catalog.get_dictionary(conn, dictionary_id), "Dictionary")
        matchings = catalog.list_matchings(conn, dictionary_id)
    placements = [
        {
            "matchingId": int(m["id"]),
            "templateId": int(m["template_id"]),
            "label": m["template_label"],
            "startSec": m["start_sec"],
            "endSec": m["end_sec"],
            "amplitude": m["amplitude"],
            "feedback": m["feedback"],
        }
        for m in matchings
        if m["start_sec"] is not None
    ]
    return {
        "dictionary": dictionary_for_ui(row),
        "placements": placements,
        "nWithoutTime": len(matchings) - len(placements),
    }


def get_matching(matching_id: int) -> dict[str, Any]:
    with closing(_read_conn()) as conn:
        return _found(catalog.get_matching(conn, matching_id), "Matching")


def list_tool_runs(*, learning_run_id: int | None = None) -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        return catalog.list_tool_runs(conn, learning_run_id=learning_run_id)


def get_tool_run(tool_run_id: int) -> dict[str, Any]:
    with closing(_read_conn()) as conn:
        return _found(catalog.get_tool_run(conn, tool_run_id), "Tool run")


def list_csvs() -> list[dict[str, Any]]:
    with closing(_read_conn()) as conn:
        return catalog.list_annotation_csvs(conn)


def get_csv(csv_id: int) -> dict[str, Any]:
    with closing(_read_conn()) as conn:
        return _found(catalog.get_annotation_csv(conn, csv_id), "CSV")


# ── writes ───────────────────────────────────────────────────────────────────


def set_dictionary_label(dictionary_id: int, label: str | None) -> dict[str, Any]:
    with closing(_write_conn()) as conn:
        try:
            row = catalog.set_dictionary_label(conn, dictionary_id, label)
        except KeyError:
            raise HTTPException(status_code=404, detail="Dictionary not found") from None
        conn.commit()
    return dictionary_for_ui(row)


def set_dictionary_template_label(dictionary_id: int, template_id: int, label: str | None) -> dict[str, Any]:
    with closing(_write_conn()) as conn:
        try:
            member = catalog.set_dictionary_template_label(conn, dictionary_id, template_id, label)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from None
        conn.commit()
    return {"dictionaryId": dictionary_id, "templateId": template_id, "label": member["label"]}


def set_csv_label(csv_id: int, label: str | None) -> dict[str, Any]:
    with closing(_write_conn()) as conn:
        try:
            row = catalog.set_annotation_csv_label(conn, csv_id, label)
        except KeyError:
            raise HTTPException(status_code=404, detail="CSV not found") from None
        conn.commit()
    return csv_for_ui(row)


def set_matching_feedback(matching_id: int, label: str, *, note: str | None = None) -> dict[str, Any]:
    with closing(_write_conn()) as conn:
        try:
            row = catalog.set_matching_feedback(conn, matching_id, label, note=note)
        except KeyError:
            raise HTTPException(status_code=404, detail="Matching not found") from None
        conn.commit()
    return row


def delete_matching_feedback(matching_id: int) -> dict[str, Any]:
    with closing(_write_conn()) as conn:
        _found(catalog.get_matching(conn, matching_id), "Matching")
        deleted = catalog.delete_matching_feedback(conn, matching_id)
        conn.commit()
    return {"matchingId": matching_id, "deleted": deleted}


def reindex() -> dict[str, Any]:
    return index_existing.reindex(
        config.CATALOG_DB,
        wav_roots=list(config.WAV_LIBRARY_ROOTS),
        runs_roots=list(config.CATALOG_RUNS_ROOTS),
        csv_roots=list(config.CATALOG_CSV_ROOTS),
        result_dirname=config.GPL_RESULT_DIRNAME,
    )
