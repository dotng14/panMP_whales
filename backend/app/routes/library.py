from fastapi import APIRouter, Query

from app.config import WAV_LIBRARY_ROOTS
from app.services import catalog_service
from app.services.gpl_cache import is_gpl_cached
from app.services.library_service import (
    list_entries,
    match_recordings,
    open_catalog_read,
    recording_facets,
    resolve_file_id,
)

router = APIRouter()


def _component_filters(
    family: str | None,
    site: str | None,
    deployment: str | None,
    disk: str | None,
    date: str | None,
    time: str | None,
    df: str | None,
    channel: str | None,
    resample: str | None,
    start: str | None = None,
) -> dict[str, str | None]:
    raw = {
        "family": family,
        "site": site,
        "deployment": deployment,
        "disk": disk,
        "date": date,
        "time": time,
        "df": df,
        "channel": channel,
        "resample": resample,
        "start": start,
    }
    return {k: v for k, v in raw.items() if v}


@router.get("/api/library/roots")
async def get_roots() -> dict:
    return {
        "roots": [
            {"id": i, "label": root.name}
            for i, root in enumerate(WAV_LIBRARY_ROOTS)
        ]
    }


@router.get("/api/library/analyses")
def get_file_analyses(fileId: str = Query(...)) -> dict:
    """Dictionaries and song-structure CSVs in the catalog for one library WAV (the picker).

    ``gplCached`` says whether its GPL spectrogram already exists (view) or must be run.
    """
    wav_path = resolve_file_id(fileId)
    return {**catalog_service.list_analyses_for_wav(wav_path), "gplCached": is_gpl_cached(wav_path)}


@router.get("/api/library/browse")
def browse(
    root: int = Query(0),
    path: str = Query(""),
    q: str = Query(""),
    sort: str = Query("name"),
    order: str = Query("asc"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    recursive: bool = Query(False),
) -> dict:
    return list_entries(
        root_idx=root,
        rel_path=path,
        q=q,
        sort=sort,
        order=order,
        offset=offset,
        limit=limit,
        recursive=recursive,
    )


@router.get("/api/library/facets")
def facets(
    q: str = Query(""),
    family: str | None = Query(None),
    site: str | None = Query(None),
    deployment: str | None = Query(None),
    disk: str | None = Query(None),
    date: str | None = Query(None),
    time: str | None = Query(None),
    df: str | None = Query(None),
    channel: str | None = Query(None),
    resample: str | None = Query(None),
    start: str | None = Query(None, description="Wall-clock start YYYY-MM-DD[ HH:MM:SS] (prefix OK)"),
) -> dict:
    """Distinct component values (+ counts) given the other filled filters."""
    filters = _component_filters(
        family, site, deployment, disk, date, time, df, channel, resample, start
    )
    with open_catalog_read() as conn:
        return recording_facets(conn, filters=filters, q=q)


@router.get("/api/library/match")
def match(
    q: str = Query(""),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    family: str | None = Query(None),
    site: str | None = Query(None),
    deployment: str | None = Query(None),
    disk: str | None = Query(None),
    date: str | None = Query(None),
    time: str | None = Query(None),
    df: str | None = Query(None),
    channel: str | None = Query(None),
    resample: str | None = Query(None),
    start: str | None = Query(None, description="Wall-clock start YYYY-MM-DD[ HH:MM:SS] (prefix OK)"),
) -> dict:
    """WAV files matching site/start or component filters (catalog only; no disk I/O)."""
    filters = _component_filters(
        family, site, deployment, disk, date, time, df, channel, resample, start
    )
    with open_catalog_read() as conn:
        return match_recordings(conn, filters=filters, q=q, offset=offset, limit=limit)
