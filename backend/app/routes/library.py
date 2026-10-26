from fastapi import APIRouter, Query

from app.config import WAV_LIBRARY_ROOTS
from app.services import catalog_service
from app.services.gpl_cache import is_gpl_cached
from app.services.library_service import list_entries, resolve_file_id

router = APIRouter()


@router.get("/api/library/roots")
async def get_roots() -> dict:
    return {
        "roots": [
            {"id": i, "label": root.name}
            for i, root in enumerate(WAV_LIBRARY_ROOTS)
        ]
    }


@router.get("/api/library/analyses")
async def get_file_analyses(fileId: str = Query(...)) -> dict:
    """Dictionaries and song-structure CSVs in the catalog for one library WAV (the picker).

    ``gplCached`` says whether its GPL spectrogram already exists (view) or must be run.
    """
    wav_path = resolve_file_id(fileId)
    return {**catalog_service.list_analyses_for_wav(wav_path), "gplCached": is_gpl_cached(wav_path)}


@router.get("/api/library/browse")
async def browse(
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
