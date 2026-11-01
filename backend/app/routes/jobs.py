from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from contextlib import closing
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import MAX_UPLOAD_MB, TMP_DIR
from app.services.command_service import (
    CommandContext,
    action_catalog,
    get_run,
    start_run,
    terminate_run,
)
from app.services.gpl_cache import cache_dir_for, is_gpl_cached, register_gpl_in_catalog
from app.services.job_store import (
    JobState,
    create_job,
    get_in_flight_library_job,
    get_job,
    set_in_flight_library_job,
)
from app.services.library_service import resolve_file_id
from app.services.pipeline_service import start_job, start_job_fast

router = APIRouter()
logger = logging.getLogger(__name__)


def _require_job(job_id: str) -> JobState:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


class LibraryJobRequest(BaseModel):
    fileId: str


@router.post("/api/jobs")
async def create_job_endpoint(file: UploadFile) -> dict:
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files accepted")

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit")

    job_id = str(uuid.uuid4())
    job_dir = TMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    wav_path = job_dir / "input.wav"
    await asyncio.to_thread(wav_path.write_bytes, content)

    job = create_job(job_id, file.filename, job_dir)
    await start_job(job)

    return {"jobId": job_id, "eventsUrl": f"/api/jobs/{job_id}/events"}


@router.post("/api/jobs/from-library")
async def create_job_from_library(body: LibraryJobRequest) -> dict:
    wav_path = resolve_file_id(body.fileId)  # raises 400/403/404 on bad input
    # Off the event loop (it can rebuild panel.bin), and before the in-flight
    # check: nothing below may await until set_in_flight_library_job, or two
    # requests for the same file could both start a pipeline run.
    cached = await asyncio.to_thread(is_gpl_cached, wav_path)

    # Attach to an already-running job for this file instead of starting a
    # second concurrent pipeline run that would race the first over the same
    # cache output dir.
    existing = get_in_flight_library_job(wav_path)
    if existing is not None:
        return {"jobId": existing.job_id, "eventsUrl": f"/api/jobs/{existing.job_id}/events"}

    job_id = str(uuid.uuid4())
    job_dir = TMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    link = job_dir / "input.wav"
    link.symlink_to(wav_path)

    job = create_job(job_id, wav_path.name, job_dir)

    # Fast path: if a pre-processed .gpl cache exists, skip the pipeline entirely.
    cache_dir = cache_dir_for(wav_path)
    manifest_path = cache_dir / "manifest.json"
    if cached:
        manifest = json.loads(manifest_path.read_text())
        manifest["jobId"] = job_id
        # Symlink the cache's output dir into the job dir (no copy) so the panel
        # endpoint serves panel.bin straight from the cache.
        (job_dir / "output").symlink_to(cache_dir / "output", target_is_directory=True)
        # Caches made before jobs registered themselves reach the catalog here.
        try:
            await asyncio.to_thread(register_gpl_in_catalog, wav_path)
        except Exception:
            logger.exception("Catalog registration failed for %s", wav_path)
        await start_job_fast(job, manifest)
    else:
        # Slow path: process into the .gpl cache dir so results persist for next time.
        cache_output = cache_dir / "output"
        cache_output.mkdir(parents=True, exist_ok=True)
        (job_dir / "output").symlink_to(cache_output, target_is_directory=True)
        job.library_wav_path = wav_path
        set_in_flight_library_job(wav_path, job_id)
        await start_job(job)

    return {"jobId": job_id, "eventsUrl": f"/api/jobs/{job_id}/events"}


@router.get("/api/jobs/{job_id}/panel")
async def get_job_panel(job_id: str) -> FileResponse:
    job = _require_job(job_id)
    panel_path = job.job_dir / "output" / "panel.bin"
    if not panel_path.exists():
        raise HTTPException(status_code=404, detail="Panel not ready")
    return FileResponse(panel_path, media_type="application/octet-stream")


@router.get("/api/jobs/{job_id}")
async def get_job_endpoint(job_id: str) -> dict:
    job = _require_job(job_id)
    return {
        "jobId": job.job_id,
        "status": job.status,
        "filename": job.filename,
        "manifest": job.manifest,
        "error": job.error,
    }


@router.get("/api/jobs/{job_id}/audio")
async def get_job_audio(job_id: str) -> FileResponse:
    job = _require_job(job_id)
    wav_path = job.job_dir / "input.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    # Content-Encoding: identity makes GZipMiddleware skip compression.
    # GZip on a 206 Range response gzips only the requested byte slice,
    # not the whole file, so the bytes no longer match what Content-Range
    # claims - this silently breaks audio seeking/streaming for any file
    # large enough that the <audio> element issues Range requests.
    return FileResponse(wav_path, media_type="audio/wav", headers={"Content-Encoding": "identity"})


@router.get("/api/jobs/{job_id}/song-structure")
def get_song_structure(job_id: str, csvId: int | None = Query(None)) -> dict:
    """Motif / song / theme / phrase units: from catalog CSV ``csvId``, else the CSV named after the WAV."""
    from app.services import catalog_service  # noqa: PLC0415
    from app.services.song_structure import load_song_structure, load_song_structure_csv  # noqa: PLC0415

    job = _require_job(job_id)
    stem = Path(job.filename).stem
    try:
        if csvId is None:
            payload = load_song_structure(stem)
        else:
            row = catalog_service.get_csv(csvId)
            path = Path(row["csv_path"])
            if not path.is_file():
                raise HTTPException(status_code=404, detail=f"CSV file missing: {path}")
            payload = load_song_structure_csv(path, row["wav_stem"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="No song-structure CSV for this WAV")
    return payload


class RepeatReviewRequest(BaseModel):
    chunkId: str
    targetTemplateId: int


class TemplateLabelRequest(BaseModel):
    label: str | None = None


_CHUNK_ID_RE = re.compile(r"^template-(\d+):")


def _review_dir(job: JobState, dictionary_id: int) -> Path:
    """Where this job keeps relabel reviews for one dictionary (chunk ids are per-dictionary)."""
    return job.job_dir / "output" / "repeat_reviews" / f"dictionary_{dictionary_id}"


@router.get("/api/jobs/{job_id}/repeats")
def get_repeats(job_id: str, dictionaryId: int = Query(...)) -> dict:
    """RepeatMatches for a catalog dictionary, with this job's staged relabels stamped on."""
    from app.services import catalog_service, repeat_reviews  # noqa: PLC0415

    job = _require_job(job_id)
    raw = catalog_service.build_repeats(dictionaryId)
    reviews = repeat_reviews.load_reviews(_review_dir(job, dictionaryId))
    return repeat_reviews.stamp_reviews_onto_templates(raw.get("templates", []), reviews)


@router.put("/api/jobs/{job_id}/repeat-reviews")
def put_repeat_review(
    job_id: str, body: RepeatReviewRequest, dictionaryId: int = Query(...)
) -> dict:
    """Stage a chunk's reassignment to another template number (no rebuild)."""
    from app.services import catalog_service, repeat_reviews  # noqa: PLC0415

    job = _require_job(job_id)
    if body.targetTemplateId < 0:
        raise HTTPException(status_code=400, detail="targetTemplateId must be non-negative")
    match = _CHUNK_ID_RE.match(body.chunkId)
    if match is None:
        raise HTTPException(status_code=400, detail=f"Malformed chunkId: {body.chunkId}")
    original = int(match.group(1)) + catalog_service.template_number_offset(dictionaryId)
    review_dir = _review_dir(job, dictionaryId)
    review_dir.mkdir(parents=True, exist_ok=True)
    repeat_reviews.save_review(review_dir, body.chunkId, body.targetTemplateId, original)
    return {"chunkId": body.chunkId, "targetTemplateId": body.targetTemplateId}


@router.patch("/api/jobs/{job_id}/templates/{template_number}")
def patch_template_label(
    job_id: str,
    template_number: int,
    body: TemplateLabelRequest,
    dictionaryId: int = Query(...),
) -> dict:
    """Rename a template (catalog alias); ``template_number`` is the UI number."""
    from app.services import catalog_service  # noqa: PLC0415
    from database import catalog  # noqa: PLC0415

    _require_job(job_id)
    template_id = template_number - catalog_service.template_number_offset(dictionaryId)
    result = catalog_service.set_dictionary_template_label(dictionaryId, template_id, body.label)
    return {"label": result["label"], "named": catalog.is_named(result["label"])}


@router.post("/api/jobs/{job_id}/review-submit")
def post_review_submit(job_id: str) -> dict:
    """Submit-rebuild of a reviewed dictionary is not ported yet."""
    _require_job(job_id)
    raise HTTPException(
        status_code=501,
        detail="Submit-rebuild is not implemented yet; relabels are saved but not applied.",
    )


class TerminalActionRequest(BaseModel):
    action: str
    params: dict = {}
    # Template the reviewer is viewing (kept for parity with gplrev3's form).
    currentTemplate: int | None = None
    # Catalog dictionary the reviewer is looking at; resolves the GPL segmentation.
    dictionaryId: int | None = None


def _source_wav(job: JobState) -> Path:
    link = job.job_dir / "input.wav"
    try:
        return link.resolve()
    except OSError:
        return link


def _terminal_context(job: JobState, body: TerminalActionRequest) -> CommandContext:
    """Resolve the job's stem / GPL segmentation (from the dictionary, else the WAV) for an action."""
    from app.config import CATALOG_DB  # noqa: PLC0415
    from database import catalog  # noqa: PLC0415

    source_wav = _source_wav(job)
    stem = Path(job.filename).stem
    segmentation_id: int | None = None
    try:
        conn = catalog.connect(CATALOG_DB, read_only=True)
    except FileNotFoundError:
        conn = None
    if conn is not None:
        with closing(conn):
            if body.dictionaryId is not None:
                dictionary = catalog.get_dictionary(conn, body.dictionaryId)
                if dictionary is None:
                    raise HTTPException(status_code=404, detail=f"Dictionary {body.dictionaryId} not found")
                if dictionary.get("segmentation_id") is not None:
                    segmentation_id = int(dictionary["segmentation_id"])
                if dictionary.get("wav_stem"):
                    stem = str(dictionary["wav_stem"])
            if segmentation_id is None:
                wav = catalog.find_wav_file(conn, source_wav) or catalog.find_wav_file(conn, job.filename)
                if wav is not None:
                    stem = wav.get("stem") or stem
                    segs = catalog.list_gpl_segmentations_for_wav(conn, int(wav["wav_id"]))
                    # Prefer a segmentation whose windows are indexed (usable for a build).
                    chosen = next((s for s in segs if s.get("n_windows")), segs[0] if segs else None)
                    if chosen is not None:
                        segmentation_id = int(chosen["segmentation_id"])
    return CommandContext(
        stem=stem,
        segmentation_id=segmentation_id,
        dictionary_id=body.dictionaryId,
        current_template=body.currentTemplate,
        source_wav=str(source_wav),
    )


@router.get("/api/jobs/{job_id}/terminal")
async def get_terminal_info(job_id: str) -> dict:
    """Action catalog (history + randomized_builder) for the repeat Actions form."""
    _require_job(job_id)
    return {"commands": action_catalog()}


@router.post("/api/jobs/{job_id}/terminal")
def post_terminal_command(job_id: str, body: TerminalActionRequest) -> dict:
    # Start the action in a child process and return its id immediately: a build
    # can run for minutes. The client polls the run endpoint below.
    job = _require_job(job_id)
    ctx = _terminal_context(job, body)
    run_id = start_run(ctx, body.action, body.params)
    return {"runId": run_id, "status": "running"}


@router.get("/api/jobs/{job_id}/terminal/runs/{run_id}")
def get_terminal_run(job_id: str, run_id: str) -> dict:
    _require_job(job_id)
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Command run not found or already collected")
    return run


@router.delete("/api/jobs/{job_id}/terminal/runs/{run_id}")
def delete_terminal_run(job_id: str, run_id: str) -> dict:
    _require_job(job_id)
    return {"ok": True, "terminated": terminate_run(run_id)}


@router.post("/api/jobs/{job_id}/terminal/runs/{run_id}/terminate")
def post_terminate_terminal_run(job_id: str, run_id: str) -> dict:
    _require_job(job_id)
    return {"ok": True, "terminated": terminate_run(run_id)}


@router.websocket("/api/jobs/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    job.ws_clients.append(websocket)

    # Send current status immediately so late-connecting clients catch up
    await websocket.send_json({"type": "status", "status": job.status})
    if job.status == "complete" and job.manifest:
        await websocket.send_json({"type": "manifest", "data": job.manifest})
        if (job.job_dir / "output" / "panel.bin").exists():
            await websocket.send_json({"type": "panel_ready"})
        await websocket.send_json({"type": "complete", "jobId": job_id})
    elif job.status == "error":
        await websocket.send_json({"type": "error", "message": job.error or "unknown error"})

    try:
        while True:
            await websocket.receive_text()  # keep connection alive; ignore client messages
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in job.ws_clients:
            job.ws_clients.remove(websocket)
