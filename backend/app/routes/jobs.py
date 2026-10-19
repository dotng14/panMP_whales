from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import MAX_UPLOAD_MB, TMP_DIR
from app.services.gpl_cache import cache_dir_for, is_gpl_cached
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
    wav_path.write_bytes(content)

    job = create_job(job_id, file.filename, job_dir)
    await start_job(job)

    return {"jobId": job_id, "eventsUrl": f"/api/jobs/{job_id}/events"}


@router.post("/api/jobs/from-library")
async def create_job_from_library(body: LibraryJobRequest) -> dict:
    wav_path = resolve_file_id(body.fileId)  # raises 400/403/404 on bad input

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
    if is_gpl_cached(wav_path):
        manifest = json.loads(manifest_path.read_text())
        manifest["jobId"] = job_id
        # Symlink the cache's output dir into the job dir (no copy) so the panel
        # endpoint serves panel.bin straight from the cache.
        (job_dir / "output").symlink_to(cache_dir / "output", target_is_directory=True)
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
async def get_song_structure(job_id: str) -> dict:
    """Motif / song / theme / phrase units for the job's WAV stem, if a CSV exists."""
    from app.services.song_structure import load_song_structure  # noqa: PLC0415

    job = _require_job(job_id)
    stem = Path(job.filename).stem
    try:
        payload = load_song_structure(stem)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="No song-structure CSV for this WAV")
    return payload


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
