from __future__ import annotations

import asyncio
from pathlib import Path

from app.services.gpl_cache import finalize_gpl_cache, register_gpl_in_catalog
from app.services.job_store import JobState, clear_in_flight_library_job
from app.services.manifest_builder import build_manifest_from_outputs
from app.ws.hub import broadcast_json


def _run_job_sync(job: JobState, loop: asyncio.AbstractEventLoop) -> None:
    """Blocking GPL pipeline run. Called from a thread-pool executor."""
    output_dir = job.job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    job.status = "processing"
    asyncio.run_coroutine_threadsafe(
        broadcast_json(job.ws_clients, {"type": "status", "status": "processing"}), loop
    ).result()

    try:
        from analysis.GPL_Python.process_v4 import ProcessingOrchestrator  # noqa: PLC0415

        ProcessingOrchestrator(output_dir).process(job.job_dir / "input.wav")
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        if job.library_wav_path is not None:
            clear_in_flight_library_job(job.library_wav_path)
        asyncio.run_coroutine_threadsafe(
            broadcast_json(job.ws_clients, {"type": "error", "message": str(exc)}), loop
        ).result()
        return

    manifest = build_manifest_from_outputs(output_dir, job.job_id)
    job.manifest = manifest
    asyncio.run_coroutine_threadsafe(
        broadcast_json(job.ws_clients, {"type": "manifest", "data": manifest}), loop
    ).result()

    # panel.bin is written by process_v4; signal ready if present.
    if (output_dir / "panel.bin").is_file():
        asyncio.run_coroutine_threadsafe(
            broadcast_json(job.ws_clients, {"type": "panel_ready"}), loop
        ).result()

    if job.library_wav_path is not None:
        try:
            finalize_gpl_cache(job.library_wav_path, output_dir, title=Path(job.filename).stem)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                broadcast_json(
                    job.ws_clients,
                    {"type": "warning", "message": f"Cache write failed: {exc}"},
                ),
                loop,
            ).result()
        try:
            register_gpl_in_catalog(job.library_wav_path)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                broadcast_json(
                    job.ws_clients,
                    {"type": "warning", "message": f"Catalog registration failed: {exc}"},
                ),
                loop,
            ).result()

    job.status = "complete"
    if job.library_wav_path is not None:
        clear_in_flight_library_job(job.library_wav_path)
    asyncio.run_coroutine_threadsafe(
        broadcast_json(job.ws_clients, {"type": "complete", "jobId": job.job_id}), loop
    ).result()


async def start_job(job: JobState) -> None:
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_job_sync, job, loop)


async def start_job_fast(job: JobState, manifest: dict) -> None:
    """Cached fast path: no pipeline run. Panel comes straight from the cache."""
    job.manifest = manifest
    job.status = "complete"
    await broadcast_json(job.ws_clients, {"type": "status", "status": "complete"})
    await broadcast_json(job.ws_clients, {"type": "manifest", "data": manifest})
    await broadcast_json(job.ws_clients, {"type": "panel_ready"})
    await broadcast_json(job.ws_clients, {"type": "complete", "jobId": job.job_id})
