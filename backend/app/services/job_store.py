from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import WebSocket

@dataclass
class JobState:
    job_id: str
    status: Literal["queued", "processing", "complete", "error"]
    filename: str
    job_dir: Path
    library_wav_path: Path | None = None  # set for library jobs → write .gpl cache on completion
    manifest: dict | None = None
    error: str | None = None
    ws_clients: list["WebSocket"] = field(default_factory=list)

JOBS: dict[str, JobState] = {}

# Tracks the in-flight job for each library wav path so a second pick of the
# same file while it's still processing attaches to that job instead of
# starting a redundant concurrent pipeline run into the same cache dir.
LIBRARY_JOBS_IN_FLIGHT: dict[str, str] = {}

def get_job(job_id: str) -> JobState | None:
    return JOBS.get(job_id)

def create_job(job_id: str, filename: str, job_dir: Path) -> JobState:
    job = JobState(job_id=job_id, status="queued", filename=filename, job_dir=job_dir)
    JOBS[job_id] = job
    return job

def get_in_flight_library_job(wav_path: Path) -> JobState | None:
    job_id = LIBRARY_JOBS_IN_FLIGHT.get(str(wav_path))
    if job_id is None:
        return None
    job = JOBS.get(job_id)
    if job is None or job.status in ("complete", "error"):
        LIBRARY_JOBS_IN_FLIGHT.pop(str(wav_path), None)
        return None
    return job

def set_in_flight_library_job(wav_path: Path, job_id: str) -> None:
    LIBRARY_JOBS_IN_FLIGHT[str(wav_path)] = job_id

def clear_in_flight_library_job(wav_path: Path) -> None:
    LIBRARY_JOBS_IN_FLIGHT.pop(str(wav_path), None)
