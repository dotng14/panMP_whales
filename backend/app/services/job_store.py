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

def get_job(job_id: str) -> JobState | None:
    return JOBS.get(job_id)

def create_job(job_id: str, filename: str, job_dir: Path) -> JobState:
    job = JobState(job_id=job_id, status="queued", filename=filename, job_dir=job_dir)
    JOBS[job_id] = job
    return job
