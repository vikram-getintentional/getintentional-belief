from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Dict, Literal, Optional

from pydantic import BaseModel

JobType = Literal["account_plan", "marketing_planner", "enrich_accounts"]
JobState = Literal["queued", "running", "completed", "failed"]


class JobStatus(BaseModel):
    job_id: str
    type: JobType
    status: JobState
    phase: str
    message: str
    progress: float
    updated_at: datetime
    result_location: Optional[str] = None
    error: Optional[str] = None

    class Config:
        validate_assignment = True


_job_store: Dict[str, JobStatus] = {}
_job_store_lock = Lock()


def _utcnow() -> datetime:
    return datetime.utcnow()


def create_job(
    job_id: str,
    job_type: JobType,
    *,
    phase: str = "queued",
    message: str = "Job queued.",
    progress: float = 0.0,
) -> JobStatus:
    job = JobStatus(
        job_id=job_id,
        type=job_type,
        status="queued",
        phase=phase,
        message=message,
        progress=progress,
        updated_at=_utcnow(),
    )
    with _job_store_lock:
        _job_store[job_id] = job
    return job


def get_job(job_id: str) -> Optional[JobStatus]:
    with _job_store_lock:
        job = _job_store.get(job_id)
        return job.copy() if job else None


def update_job(
    job_id: str,
    *,
    status: Optional[JobState] = None,
    phase: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    result_location: Optional[str] = None,
    error: Optional[str] = None,
) -> JobStatus:
    with _job_store_lock:
        job = _job_store.get(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")
        if status:
            job.status = status
        if phase is not None:
            job.phase = phase
        if message is not None:
            job.message = message
        if progress is not None:
            job.progress = progress
        if result_location is not None:
            job.result_location = result_location
        if error is not None:
            job.error = error
        job.updated_at = _utcnow()
        return job.copy()
