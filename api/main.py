import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crew.pipeline import run_pipeline

log = logging.getLogger("research_api")
app = FastAPI(title="Multi-Agent Research Assistant")

# One worker = one pipeline run at a time (protects Groq rate limits)
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
MAX_PENDING = 3          # running + queued jobs allowed at once
JOB_TTL_SECONDS = 3600   # forget finished jobs after 1 hour


class ResearchRequest(BaseModel):
    question: str


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str                  # queued | running | done | failed
    question: str
    report: str | None = None
    claims: list[dict] = []
    seconds: float = 0.0
    error: str | None = None


def _cleanup():
    now = time.time()
    with _jobs_lock:
        stale = [
            jid for jid, j in _jobs.items()
            if j["status"] in ("done", "failed") and now - j["finished_at"] > JOB_TTL_SECONDS
        ]
        for jid in stale:
            del _jobs[jid]


def _run_job(job_id: str, question: str):
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        out = run_pipeline(question)
        with _jobs_lock:
            _jobs[job_id].update(
                status="done",
                report=out["report"],
                claims=out["claims"],
                seconds=round(out["seconds"], 1),
                finished_at=time.time(),
            )
    except Exception:
        log.exception("Research pipeline failed for job %s", job_id)
        with _jobs_lock:
            _jobs[job_id].update(
                status="failed",
                error="Research pipeline failed. Check the server logs.",
                finished_at=time.time(),
            )


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Multi-Agent Research Assistant"}


@app.post("/research", response_model=JobCreated, status_code=202)
def submit_research(request: ResearchRequest):
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    _cleanup()
    with _jobs_lock:
        pending = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))
        if pending >= MAX_PENDING:
            raise HTTPException(status_code=429, detail="Server is busy. Try again shortly.")
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"status": "queued", "question": question, "claims": []}

    _executor.submit(_run_job, job_id, question)
    return JobCreated(job_id=job_id, status="queued")


@app.get("/research/{job_id}", response_model=JobStatus)
def get_research(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found (it may have expired).")
        return JobStatus(job_id=job_id, **{k: v for k, v in job.items() if k != "finished_at"})