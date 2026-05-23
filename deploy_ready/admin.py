"""
CityFoam — Admin Router
=========================
Mounted at /admin.  All routes require the X-Admin-Key header.

Endpoints
---------
POST  /admin/reingest          Trigger ingestion pipeline in the background.
GET   /admin/reingest/status   Check progress of the last ingestion run.
GET   /admin/logs              Recent query log entries (JSON).
GET   /admin/stats             Aggregate metrics (latency, RAG quality, etc.).
POST  /admin/secrets/refresh   Invalidate the secrets cache (after key rotation).
"""

import asyncio
import logging
import os
import time
from enum import Enum
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse

from monitoring import get_recent_logs, get_stats
from app_secrets import get_secret, invalidate_cache

logger = logging.getLogger("cityfoam.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Admin key dependency
# ---------------------------------------------------------------------------
def _require_admin_key(x_admin_key: str = Header(...)):
    expected = get_secret("CITYFOAM_ADMIN_KEY")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="CITYFOAM_ADMIN_KEY not configured on this server.",
        )
    if x_admin_key != expected:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    return x_admin_key


# ---------------------------------------------------------------------------
# Re-ingestion state
# ---------------------------------------------------------------------------
class IngestStatus(str, Enum):
    IDLE      = "idle"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"


_ingest_state: dict = {
    "status":      IngestStatus.IDLE,
    "started_at":  None,
    "finished_at": None,
    "chunks":      None,
    "error":       None,
}


async def _run_ingest() -> None:
    """
    Background task: runs the ingestion pipeline without blocking the server.
    Imports are deferred so the module loads instantly at server startup.
    """
    global _ingest_state
    _ingest_state.update(
        status=IngestStatus.RUNNING,
        started_at=time.time(),
        finished_at=None,
        chunks=None,
        error=None,
    )
    logger.info("Re-ingestion started in background.")

    try:
        # Run the CPU-heavy ingest in a thread so the event loop stays free
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, _sync_ingest)
        _ingest_state.update(
            status=IngestStatus.SUCCESS,
            finished_at=time.time(),
            chunks=chunks,
        )
        logger.info("Re-ingestion complete — %d chunks stored.", chunks)

    except Exception as exc:
        _ingest_state.update(
            status=IngestStatus.FAILED,
            finished_at=time.time(),
            error=str(exc),
        )
        logger.error("Re-ingestion failed: %s", exc)


def _sync_ingest() -> int:
    """Synchronous wrapper called in a thread executor."""
    from ingest import process_data_folder, build_vector_db  # noqa: PLC0415
    docs = process_data_folder()
    build_vector_db(docs)
    return len(docs)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/reingest")
async def trigger_reingest(
    background_tasks,                         # injected by caller
    _key: str = Depends(_require_admin_key),
):
    """
    Kick off the ingestion pipeline in the background.
    Returns immediately with status 202 Accepted.
    """
    if _ingest_state["status"] == IngestStatus.RUNNING:
        return JSONResponse(
            status_code=409,
            content={"detail": "Ingestion already running.", "state": _ingest_state},
        )
    background_tasks.add_task(_run_ingest)
    return {"detail": "Ingestion started.", "state": _ingest_state}


@router.get("/reingest/status")
async def reingest_status(_key: str = Depends(_require_admin_key)):
    """Return the current / last ingestion run state."""
    state = dict(_ingest_state)
    if state["started_at"]:
        state["started_at"]  = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(state["started_at"])
        )
    if state["finished_at"]:
        state["finished_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(state["finished_at"])
        )
    return state


@router.get("/logs")
async def query_logs(
    limit: int = 50,
    _key: str  = Depends(_require_admin_key),
):
    """Return the most recent query log entries."""
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000.")
    return {"status": "success", "count": limit, "logs": get_recent_logs(limit)}


@router.get("/stats")
async def query_stats(_key: str = Depends(_require_admin_key)):
    """Return aggregate metrics: latency, RAG quality, language breakdown."""
    return {"status": "success", "stats": get_stats()}


@router.post("/secrets/refresh")
async def refresh_secrets(_key: str = Depends(_require_admin_key)):
    """Invalidate the in-process secret cache — useful after key rotation."""
    invalidate_cache()
    return {"detail": "Secret cache cleared. Next call will re-fetch from Key Vault / env."}

@router.get("/scheduler")
async def scheduler_status(_key: str = Depends(_require_admin_key)):
    """Return retraining scheduler state: next run, last run, status."""
    from scheduler import get_scheduler_status
    return get_scheduler_status()


@router.post("/scheduler/trigger")
async def scheduler_trigger(_key: str = Depends(_require_admin_key)):
    """Manually trigger the retraining job immediately (background thread)."""
    from scheduler import trigger_now
    trigger_now()
    return {"detail": "Retraining job triggered. Check /admin/scheduler for status."}