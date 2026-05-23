"""
CityFoam — Retraining Scheduler
=================================
Runs the ingestion pipeline automatically on a configurable schedule
using APScheduler inside the FastAPI process.

Configuration (in .env)
------------------------
REINGEST_SCHEDULE     cron expression  default: "0 2 * * 0"  (Sundays 2am UTC)
REINGEST_ON_STARTUP   true|false       default: false
MIN_CHUNKS_THRESHOLD  integer          default: 10

Usage in server.py lifespan
----------------------------
    from scheduler import start_scheduler, stop_scheduler
    start_scheduler(collection=_collection)
    yield
    stop_scheduler()
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("cityfoam.scheduler")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REINGEST_SCHEDULE    = os.environ.get("REINGEST_SCHEDULE",    "0 2 * * 0")
REINGEST_ON_STARTUP  = os.environ.get("REINGEST_ON_STARTUP",  "false").lower() == "true"
MIN_CHUNKS_THRESHOLD = int(os.environ.get("MIN_CHUNKS_THRESHOLD", "10"))

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_scheduler                     = None
_last_run_ts:     Optional[str] = None
_last_run_status: str           = "never"
_last_run_chunks: Optional[int] = None
_lock                           = threading.Lock()


# ---------------------------------------------------------------------------
# Core ingest job
# ---------------------------------------------------------------------------
def _ingest_job() -> None:
    global _last_run_ts, _last_run_status, _last_run_chunks

    with _lock:
        logger.info("Scheduled retraining job started.")
        t0 = time.perf_counter()
        try:
            from ingest import process_data_folder, build_vector_db

            docs    = process_data_folder()
            build_vector_db(docs)
            elapsed = round(time.perf_counter() - t0, 1)

            _last_run_ts     = datetime.now(timezone.utc).isoformat()
            _last_run_status = "success"
            _last_run_chunks = len(docs)
            logger.info("Retraining complete — %d chunks in %.1fs.", len(docs), elapsed)

            # Log to MLflow (best-effort)
            try:
                from mlflow_tracker import _mlflow
                ml = _mlflow()
                if ml:
                    with ml.start_run(run_name="scheduled-reingest"):
                        ml.log_params({"trigger": "scheduler",
                                       "schedule": REINGEST_SCHEDULE})
                        ml.log_metrics({"chunks": len(docs), "ingest_secs": elapsed})
            except Exception:
                pass

        except Exception as exc:
            _last_run_ts     = datetime.now(timezone.utc).isoformat()
            _last_run_status = f"failed: {exc}"
            _last_run_chunks = None
            logger.error("Scheduled retraining FAILED: %s", exc)


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------
def start_scheduler(collection=None) -> None:
    global _scheduler

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning("apscheduler not installed — scheduled retraining disabled.")
        return

    # Parse 5-field cron
    try:
        m, h, d, mo, dow = (REINGEST_SCHEDULE.split() + ["*"] * 5)[:5]
        trigger = CronTrigger(minute=m, hour=h, day=d, month=mo, day_of_week=dow)
    except Exception as exc:
        logger.error("Bad REINGEST_SCHEDULE '%s': %s — defaulting to Sun 2am.",
                     REINGEST_SCHEDULE, exc)
        trigger = CronTrigger(hour=2, minute=0, day_of_week=0)

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _ingest_job,
        trigger            = trigger,
        id                 = "reingest",
        name               = "CityFoam scheduled retraining",
        misfire_grace_time = 3600,
        coalesce           = True,
    )
    _scheduler.start()

    job = _scheduler.get_job("reingest")
    next_run = job.next_run_time.isoformat() if job and job.next_run_time else "unknown"
    logger.info("Scheduler started — cron: '%s' UTC  |  next run: %s",
                REINGEST_SCHEDULE, next_run)

    # Startup threshold check
    if REINGEST_ON_STARTUP and collection is not None:
        try:
            if collection.count() < MIN_CHUNKS_THRESHOLD:
                logger.warning("ChromaDB below threshold — triggering immediate retraining.")
                threading.Thread(target=_ingest_job, daemon=True).start()
        except Exception:
            pass


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")
    _scheduler = None


# ---------------------------------------------------------------------------
# Status + manual trigger  (used by admin.py)
# ---------------------------------------------------------------------------
def get_scheduler_status() -> dict:
    if _scheduler is None:
        return {
            "running": False, "schedule": REINGEST_SCHEDULE,
            "next_run": None, "last_run_ts": _last_run_ts,
            "last_run_status": _last_run_status,
            "last_run_chunks": _last_run_chunks,
        }
    job = _scheduler.get_job("reingest")
    return {
        "running":         _scheduler.running,
        "schedule":        REINGEST_SCHEDULE,
        "next_run":        job.next_run_time.isoformat() if job and job.next_run_time else None,
        "last_run_ts":     _last_run_ts,
        "last_run_status": _last_run_status,
        "last_run_chunks": _last_run_chunks,
        "job_count":       len(_scheduler.get_jobs()),
    }


def trigger_now() -> None:
    """Fire the ingest job immediately in a background thread."""
    threading.Thread(target=_ingest_job, daemon=True, name="reingest-manual").start()
    logger.info("Manual retraining triggered.")