"""
CityFoam — Monitoring & Observability
=======================================
Every chat request is logged to a local SQLite database with:
  • timestamp
  • chat_id
  • user query (truncated for privacy if needed)
  • AI response length
  • total latency (ms)
  • RAG retrieval quality score (0–1)
  • detected language
  • optimized query

The /admin/logs endpoint exposes recent entries as JSON.
A FastAPI middleware measures wall-clock latency for every request.
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("cityfoam.monitoring")

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
DB_PATH = Path("./monitoring.db")


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL,
            chat_id         TEXT,
            language        TEXT,
            raw_query       TEXT,
            optimized_query TEXT,
            response_chars  INTEGER,
            latency_ms      REAL,
            rag_score       REAL,
            rag_warning     INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_query_log_ts ON query_log (ts DESC)
    """)
    conn.commit()


@contextmanager
def _db():
    """Thread-safe SQLite connection context manager."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        _create_tables(conn)
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Log entry dataclass
# ---------------------------------------------------------------------------
@dataclass
class QueryLogEntry:
    chat_id:         str
    raw_query:       str
    optimized_query: str   = ""
    language:        str   = "UNKNOWN"
    response_chars:  int   = 0
    latency_ms:      float = 0.0
    rag_score:       float = 1.0
    rag_warning:     bool  = False


# ---------------------------------------------------------------------------
# Public logging function
# ---------------------------------------------------------------------------
def log_query(entry: QueryLogEntry) -> None:
    """Persist a query log entry to SQLite. Called after each chat turn."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _db() as conn:
            conn.execute(
                """
                INSERT INTO query_log
                    (ts, chat_id, language, raw_query, optimized_query,
                     response_chars, latency_ms, rag_score, rag_warning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    entry.chat_id,
                    entry.language,
                    entry.raw_query[:500],        # cap stored query length
                    entry.optimized_query[:500],
                    entry.response_chars,
                    round(entry.latency_ms, 2),
                    round(entry.rag_score, 4),
                    int(entry.rag_warning),
                ),
            )
            conn.commit()
        if entry.rag_warning:
            logger.warning(
                "LOW RAG QUALITY (score=%.2f) for query: %r",
                entry.rag_score, entry.raw_query[:80],
            )
    except Exception as exc:
        logger.error("Failed to write query log: %s", exc)


# ---------------------------------------------------------------------------
# Retrieve recent logs (for /admin/logs)
# ---------------------------------------------------------------------------
def get_recent_logs(limit: int = 100) -> list[dict]:
    """Return the most recent `limit` log entries as a list of dicts."""
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM query_log ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
    except Exception as exc:
        logger.error("Failed to read query log: %s", exc)
        return []


def get_stats() -> dict:
    """Aggregate statistics for the /admin/stats endpoint."""
    try:
        with _db() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)            AS total_queries,
                    AVG(latency_ms)     AS avg_latency_ms,
                    MAX(latency_ms)     AS max_latency_ms,
                    AVG(rag_score)      AS avg_rag_score,
                    SUM(rag_warning)    AS low_quality_count
                FROM query_log
            """).fetchone()
            lang_rows = conn.execute("""
                SELECT language, COUNT(*) as cnt
                FROM query_log GROUP BY language
            """).fetchall()
            return {
                "total_queries":    row["total_queries"],
                "avg_latency_ms":   round(row["avg_latency_ms"] or 0, 1),
                "max_latency_ms":   round(row["max_latency_ms"] or 0, 1),
                "avg_rag_score":    round(row["avg_rag_score"] or 0, 3),
                "low_quality_pct":  round(
                    100 * (row["low_quality_count"] or 0) / max(row["total_queries"], 1), 1
                ),
                "language_breakdown": {r["language"]: r["cnt"] for r in lang_rows},
            }
    except Exception as exc:
        logger.error("Failed to compute stats: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# FastAPI Latency Middleware
# ---------------------------------------------------------------------------
class LatencyMiddleware(BaseHTTPMiddleware):
    """
    Attaches wall-clock request timing to every response.
    Adds  X-Process-Time-Ms  header so monitoring tools can pick it up.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        if request.url.path.startswith("/api/"):
            logger.info(
                "%s %s → %d  [%.1f ms]",
                request.method, request.url.path,
                response.status_code, elapsed_ms,
            )
        return response
