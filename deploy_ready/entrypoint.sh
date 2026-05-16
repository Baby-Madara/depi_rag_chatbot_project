#!/usr/bin/env bash
# =============================================================================
# CityFoam RAG Chatbot — Container Entrypoint
# =============================================================================
# 1. Checks whether ChromaDB has already been populated.
# 2. If not (first run or after volume wipe), runs the ingestion pipeline.
# 3. Starts the production gunicorn server.
# =============================================================================

set -e   # exit immediately on error

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   CityFoam RAG Chatbot — Startup     ║"
echo "╚══════════════════════════════════════╝"
echo ""

CHROMA_DIR="${CHROMA_DB_DIR:-./chroma_db}"

# A populated ChromaDB always contains at least one .sqlite3 file.
if find "$CHROMA_DIR" -name "*.sqlite3" 2>/dev/null | grep -q .; then
    echo "ChromaDB already populated — skipping ingestion."
else
    echo "ChromaDB is empty. Running ingestion pipeline…"
    python ingest.py
    echo "Ingestion complete."
fi

echo ""
echo "Starting production server (gunicorn) on port ${PORT:-8000}…"
echo ""

exec gunicorn \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --worker-class gthread \
    --access-logfile - \
    --error-logfile - \
    server:app