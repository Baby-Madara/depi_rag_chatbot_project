#!/bin/bash
# =============================================================================
# CityFoam — Container Entrypoint
# =============================================================================
# Logic:
#   1. If chroma_db/ is empty (first boot or volume was cleared), run ingest.
#   2. Otherwise skip ingest — it's slow (embedding takes ~60s) and the DB
#      is already persisted on the volume.
#   3. Start the FastAPI server with uvicorn.
#
# To force a re-ingest without restarting the container use the admin API:
#   POST /admin/reingest  (X-Admin-Key header required)
# =============================================================================

set -e

CHROMA_DIR="${CHROMA_DB_DIR:-/app/chroma_db}"
DATA_DIR="${DATA_DIR:-/app/data}"

echo "========================================"
echo "  CityFoam RAG Chatbot — Starting up"
echo "========================================"
echo "  LLM Provider : ${LLM_PROVIDER:-azure}"
echo "  Data dir     : ${DATA_DIR}"
echo "  Chroma dir   : ${CHROMA_DIR}"
echo "========================================"

# ── Ingest check ──────────────────────────────────────────────────────────
# Count files in chroma_db to decide whether ingest is needed.
CHROMA_FILE_COUNT=$(find "${CHROMA_DIR}" -type f 2>/dev/null | wc -l)

if [ "${CHROMA_FILE_COUNT}" -eq 0 ]; then
    echo ""
    echo ">> ChromaDB is empty — running ingestion pipeline..."
    echo ""
    python ingest.py
    echo ""
    echo ">> Ingestion complete."
    echo ""
else
    echo ">> ChromaDB already populated (${CHROMA_FILE_COUNT} files). Skipping ingest."
    echo "   To re-ingest: POST /admin/reingest with your admin key."
fi

# ── Start server ──────────────────────────────────────────────────────────
echo ""
echo ">> Starting FastAPI server on port ${PORT:-8000}..."
echo ""

exec uvicorn server:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --log-level info