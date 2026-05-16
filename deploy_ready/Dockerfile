# =============================================================================
# CityFoam RAG Chatbot — Dockerfile
# =============================================================================
# Build:  docker build -t cityfoam-rag .
# Run:    docker compose up
# =============================================================================

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────
COPY config.py ingest.py server.py ./

# ── Frontend (served as Flask static files) ──────────────────────────────
RUN mkdir -p static
COPY static/index.html static/
COPY static/style.css  static/
COPY static/main.js    static/

# ── Knowledge-base data files ─────────────────────────────────────────────
# All files in ./data/ will be ingested on first startup.
RUN mkdir -p data
COPY data/knowledge_base.md data/
COPY data/zbranches.xlsx    data/
COPY data/refundpolicy.docx data/

# ── Persistent ChromaDB directory (overridden by volume mount) ────────────
RUN mkdir -p chroma_db

# ── Entrypoint ────────────────────────────────────────────────────────────
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

# gunicorn is started by entrypoint.sh after ingest check
CMD ["./entrypoint.sh"]