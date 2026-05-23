# =============================================================================
# CityFoam RAG Chatbot — Dockerfile (v3, FastAPI + MLflow)
# =============================================================================
#
# Build:
#   docker build -t cityfoam-rag .
#
# Run (standalone):
#   docker run --env-file .env -p 8000:8000 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/chroma_db:/app/chroma_db \
#     cityfoam-rag
#
# Run (recommended — use docker-compose.yml):
#   docker compose up
#
# Why Docker instead of bare Python:
#   - Completely isolated from your host Anaconda / Keras / TF conflicts.
#   - NO TensorFlow installed at all — sentence-transformers runs on
#     PyTorch (CPU-only, ~700 MB) which is all we need.
#   - Reproducible on any machine, CI, and Azure Container Apps.
# =============================================================================

FROM python:3.11-slim

# ── System packages needed by unstructured, chromadb, and curl health-check
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libmagic1 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 0: Pin NumPy to 1.x before anything else resolves it
RUN pip install --no-cache-dir "numpy==1.26.4"

# ── Step 1: Install PyTorch CPU-only FIRST (before sentence-transformers)
#    This prevents pip from pulling in a CUDA build (4 GB) and also
#    ensures there is NO TensorFlow on this image at all.
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchvision==0.20.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Install all other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# ── Step 3: Pre-download the embedding model so the container is self-contained
#    (no network call needed at runtime — important for air-gapped Azure deployments)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
print('Embedding model cached.')"

# ── Application source files (copy every .py file at the project root)
COPY *.py ./

# ── Frontend static files
RUN mkdir -p static
COPY static/ static/

# ── Knowledge-base data files
#    Mounted as a volume in production; copied here so a standalone
#    docker run also works without a volume.
RUN mkdir -p data
COPY data/ data/

# ── Persistent directories (overridden by volume mounts in production)
RUN mkdir -p chroma_db mlruns monitoring_db

# ── Entrypoint script
COPY dashboard.html .
COPY entrypoint.sh .
RUN sed -i 's/\r//g' entrypoint.sh && chmod +x entrypoint.sh

# Suppress the oneDNN TF noise (harmless, just noisy)
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TRANSFORMERS_NO_TF=1

EXPOSE 8000

# Health-check so Docker / Azure knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["./entrypoint.sh"]