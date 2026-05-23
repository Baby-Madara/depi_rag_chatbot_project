#!/bin/bash
# =============================================================================
# CityFoam RAG Chatbot — Master Deployment Script
# =============================================================================
# Usage:
#   ./deploy.sh              — full deploy (build + start + health check)
#   ./deploy.sh --rebuild    — force Docker image rebuild from scratch
#   ./deploy.sh --experiment — also run RAG hyperparameter sweep after deploy
#   ./deploy.sh --down       — stop and remove all containers
#   ./deploy.sh --logs       — tail live logs from all containers
#   ./deploy.sh --status     — print health + stats without starting anything
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
section() { echo -e "\n${BOLD}━━━  $*  ━━━${RESET}"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
REBUILD=false
RUN_EXPERIMENT=false
COMPOSE_CMD="docker compose"

APP_URL="http://localhost:8000"
MLFLOW_URL="http://localhost:5000"
HEALTH_RETRIES=30        # × 5s = 150s max wait
ADMIN_KEY="${CITYFOAM_ADMIN_KEY:-}"   # read from env if set, else from .env below

# ── Argument parsing ──────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --rebuild)    REBUILD=true ;;
    --experiment) RUN_EXPERIMENT=true ;;
    --down)
      section "Stopping all containers"
      $COMPOSE_CMD down
      success "All containers stopped."
      exit 0
      ;;
    --logs)
      exec $COMPOSE_CMD logs -f
      ;;
    --status)
      section "CityFoam — Status"
      echo ""
      info "App health:"
      curl -s "${APP_URL}/api/health" | python3 -m json.tool 2>/dev/null || \
        error "App not responding at ${APP_URL}"
      echo ""
      # Try to read admin key for stats
      if [ -f .env ]; then
        ADMIN_KEY_FROM_ENV=$(grep '^CITYFOAM_ADMIN_KEY=' .env | cut -d= -f2 | tr -d ' \r')
        KEY="${ADMIN_KEY:-$ADMIN_KEY_FROM_ENV}"
      fi
      if [ -n "${KEY:-}" ]; then
        info "Query stats:"
        curl -s "${APP_URL}/admin/stats" \
          -H "X-Admin-Key: ${KEY}" | python3 -m json.tool 2>/dev/null || \
          warn "Could not fetch stats (admin key wrong or server not ready)"
      else
        warn "CITYFOAM_ADMIN_KEY not set — skipping stats endpoint"
      fi
      exit 0
      ;;
    *)
      error "Unknown argument: $arg"
      echo "Usage: $0 [--rebuild] [--experiment] [--down] [--logs] [--status]"
      exit 1
      ;;
  esac
done

# =============================================================================
# STEP 1 — Pre-flight checks
# =============================================================================
section "Step 1 — Pre-flight checks"

# Docker running?
if ! docker info &>/dev/null; then
  error "Docker is not running. Start Docker Desktop and try again."
  exit 1
fi
success "Docker is running."

# docker compose available?
if ! docker compose version &>/dev/null; then
  error "docker compose not found. Install Docker Desktop >= 4.x."
  exit 1
fi
success "docker compose is available."

# .env file exists?
if [ ! -f ".env" ]; then
  if [ -f "_env" ]; then
    warn ".env not found — copying _env to .env"
    cp _env .env
    warn "Please edit .env and fill in AZURE_API_KEY, AZURE_ENDPOINT, etc."
    warn "Then re-run this script."
    exit 1
  else
    error "No .env or _env file found. Cannot start without configuration."
    exit 1
  fi
fi
success ".env file found."

# Read key values from .env for validation
AZURE_KEY=$(grep '^AZURE_API_KEY=' .env | cut -d= -f2 | tr -d ' \r' | sed 's/#.*//')
AZURE_EP=$(grep '^AZURE_ENDPOINT=' .env | cut -d= -f2 | tr -d ' \r' | sed 's/#.*//')
ADMIN_KEY_FROM_ENV=$(grep '^CITYFOAM_ADMIN_KEY=' .env | cut -d= -f2 | tr -d ' \r' | sed 's/#.*//')
LLM_PROVIDER=$(grep '^LLM_PROVIDER=' .env | cut -d= -f2 | tr -d ' \r' | sed 's/#.*//')
ADMIN_KEY="${CITYFOAM_ADMIN_KEY:-$ADMIN_KEY_FROM_ENV}"

# Validate required .env values
MISSING=false
if [ "${LLM_PROVIDER:-azure}" = "azure" ]; then
  if [ -z "$AZURE_KEY" ] || [ "$AZURE_KEY" = "your-key-here" ]; then
    error "AZURE_API_KEY is not set in .env"
    MISSING=true
  fi
  if [ -z "$AZURE_EP" ] || [ "$AZURE_EP" = "https://YOUR-RESOURCE-NAME.openai.azure.com/" ]; then
    error "AZURE_ENDPOINT is not set in .env"
    MISSING=true
  fi
fi
if [ -z "$ADMIN_KEY" ]; then
  warn "CITYFOAM_ADMIN_KEY is not set — /admin/* endpoints will return 503"
fi
if [ "$MISSING" = true ]; then
  error "Fix the above values in .env then re-run."
  exit 1
fi
success ".env values look good."

# data/ folder has files?
DATA_COUNT=$(find ./data -type f 2>/dev/null | wc -l)
if [ "$DATA_COUNT" -eq 0 ]; then
  error "data/ folder is empty. Add your KB files (refundpolicy.docx, zbranches.xlsx, knowledge_base.md)."
  exit 1
fi
success "data/ folder has ${DATA_COUNT} file(s)."

# =============================================================================
# STEP 2 — Build Docker image
# =============================================================================
section "Step 2 — Building Docker image"

if [ "$REBUILD" = true ]; then
  info "Force rebuild requested — clearing build cache..."
  $COMPOSE_CMD build --no-cache
else
  $COMPOSE_CMD build
fi
success "Docker image built."

# =============================================================================
# STEP 3 — Start all containers
# =============================================================================
section "Step 3 — Starting containers"

# Bring down any stale containers cleanly first
$COMPOSE_CMD down --remove-orphans 2>/dev/null || true

$COMPOSE_CMD up -d
success "Containers started."
echo ""
info "Services:"
echo "   App     → ${APP_URL}"
echo "   MLflow  → ${MLFLOW_URL}"

# =============================================================================
# STEP 4 — Wait for app to be healthy
# =============================================================================
section "Step 4 — Waiting for app to be ready"

info "Waiting for ${APP_URL}/api/health  (up to $((HEALTH_RETRIES * 5))s)..."
ATTEMPT=0
until curl -sf "${APP_URL}/api/health" &>/dev/null; do
  ATTEMPT=$((ATTEMPT + 1))
  if [ "$ATTEMPT" -ge "$HEALTH_RETRIES" ]; then
    error "App did not become healthy after $((HEALTH_RETRIES * 5))s."
    error "Check logs with:  docker compose logs -f app"
    exit 1
  fi
  printf "."
  sleep 5
done
echo ""

# Pretty-print the health response
HEALTH=$(curl -s "${APP_URL}/api/health")
success "App is healthy:"
echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"

# =============================================================================
# STEP 5 — Smoke test the chat endpoint
# =============================================================================
section "Step 5 — Smoke test"

info "Creating a test chat session..."
CHAT_RESP=$(curl -sf -X POST "${APP_URL}/api/chats" \
  -H "Content-Type: application/json" 2>/dev/null || echo "")

if [ -z "$CHAT_RESP" ]; then
  warn "Could not create chat session — check app logs."
else
  CHAT_ID=$(echo "$CHAT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('chat_id',''))" 2>/dev/null || echo "")
  if [ -n "$CHAT_ID" ]; then
    success "Chat session created (id: ${CHAT_ID:0:8}...)"
  else
    warn "Chat response was unexpected: $CHAT_RESP"
  fi
fi

# =============================================================================
# STEP 6 — Admin stats (if key is available)
# =============================================================================
section "Step 6 — Admin dashboard"

if [ -n "$ADMIN_KEY" ]; then
  info "Fetching query stats..."
  STATS=$(curl -sf "${APP_URL}/admin/stats" -H "X-Admin-Key: ${ADMIN_KEY}" 2>/dev/null || echo "")
  if [ -n "$STATS" ]; then
    success "Stats endpoint responding:"
    echo "$STATS" | python3 -m json.tool 2>/dev/null || echo "$STATS"
  else
    warn "Stats endpoint not responding yet — try again in a moment."
  fi

  info "Recent query logs (last 5):"
  LOGS=$(curl -sf "${APP_URL}/admin/logs?limit=5" -H "X-Admin-Key: ${ADMIN_KEY}" 2>/dev/null || echo "")
  if [ -n "$LOGS" ]; then
    echo "$LOGS" | python3 -m json.tool 2>/dev/null || echo "$LOGS"
  fi
else
  warn "Skipping admin stats — CITYFOAM_ADMIN_KEY not set in .env"
fi

# =============================================================================
# STEP 7 — RAG experiment sweep (optional, --experiment flag)
# =============================================================================
if [ "$RUN_EXPERIMENT" = true ]; then
  section "Step 7 — RAG Hyperparameter Sweep"
  info "Running experiment sweep inside container (this takes ~5 minutes)..."
  docker exec cityfoam-app python run_rag_experiment.py
  success "Experiment complete. View results at ${MLFLOW_URL}"
fi

# =============================================================================
# Final summary
# =============================================================================
section "Deployment Complete"
echo ""
echo -e "  ${GREEN}✔${RESET}  App         →  ${BOLD}${APP_URL}${RESET}"
echo -e "  ${GREEN}✔${RESET}  MLflow UI   →  ${BOLD}${MLFLOW_URL}${RESET}"
echo -e "  ${GREEN}✔${RESET}  Health      →  ${APP_URL}/api/health"
if [ -n "$ADMIN_KEY" ]; then
echo -e "  ${GREEN}✔${RESET}  Logs        →  ${APP_URL}/admin/logs  (X-Admin-Key: ${ADMIN_KEY:0:4}****)"
echo -e "  ${GREEN}✔${RESET}  Stats       →  ${APP_URL}/admin/stats (X-Admin-Key: ${ADMIN_KEY:0:4}****)"
echo -e "  ${GREEN}✔${RESET}  Re-ingest   →  POST ${APP_URL}/admin/reingest"
fi
echo ""
echo -e "  ${CYAN}Useful commands:${RESET}"
echo "    ./deploy.sh --logs        — tail live logs"
echo "    ./deploy.sh --status      — quick health + stats"
echo "    ./deploy.sh --experiment  — run RAG sweep"
echo "    ./deploy.sh --down        — stop everything"
echo "    ./deploy.sh --rebuild     — force full rebuild"
echo ""