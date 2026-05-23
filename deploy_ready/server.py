"""
CityFoam RAG Chatbot — FastAPI Server (v3)
==========================================
Migration from Flask → FastAPI plus five new capabilities:

  1. FastAPI + async throughout — true async SSE via StreamingResponse,
     async LangChain .astream(), Pydantic request/response models.
  2. Re-ingestion admin endpoint — POST /admin/reingest triggers the
     pipeline in a BackgroundTask without restarting the server.
  3. Monitoring & observability — every query is timed and logged to
     SQLite (latency_ms, rag_score, language, optimized query).
     LatencyMiddleware adds X-Process-Time-Ms to every response.
  4. RAG quality evaluation — retrieval distances are scored; low-score
     queries get a caution note injected into the system prompt and a
     warning flag in the monitoring log.
  5. Secret management — all sensitive values are resolved via secrets.py
     which tries Azure Key Vault first, then falls back to .env.

Retained from v2:
  - Dual LLM provider: Ollama + Azure OpenAI (LangChain).
  - Query optimizer (LLM pre-cleans search query).
  - Language detection and lock (Arabic / English).
  - Multi-chat session management + SSE streaming.
  - Static file serving for the frontend.
"""

import json
import logging
import os
import re
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from typing import Optional

import chromadb
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from admin import router as admin_router
from ingest import normalize_arabic
from mlflow_tracker import init_mlflow, log_query_to_mlflow
from monitoring import LatencyMiddleware, QueryLogEntry, log_query
from rag_eval import evaluate_retrieval
#from secrets import get_secret
from app_secrets import get_secret
from scheduler import start_scheduler, stop_scheduler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("cityfoam.server")

# ---------------------------------------------------------------------------
# Load .env (local dev) — Key Vault takes priority in get_secret()
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------
PORT          = int(get_secret("PORT", "8000"))
LLM_PROVIDER  = get_secret("LLM_PROVIDER", "ollama")
COMPANY_NAME  = get_secret("COMPANY_NAME", "CityFoam")

REQUIRE_API_KEY     = get_secret("REQUIRE_API_KEY", "false").lower() == "true"
CITYFOAM_SECRET_KEY = get_secret("CITYFOAM_SECRET_KEY", "")

VALID_CATALOG = [
    "بيرلا بوكيت", "اريجاتو", "هيفين", "ريفيرا", "برنسيسه",
    "نيو ماريوت", "كوين", "لكشري بوكيت", "ماريوت قطن",
    "مون لند بوكيت", "امبر بوكيت", "كومفي", "دايموند بوكيت",
    "بيلو توب", "كونتور ميموري فوم",
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    chat_id: str


# ---------------------------------------------------------------------------
# Module-level singletons (populated in lifespan)
# ---------------------------------------------------------------------------
llm              = None
_embedding_model = None
_collection      = None


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def _build_llm():
    if LLM_PROVIDER == "ollama":
        logger.info("LLM provider: Ollama (%s)", get_secret("OLLAMA_MODEL", "qwen2.5:3b"))
        return ChatOllama(
            model=get_secret("OLLAMA_MODEL", "qwen2.5:3b"),
            base_url=get_secret("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    elif LLM_PROVIDER == "azure":
        logger.info("LLM provider: Azure OpenAI (%s)", get_secret("AZURE_DEPLOYMENT_NAME"))
        return AzureChatOpenAI(
            deployment_name=get_secret("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
            api_version=get_secret("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=get_secret("AZURE_ENDPOINT"),
            api_key=get_secret("AZURE_API_KEY"),
        )
    else:
        warnings.warn(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER!r}")
        return None


# ---------------------------------------------------------------------------
# Lifespan — init singletons once at startup, clean up on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, _embedding_model, _collection

    logger.info("-- Startup --")
    llm = _build_llm()

    logger.info("Loading embedding model...")
    _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    chroma_dir      = get_secret("CHROMA_DB_DIR", "./chroma_db")
    collection_name = get_secret("COLLECTION_NAME", "cityfoam_rag")
    logger.info("Connecting to ChromaDB at %s...", chroma_dir)
    _chroma_client = chromadb.PersistentClient(path=chroma_dir)
    _collection    = _chroma_client.get_or_create_collection(name=collection_name)
    logger.info("ChromaDB '%s' ready (%d chunks).", collection_name, _collection.count())
    init_mlflow(env=os.environ.get("ENV", "production"))
    start_scheduler(collection=_collection)
    logger.info("-- Ready --")

    yield

    stop_scheduler()
    logger.info("-- Shutdown --")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="CityFoam Support API", version="3.0.0", lifespan=lifespan)

# Add CORS Middleware to allow requests from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-Admin-Key"],  # Explicitly allow the custom admin key header
)

app.add_middleware(LatencyMiddleware)
app.include_router(admin_router)

# Dashboard — served directly from app root (not from static/)
_dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")

@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    return FileResponse(_dashboard_path)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        fp = os.path.join(static_dir, full_path)
        return FileResponse(fp) if os.path.isfile(fp) else FileResponse(
            os.path.join(static_dir, "index.html")
        )


# ---------------------------------------------------------------------------
# Optional API-key guard (dependency)
# ---------------------------------------------------------------------------
def _optional_api_key(x_api_key: Optional[str] = Header(default=None)):
    if REQUIRE_API_KEY and x_api_key != CITYFOAM_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")


# ---------------------------------------------------------------------------
# SQLite Chat Store
# ---------------------------------------------------------------------------
import sqlite3
from contextlib import contextmanager

CHATS_DB_PATH = "chats.db"

def _init_chats_db():
    with sqlite3.connect(CHATS_DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, data TEXT)"
        )
        conn.commit()

_init_chats_db()

@contextmanager
def _chat_db():
    conn = sqlite3.connect(CHATS_DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()

def get_all_chats():
    with _chat_db() as conn:
        rows = conn.execute("SELECT id, data FROM chats ORDER BY created_at DESC").fetchall()
        return [(row[0], json.loads(row[1])) for row in rows]

def get_chat(chat_id: str):
    with _chat_db() as conn:
        row = conn.execute("SELECT data FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return json.loads(row[0]) if row else None

def save_chat(chat_id: str, data: dict):
    with _chat_db() as conn:
        conn.execute("INSERT OR REPLACE INTO chats (id, data) VALUES (?, ?)", (chat_id, json.dumps(data)))
        conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_language(text: str) -> str:
    return "ARABIC" if re.search(r"[\u0600-\u06FF]", text) else "ENGLISH"


async def optimize_query_async(raw_query: str) -> str:
    """LLM pre-processes the query: strips filler, fixes typos, outputs Arabic."""
    if not llm:
        return raw_query
    optimizer_prompt = (
        "You are a search-engine query optimizer for a furniture and mattress store.\n"
        "- Strip conversational filler.\n"
        f"- Fix typos using this catalog: {VALID_CATALOG}\n"
        "- Output ARABIC keywords only. No explanation."
    )
    try:
        result = await llm.ainvoke([
            SystemMessage(content=optimizer_prompt),
            HumanMessage(content=normalize_arabic(raw_query)),
        ])
        optimized = result.content.strip()
        logger.info("Query optimizer: %r -> %r", raw_query[:60], optimized[:60])
        return optimized or raw_query
    except Exception as exc:
        logger.warning("Query optimizer failed (%s); using raw query.", exc)
        return raw_query


def retrieve_and_evaluate(query: str, n_results: int = 5):
    """Embed query, query ChromaDB, score retrieval quality."""
    vector  = _embedding_model.encode([normalize_arabic(query)]).tolist()
    results = _collection.query(
        query_embeddings=vector,
        n_results=n_results,
        include=["documents", "distances"],
    )
    docs      = results.get("documents", [[]])[0]
    distances = results.get("distances",  [[]])[0]
    return evaluate_retrieval(docs, distances, query=query)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status":        "ok",
        "llm_provider":  LLM_PROVIDER,
        "chroma_chunks": _collection.count() if _collection else 0,
    }


@app.get("/api/config")
async def get_config():
    return {
        "companyName":   COMPANY_NAME,
        "theme":         get_secret("THEME", "dark"),
        "enableSidebar": get_secret("ENABLE_SIDEBAR", "true").lower() == "true",
    }


@app.get("/api/chats")
async def get_chats():
    chats = [{"id": cid, "title": data["title"]} for cid, data in get_all_chats()]
    return {"status": "success", "chats": chats}


@app.post("/api/chats")
async def create_chat():
    chat_id = str(uuid.uuid4())
    save_chat(chat_id, {
        "title": "New Chat",
        "messages": [
            {"sender": "ai",
             "text": f"Hello! I'm the {COMPANY_NAME} support assistant. How can I help you today?"}
        ],
    })
    return {"status": "success", "chat_id": chat_id, "title": "New Chat"}


@app.get("/api/chats/{chat_id}")
async def get_chat_history(chat_id: str):
    chat_data = get_chat(chat_id)
    if not chat_data:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {"status": "success", "messages": chat_data["messages"]}


# ---------------------------------------------------------------------------
# Streaming chat — the main endpoint
# ---------------------------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(
    body:             ChatRequest,
    background_tasks: BackgroundTasks,
    _key:             None = Depends(_optional_api_key),
):
    """
    RAG-augmented streaming chat.

    Flow:
      1. Detect user language.
      2. Async query optimizer.
      3. RAG retrieval + quality evaluation.
      4. Build LangChain messages.
      5. Async-stream LLM response via SSE.
      6. Log query metadata to SQLite (background task).
    """
    user_message = body.message.strip()
    chat_id      = body.chat_id

    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message.")
    
    chat_data = get_chat(chat_id)
    if not chat_data:
        raise HTTPException(status_code=400, detail="Invalid chat_id.")
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not configured.")

    history = chat_data["messages"]

    # Auto-title
    title_updated = False
    if chat_data["title"] == "New Chat":
        chat_data["title"] = (
            user_message[:30] + "..." if len(user_message) > 30 else user_message
        )
        title_updated = True

    history.append({"sender": "user", "text": user_message})
    save_chat(chat_id, chat_data)

    # Language detection
    target_language = detect_language(user_message)

    # Query optimization
    optimized_query = await optimize_query_async(user_message)

    # RAG retrieval + quality scoring
    retrieval      = retrieve_and_evaluate(optimized_query)
    context_string = retrieval.context_str

    logger.info(
        "RAG: score=%.3f  poor=%s  chars=%d  query=%r",
        retrieval.quality_score, retrieval.is_poor,
        len(context_string), optimized_query[:60],
    )

    # Build LangChain messages
    system_content = f"""You are the official Customer Support AI Assistant for {COMPANY_NAME}.
Answer accurately based STRICTLY on the Knowledge Base below.

CRITICAL RULES:
1. ZERO HALLUCINATION - only use facts from the KNOWLEDGE BASE CONTEXT.
2. UNKNOWN ANSWERS - if not in KB, say so and offer a human agent.
3. LANGUAGE LOCK - reply ENTIRELY in {target_language}. No mixing.
4. CURRENCY - prices are in Egyptian Pounds (EGP).
5. OCR TYPOS - interpret PDF artifacts gracefully.

--- KNOWLEDGE BASE CONTEXT ---
{context_string}
--- END OF CONTEXT ---"""

    lc_messages = [SystemMessage(content=system_content)]
    prior = history[:-1]
    if len(prior) > 10:
        prior = prior[:2] + prior[-8:]
    for msg in prior:
        if msg["sender"] == "user":
            lc_messages.append(HumanMessage(content=msg["text"]))
        elif msg["sender"] == "ai":
            lc_messages.append(AIMessage(content=msg["text"]))
    lc_messages.append(HumanMessage(content=user_message))

    request_start = time.perf_counter()

    async def event_generator():
        ai_text = ""
        try:
            async for chunk in llm.astream(lc_messages):
                content = chunk.content
                if content:
                    ai_text += content
                    yield f"data: {json.dumps({'chunk': content})}\n\n"

            history.append({"sender": "ai", "text": ai_text})
            save_chat(chat_id, chat_data)

            if title_updated:
                yield f"data: {json.dumps({'title': chat_data['title']})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.exception("LLM streaming error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        finally:
            elapsed_ms = (time.perf_counter() - request_start) * 1000
            entry = QueryLogEntry(
                chat_id         = chat_id,
                raw_query       = user_message,
                optimized_query = optimized_query,
                language        = target_language,
                response_chars  = len(ai_text),
                latency_ms      = elapsed_ms,
                rag_score       = retrieval.quality_score,
                rag_warning     = retrieval.is_poor,
            )
            # Log to SQLite (fast, local)
            background_tasks.add_task(log_query, entry)
            # Log to MLflow / Azure ML (async-safe background task)
            background_tasks.add_task(
                log_query_to_mlflow,
                chat_id         = chat_id,
                raw_query       = user_message,
                optimized_query = optimized_query,
                language        = target_language,
                rag_score       = retrieval.quality_score,
                rag_poor        = retrieval.is_poor,
                latency_ms      = elapsed_ms,
                response_chars  = len(ai_text),
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True)