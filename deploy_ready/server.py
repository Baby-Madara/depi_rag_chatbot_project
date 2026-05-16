"""
CityFoam RAG Chatbot — Flask Server
=====================================
Fixes applied vs. original:
  1. ChromaDB client + embedding model are now module-level singletons (not re-created per request).
  2. LLM is called with the proper lc_messages list, not a raw system-prompt string.
     RAG context lives in SystemMessage; conversation history follows as Human/AI turns.
  3. AZURE_BASE_URL → AZURE_ENDPOINT (matches .env).
  4. Duplicate imports + double clean_query removed.
  5. Static files served from ./static/ (matches Flask config).
  6. Added /api/health endpoint for Docker health-check.
  7. Debug print statements replaced with structured logging.
"""

import os
import json
import uuid
import logging
import warnings

import chromadb
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer

from ingest import normalize_arabic
from config import Config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("cityfoam.server")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env"))

PORT            = int(os.environ.get("PORT", 8000))
LLM_PROVIDER    = os.environ.get("LLM_PROVIDER", "ollama")   # "ollama" | "azure"
COMPANY_NAME    = os.environ.get("COMPANY_NAME", "AI Chatbot")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="")

# ---------------------------------------------------------------------------
# LLM — initialised once at startup
# ---------------------------------------------------------------------------
def _build_llm():
    if LLM_PROVIDER == "ollama":
        logger.info("LLM provider: Ollama (%s)", os.environ.get("OLLAMA_MODEL"))
        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    elif LLM_PROVIDER == "azure":
        logger.info("LLM provider: Azure OpenAI (%s)", os.environ.get("AZURE_DEPLOYMENT_NAME"))
        return AzureChatOpenAI(
            deployment_name=os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
            api_version=os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ.get("AZURE_ENDPOINT"),   # ← matches .env
            api_key=os.environ.get("AZURE_API_KEY"),
        )
    else:
        warnings.warn(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
        return None

llm = _build_llm()

# ---------------------------------------------------------------------------
# RAG — ChromaDB client + embedding model as module-level singletons
# ---------------------------------------------------------------------------
logger.info("Loading embedding model (paraphrase-multilingual-MiniLM-L12-v2)…")
_embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

logger.info("Connecting to ChromaDB at %s…", Config.CHROMA_DB_DIR)
_chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)
_collection    = _chroma_client.get_or_create_collection(name=Config.COLLECTION_NAME)
logger.info("ChromaDB collection '%s' ready (%d chunks).",
            Config.COLLECTION_NAME, _collection.count())


def retrieve_context(query: str, n_results: int = 5) -> str:
    """Embed the query and return the top-n matching chunks as a single string."""
    clean_q  = normalize_arabic(query)
    vector   = _embedding_model.encode([clean_q]).tolist()
    results  = _collection.query(query_embeddings=vector, n_results=n_results)
    docs     = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(docs)


# ---------------------------------------------------------------------------
# In-memory chat store
# ---------------------------------------------------------------------------
# Structure: { user_id: { chat_id: { "title": str, "messages": [...] } } }
db = {"guest": {}}

# ---------------------------------------------------------------------------
# Routes — static
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    """Used by Docker health-check."""
    return jsonify({
        "status": "ok",
        "llm_provider": LLM_PROVIDER,
        "chroma_chunks": _collection.count(),
    })


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "companyName": COMPANY_NAME,
        "theme": os.environ.get("THEME", "dark"),
        "enableSidebar": os.environ.get("ENABLE_SIDEBAR", "true").lower() == "true",
    })


@app.route("/api/chats", methods=["GET"])
def get_chats():
    user_id = "guest"
    chats = [
        {"id": cid, "title": info["title"]}
        for cid, info in db.get(user_id, {}).items()
    ]
    chats.reverse()   # newest first
    return jsonify({"status": "success", "chats": chats})


@app.route("/api/chats", methods=["POST"])
def create_chat():
    user_id = "guest"
    chat_id = str(uuid.uuid4())
    db[user_id][chat_id] = {
        "title": "New Chat",
        "messages": [
            {"sender": "ai", "text": f"Hello! I'm the {COMPANY_NAME} support assistant. How can I help you today?"}
        ],
    }
    return jsonify({"status": "success", "chat_id": chat_id, "title": "New Chat"})


@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat_history(chat_id):
    user_id = "guest"
    if chat_id not in db.get(user_id, {}):
        return jsonify({"status": "error", "error": "Chat not found"}), 404
    return jsonify({"status": "success", "messages": db[user_id][chat_id]["messages"]})


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """
    RAG-augmented streaming chat endpoint.

    Flow:
      1. Retrieve relevant KB chunks for the user query.
      2. Build a proper LangChain message list:
             [SystemMessage(rag_context), ...history..., HumanMessage(query)]
      3. Stream the LLM response chunk-by-chunk via SSE.
    """
    data         = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    chat_id      = data.get("chat_id", "")
    user_id      = "guest"

    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    if not chat_id or chat_id not in db.get(user_id, {}):
        return jsonify({"error": "Invalid chat_id"}), 400
    if not llm:
        return jsonify({"error": "LLM not configured"}), 500

    history = db[user_id][chat_id]["messages"]

    # --- Auto-title on first real message ---
    title_updated = False
    if db[user_id][chat_id]["title"] == "New Chat":
        db[user_id][chat_id]["title"] = (user_message[:30] + "…") if len(user_message) > 30 else user_message
        title_updated = True

    # --- Append user message to history ---
    history.append({"sender": "user", "text": user_message})

    # --- RAG retrieval ---
    context_string = retrieve_context(user_message)
    logger.info("RAG: retrieved %d chars of context for query: %r",
                len(context_string), user_message[:60])

    # --- Build LangChain message list ---
    system_content = f"""You are the official Customer Support AI Assistant for {COMPANY_NAME}.
Your primary function is to provide accurate, helpful answers based STRICTLY on the Knowledge Base below.

CRITICAL RULES:
1. ZERO HALLUCINATION — only state facts found in the KNOWLEDGE BASE CONTEXT.
2. UNKNOWN ANSWERS — if the answer is not in the KB, politely say so and offer to connect the customer with a human agent.
3. BILINGUAL — always reply in the EXACT SAME LANGUAGE the customer used (Arabic or English).
4. OCR TYPOS — the context may contain PDF extraction artifacts; interpret them gracefully.
5. CURRENCY — prices are in Egyptian Pounds (EGP) unless stated otherwise.

--- KNOWLEDGE BASE CONTEXT ---
{context_string}
--- END OF CONTEXT ---"""

    lc_messages = [SystemMessage(content=system_content)]

    # Inject trimmed conversation history (exclude the message we just appended)
    prior_history = history[:-1]
    if len(prior_history) > 10:
        # Keep first 2 (opening pleasantries) + last 8 (recent context)
        prior_history = prior_history[:2] + prior_history[-8:]

    for msg in prior_history:
        if msg["sender"] == "user":
            lc_messages.append(HumanMessage(content=msg["text"]))
        elif msg["sender"] == "ai":
            lc_messages.append(AIMessage(content=msg["text"]))

    # Current user turn
    lc_messages.append(HumanMessage(content=user_message))

    # --- SSE streaming generator ---
    def generate():
        ai_response_text = ""
        try:
            for chunk in llm.stream(lc_messages):
                content = chunk.content
                if content:
                    ai_response_text += content
                    yield f"data: {json.dumps({'chunk': content})}\n\n"

            # Persist AI reply
            history.append({"sender": "ai", "text": ai_response_text})

            if title_updated:
                yield f"data: {json.dumps({'title': db[user_id][chat_id]['title']})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.exception("LLM streaming error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(stream_with_context(generate()), content_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry-point (dev only — production uses gunicorn via entrypoint.sh)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting dev server on port %d (provider: %s)", PORT, LLM_PROVIDER)
    app.run(host="0.0.0.0", port=PORT, debug=True)