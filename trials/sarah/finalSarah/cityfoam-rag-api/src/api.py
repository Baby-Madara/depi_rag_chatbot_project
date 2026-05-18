import os
import re
import json
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from src.config import Config

Config.load_pso_params()

# ==========================================
# 1. INITIALIZATION & SECURITY SETUP
# ==========================================
app = FastAPI(title="CityFoam Customer Support API", version="1.0.0")

# Security header definition
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    """Validates the API key. Blocks the request if it doesn't match."""
    if api_key != Config.CITYFOAM_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Access Denied: Invalid API Key")
    return api_key

# ==========================================
# 2. SETUP AI AND DATABASE CLIENTS
# ==========================================
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)

try:
    collection = chroma_client.get_collection(name=Config.COLLECTION_NAME)
except ValueError:
    raise RuntimeError(
        f"Collection '{Config.COLLECTION_NAME}' not found. Please run ingest.py first."
    )

llm_client = AzureOpenAI(
    azure_endpoint=Config.AZURE_ENDPOINT,
    api_key=Config.AZURE_API_KEY,
    api_version=Config.AZURE_API_VERSION,
)

# ==========================================
# 3. HELPER FUNCTIONS & DATA STRUCTURES
# ==========================================
class ChatMessage(BaseModel):
    role: str      # Either "user" or "assistant"
    content: str   # The text of the message
class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []
    language: str = "auto"

# Catalog used by the LLM to fix OCR typos
VALID_CATALOG = [
    "بيرلا بوكيت",
    "أريجاتو",
    "هيفين",
    "ريفيرا",
    "برنسيسة",
    "نيو ماريوت",
    "كوين",
]

def normalize_arabic(text: str) -> str:
    """Standardizes Arabic characters to eliminate common spelling mismatches."""
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[\u064B-\u065F]", "", text)
    return text

# ==========================================
# 4. RAG LOGIC (Optimized & Verified with PSO Configuration)
# ==========================================
def get_rag_response(user_query: str, history: list) -> tuple:
    """
    Processes query via RAG pipeline using PSO weights.
    Returns a tuple: (final_answer, optimized_keyword)
    """
    clean_query = normalize_arabic(user_query)

    # Detect if the user is typing in Arabic using Regex
    is_arabic = bool(re.search(r"[\u0600-\u06FF]", user_query))
    target_language = "ARABIC" if is_arabic else "ENGLISH"

    # 1. Optimize the search query using LLM
    optimizer_prompt = f"""You are a search engine query optimizer. Extract the core product name and features from the user's query.
    - Remove conversational filler.
    - FIX TYPOS: Compare the user's query against this official catalog: {VALID_CATALOG}
    - CRITICAL: You MUST output the final keywords in ARABIC. Do not translate them to English.
    Output ONLY the raw, corrected keywords for the database search."""

    try:
        optimized_query = (
            llm_client.chat.completions.create(
                model=Config.AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": optimizer_prompt},
                    {"role": "user", "content": clean_query},
                ],
                temperature=0.0,
            )
            .choices[0]
            .message.content
        )
    except Exception:
        optimized_query = clean_query

    # 2. Retrieve Context — Single query execution using PSO-optimized TOP_K
    query_vector = embedding_model.encode([optimized_query]).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=int(Config.TOP_K),
        include=["documents", "distances", "metadatas"],
    )
    
    # 3. Rank results using PSO-optimized weights
    candidates = []
    if results and results["documents"] and len(results["documents"][0]) > 0:
        for doc, distance, meta in zip(
            results["documents"][0], results["distances"][0], results["metadatas"][0]
        ):
            semantic_score = 1 / (1 + distance)

            # Filter by PSO similarity threshold dynamic constraint
            if semantic_score < Config.SIMILARITY_THRESHOLD:
                continue

            query_tokens = set(optimized_query.lower().split())
            doc_tokens = set(doc.lower().split())
            keyword_score = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            recency_score = float(meta.get("recency", 0.5))
            authority_score = float(meta.get("authority", 0.5))

            # Calculation of PSO weighted convergence function
            final_score = (
                Config.WEIGHT_SEMANTIC * semantic_score
                + Config.WEIGHT_KEYWORD * keyword_score
                + Config.WEIGHT_RECENCY * recency_score
                + Config.WEIGHT_AUTHORITY * authority_score
            )
            candidates.append((final_score, doc))

    candidates.sort(key=lambda x: x[0], reverse=True)
    context_string = (
        "\n\n".join(doc for _, doc in candidates)
        if candidates
        else "No relevant context found."
    )

    # 4. Generate Final Answer with Language Lock
    system_prompt = f"""You are an expert AI Sales Assistant for CityFoam, a premium mattress and furniture company. 
Your goal is to answer user queries using ONLY the provided context from the database.

When formatting your response, always strictly follow these professional guidelines:
1. Start with a welcoming and enthusiastic tone (e.g., Use emojis like 🎉, 🛏️, 📏 contextually).
2. Present products in a clean, scannable, bulleted or numbered list.
3. Highlight pricing clearly. If a discount is available, explicitly mention the saving (e.g., "Price: X EGP (Save money! Original price was Y EGP)").
4. Clearly state the primary health or comfort benefit of the product under a "Best For:" label.
5. Always end your response with a strong, proactive Call to Action (CTA) that guides the customer to the next step, such as asking if they want to complete the order, or offering details about shipping and warranty policies.

If the context does not contain the answer, politely inform the user that you don't have this information right now. Do not make up prices or specifications.
 
CRITICAL RULES:
1. ZERO HALLUCINATION: You must only use facts stated in the 'KNOWLEDGE BASE CONTEXT'. 
2. UNKNOWN ANSWERS: If the answer cannot be found, politely state that you do not have that information.
3. LANGUAGE LOCK: You MUST write your ENTIRE reply strictly in {target_language}. Even if the knowledge base context is in a different language, you must translate your final answer to {target_language} and NEVER mix languages.
4. CONTEXT AWARENESS: Use the previous conversation history to understand pronouns or follow-up questions.

KNOWLEDGE BASE CONTEXT:
{context_string}
"""
    #------------------------------
    # 2. Retrieve Context
    query_vector = embedding_model.encode([optimized_query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=5)
    context_string = "\n\n".join(results['documents'][0])
    
    # ADD THIS LINE RIGHT HERE:
    print(f"\n===== WHAT THE BOT IS READING =====\n{context_string}\n===================================\n")
    # Build the message chain: System -> Past History -> New Query
    messages = [{"role": "system", "content": system_prompt}]
    # Add previous chat history
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

   
        
    # Add the current user question
    messages.append({"role": "user", "content": user_query})

    response = llm_client.chat.completions.create(
        model=Config.AZURE_DEPLOYMENT_NAME, messages=messages, temperature=0.0
    )
    
    return response.choices[0].message.content, optimized_query


# ==========================================
# 5. API ENDPOINTS & LOGGING MECHANISM
# ==========================================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that processes RAG and dynamically logs real interactions."""
    try:
        

        Config.load_pso_params(verbose=False)
        
        # 1. Execute the correct pipeline loop
        answer, extracted_keyword = get_rag_response(request.query, request.history)
        
        # 2. DYNAMIC LOGGING FOR THE PSO LOOP
        # Automatically updates the user_queries.json log structure for the background swarm
        log_file = "./user_queries.json"
        new_interaction = {"query": request.query, "expected": extracted_keyword}
        
        existing_logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    existing_logs = json.load(f)
                    if not isinstance(existing_logs, list):
                        existing_logs = []
            except Exception:
                existing_logs = []
        
        existing_logs.append(new_interaction)
        
        # Save interaction history back to file cleanly
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(existing_logs, f, indent=2, ensure_ascii=False)
            
        return {"status": "success", "query": request.query, "response": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
