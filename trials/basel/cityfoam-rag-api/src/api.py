import re
import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from src.config import Config

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
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# تعديل DevOps: التأكد من مسار قاعدة البيانات جوه الحاوية
db_path = os.getenv("CHROMA_DB_DIR", Config.CHROMA_DB_DIR)
chroma_client = chromadb.PersistentClient(path=db_path)

try:
    collection = chroma_client.get_collection(name=Config.COLLECTION_NAME)
except ValueError:
    # تعديل بسيط لضمان عدم توقف السيرفر إذا كانت الـ DB فارغة في أول مرة للـ Docker
    print(f"Warning: Collection '{Config.COLLECTION_NAME}' not found.")
    collection = None 

llm_client = AzureOpenAI(
    azure_endpoint=Config.AZURE_ENDPOINT,
    api_key=Config.AZURE_API_KEY,
    api_version=Config.AZURE_API_VERSION
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

VALID_CATALOG = ["بيرلا بوكيت", "أريجاتو", "هيفين", "ريفيرا", "برنسيسة", "نيو ماريوت", "كوين"]

def normalize_arabic(text: str) -> str:
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    return text

# ==========================================
# 4. RAG LOGIC
# ==========================================
def get_rag_response(user_query: str, history: list) -> str:
    if collection is None:
        return "عذراً، نظام البيانات قيد التحديث حالياً."
        
    clean_query = normalize_arabic(user_query)
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', user_query))
    target_language = "ARABIC" if is_arabic else "ENGLISH"

    optimizer_prompt = f"""You are a search engine query optimizer. Extract the core product name and features from the user's query.
    - Remove conversational filler.
    - FIX TYPOS: Compare the user's query against this official catalog: {VALID_CATALOG}
    Output ONLY the raw, corrected keywords for the database search."""
    
    try:
        optimized_query = llm_client.chat.completions.create(
            model=Config.AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": optimizer_prompt},
                {"role": "user", "content": clean_query}
            ],
            temperature=0.0
        ).choices[0].message.content
    except Exception:
        optimized_query = clean_query

    query_vector = embedding_model.encode([optimized_query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=5)
    context_string = "\n\n".join(results['documents'][0])
        
    system_prompt = f"""You are the official Customer Support AI Assistant for CityFoam. 
Your primary function is to provide accurate, helpful answers based STRICTLY on the official Knowledge Base provided below.
CRITICAL RULES:
1. ZERO HALLUCINATION.
2. UNKNOWN ANSWERS: Polite refusal.
3. LANGUAGE LOCK: {target_language}.
KNOWLEDGE BASE CONTEXT:
{context_string}
"""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_query})

    response = llm_client.chat.completions.create(
        model=Config.AZURE_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.0
    )
    return response.choices[0].message.content

# ==========================================
# 5. API ENDPOINTS
# ==========================================
# ملاحظة DevOps: التأكد من وجود فولدر static
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "CityFoam API is running. UI file not found in static/"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        answer = get_rag_response(request.query, request.history)
        return {
            "status": "success",
            "query": request.query,
            "response": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
