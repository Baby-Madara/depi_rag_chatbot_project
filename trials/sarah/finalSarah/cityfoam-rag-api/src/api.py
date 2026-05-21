import re
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from src.config import Config

#1. Initialize FastAPI and define security scheme
app = FastAPI(title="CityFoam Customer Support API", version="1.0.0")

# Security header definition
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    """Validates the API key. Blocks the request if it doesn't match."""
    if api_key != Config.CITYFOAM_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Access Denied: Invalid API Key")
    return api_key

#2. Setup ChromaDB client and SentenceTransformer embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)

try:
    collection = chroma_client.get_collection(name=Config.COLLECTION_NAME)
except ValueError:
    raise RuntimeError(f"Collection '{Config.COLLECTION_NAME}' not found. Please run ingest.py first.")

llm_client = AzureOpenAI(
    azure_endpoint=Config.AZURE_ENDPOINT,
    api_key=Config.AZURE_API_KEY,
    api_version=Config.AZURE_API_VERSION
)

#3. Helper function to normalize Arabic text
class ChatMessage(BaseModel):
    role: str      # Either "user" or "assistant"
    content: str   # The text of the message
class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = [] 
    language: str = "auto"


def normalize_arabic(text: str) -> str:
    """Standardizes Arabic characters to eliminate common spelling mismatches."""
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    return text

# RAG Pipeline Function

# Catalog used by the LLM to fix OCR typos
VALID_CATALOG = ["بيرلا بوكيت", "أريجاتو", "هيفين", "ريفيرا", "برنسيسة", "نيو ماريوت", "كوين"]
def get_rag_response(user_query: str, history: list) -> str:
    clean_query = normalize_arabic(user_query)

    # NEW: Detect if the user is typing in Arabic using Regex
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', user_query))
    target_language = "ARABIC" if is_arabic else "ENGLISH"
    optimizer_prompt = f"""You are a search engine query optimizer. Extract the core product name and features from the user's query.
    - Remove conversational filler.
    - FIX TYPOS: Compare the user's query against this official catalog: {VALID_CATALOG}
    - CRITICAL: You MUST output the final keywords in ARABIC. Do not translate them to English.
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

    # ---> ADD THESE TWO DEBUG LINES HERE <---
    print(f"\nDEBUG: Original User Query: {user_query}")
    print(f"DEBUG: Optimized Search Query: {optimized_query}\n")

    # 2. Retrieve Context
    query_vector = embedding_model.encode([optimized_query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=5)
    context_string = "\n\n".join(results['documents'][0])
        
    # 3. Generate Final Answer with History & Language Lock
    system_prompt = f"""You are the official Customer Support AI Assistant for CityFoam. 
Your primary function is to provide accurate, helpful answers based STRICTLY on the official Knowledge Base provided below.

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
    #print(f"\n===== WHAT THE BOT IS READING =====\n{context_string}\n===================================\n")
    # Build the message chain: System -> Past History -> New Query
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add previous chat history
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
        
    # Add the current user question
    messages.append({"role": "user", "content": user_query})

    response = llm_client.chat.completions.create(
        model=Config.AZURE_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.0
    )
    return response.choices[0].message.content

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")
@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")
# def health_check():
#     """Simple endpoint to verify the API is running."""
#     return {"status": "CityFoam Support API is securely running."}

@app.post("/api/chat")
# def chat_endpoint(request: ChatRequest, api_key: str = Depends(verify_api_key)):
def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that processes the RAG pipeline."""
    try:
        # Notice we are passing BOTH the query and the history here!
        answer = get_rag_response(request.query, request.history)
        print(f"\nDEBUG: Final Answer from LLM:\n{answer}\n")
        return {
            "status": "success",
            "query": request.query,
            "response": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))