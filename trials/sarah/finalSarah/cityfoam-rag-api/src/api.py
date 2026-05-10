import re
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
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

# ==========================================
# 3. HELPER FUNCTIONS & DATA STRUCTURES
# ==========================================
class ChatRequest(BaseModel):
    query: str
    language: str = "auto"

# Catalog used by the LLM to fix OCR typos
VALID_CATALOG = ["بيرلا بوكيت", "أريجاتو", "هيفين", "ريفيرا", "برنسيسة", "نيو ماريوت", "كوين"]

def normalize_arabic(text: str) -> str:
    """Standardizes Arabic characters to eliminate common spelling mismatches."""
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    return text

# ==========================================
# 4. RAG LOGIC
# ==========================================
def get_rag_response(user_query: str) -> str:
    clean_query = normalize_arabic(user_query)

    # # The Query Optimizer: Strips chatty text and fixes catalog typos
    # optimizer_prompt = f"""You are a search engine query optimizer. Extract the core product name and features from the user's query.
    # - Remove conversational filler.
    # - FIX TYPOS: Compare the user's query against this official catalog: {VALID_CATALOG}
    # Output ONLY the raw, corrected keywords for the database search."""
    
    # try:
    #     optimized_query = llm_client.chat.completions.create(
    #         model=Config.AZURE_DEPLOYMENT_NAME,
    #         messages=[
    #             {"role": "system", "content": optimizer_prompt},
    #             {"role": "user", "content": clean_query}
    #         ],
    #         temperature=0.0
    #     ).choices[0].message.content
    # except Exception:
    #     optimized_query = clean_query # Fallback if optimizer fails

    # Retrieve Context from Vector DB
    query_vector = embedding_model.encode([clean_query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=5)
    
    context_string = "\n\n".join(results['documents'][0])
        
    # Generate the Final Answer
    system_prompt = f"""You are the official Customer Support AI Assistant for CityFoam. 
Your primary function is to provide accurate, helpful answers based STRICTLY on the official Knowledge Base provided below.

CRITICAL RULES:
1. ZERO HALLUCINATION: You must only use facts stated in the 'KNOWLEDGE BASE CONTEXT'. 
2. UNKNOWN ANSWERS: If the answer cannot be found, politely state that you do not have that information and offer to transfer them to a human agent.
3. STRICT BILINGUAL MATCHING: You must reply in the EXACT SAME LANGUAGE as the user's question. 
4. FIX TYPOS: The context is extracted from PDFs and may contain OCR typos. Correct them gracefully.

KNOWLEDGE BASE CONTEXT:
{context_string}
"""

    response = llm_client.chat.completions.create(
        model=Config.AZURE_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clean_query} # Use original query so it sounds natural
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

# ==========================================
# 5. API ENDPOINTS
# ==========================================
@app.get("/")
def health_check():
    """Simple endpoint to verify the API is running."""
    return {"status": "CityFoam Support API is securely running."}

@app.post("/chat")
def chat_endpoint(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """Main chat endpoint that processes the RAG pipeline."""
    try:
        answer = get_rag_response(request.query)
        return {
            "status": "success",
            "query": request.query,
            "response": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))