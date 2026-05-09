import os
import re
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from dotenv import load_dotenv  # NEW: For secure environment variables

# ==========================================
# 1. INITIALIZATION & SECURITY SETUP
# ==========================================
# Load the hidden variables from your .env file
load_dotenv()

app = FastAPI(title="CityFoam Customer Support API", version="1.2")

# Securely fetch API Key from .env (fallback to a default just in case)
CITYFOAM_SECRET_KEY = os.getenv("CITYFOAM_SECRET_KEY", "cf_secure_key_2026") 
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    """Validates the key. If it doesn't match, it blocks the request."""
    if api_key != CITYFOAM_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Access Denied: Invalid API Key")
    return api_key

# ==========================================
# 2. SETUP AI AND DATABASE CLIENTS
# ==========================================
print("Loading Embedding Model...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Connecting to ChromaDB...")
db_path = os.path.join(os.getcwd(), "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)
# UPDATED: Pointing to the clean, rebuilt database!
collection = chroma_client.get_collection(name="cityfoam_rag")

print("Connecting to Azure OpenAI...")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "cityfoam-gpt")

if not AZURE_API_KEY:
    print("⚠️ WARNING: Azure API Key not found in .env file!")

llm_client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

# ==========================================
# 3. HELPER FUNCTIONS & DATA STRUCTURES
# ==========================================
class ChatRequest(BaseModel):
    query: str
    language: str = "auto"

VALID_CATALOG = ["بيرلا بوكيت", "أريجاتو", "هيفين", "ريفيرا", "برنسيسة", "نيو ماريوت", "كوين"]

def normalize_arabic(text: str) -> str:
    """Standardizes Arabic characters to eliminate common spelling mismatches."""
    text = re.sub(r'[أإآ]', 'ا', text) # Normalize all Alefs
    text = re.sub(r'ة', 'ه', text)     # Convert Taa Marbouta to Haa
    text = re.sub(r'ى', 'ي', text)     # Convert Alef Maksoura to Yaa
    text = re.sub(r'[\u064B-\u065F]', '', text) # Remove all Harakat (Tashkeel)
    return text

# ==========================================
# 4. THE CORE GENERATION LOGIC (Advanced RAG)
# ==========================================
def get_rag_response(user_query: str):
    # Step A: Normalize the user query to catch basic typos
    clean_query = normalize_arabic(user_query)

    # Step B: The Query Optimizer (Strips chatty text and fixes catalog typos)
    optimizer_prompt = f"""You are a search engine query optimizer. Extract the core product name and features from the user's query.
    - Remove conversational filler.
    - FIX TYPOS: Compare the user's query against this official catalog: {VALID_CATALOG}
    - If the user types something similar or with transposed letters (e.g. "بيرال" -> "بيرلا"), output the exact matching name from the catalog.
    Output ONLY the raw, corrected keywords for the database search."""
    
    try:
        optimized_query = llm_client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": optimizer_prompt},
                {"role": "user", "content": clean_query}
            ],
            temperature=0.0
        ).choices[0].message.content
    except Exception:
        optimized_query = clean_query # Fallback if optimizer fails

    # Step C: Retrieve Context
    query_vector = embedding_model.encode([optimized_query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=5)
    
    context_string = "\n\n".join(results['documents'][0])
        
    # Step D: Generate the Final Answer
    system_prompt = f"""You are the official Customer Support AI Assistant for CityFoam. 
Your primary function is to provide accurate, helpful answers based STRICTLY on the official Knowledge Base provided below.

CRITICAL RULES (ABSOLUTE COMPLIANCE REQUIRED):
1. ZERO HALLUCINATION: You must only use the facts, prices, and policies explicitly stated in the 'KNOWLEDGE BASE CONTEXT'. 
2. UNKNOWN ANSWERS: If the answer cannot be confidently found in the context, you MUST NOT guess. Politely state that you do not have that information and offer to transfer them to a human agent.
3. STRICT BILINGUAL MATCHING: You must reply in the EXACT SAME LANGUAGE as the user's question. 
4. FIX TYPOS: The context is extracted from PDFs and contains OCR spelling typos (e.g., "بيرال"). You MUST realize this means "بيرلا" and correct it gracefully in your final response.

KNOWLEDGE BASE CONTEXT:
{context_string}
"""

    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query} # Use original query so it sounds natural
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

# ==========================================
# 5. API ENDPOINTS
# ==========================================
@app.get("/")
def health_check():
    return {"status": "CityFoam Support API is securely running."}

@app.post("/chat")
def chat_endpoint(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    try:
        answer = get_rag_response(request.query)
        return {
            "status": "success",
            "query": request.query,
            "response": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))