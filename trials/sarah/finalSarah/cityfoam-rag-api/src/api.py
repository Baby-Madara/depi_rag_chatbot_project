import re
import json
import sys
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from src.config import Config

# 1. Initialize FastAPI and define security scheme
app = FastAPI(title="CityFoam Customer Support API", version="1.0.0")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != Config.CITYFOAM_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Access Denied: Invalid API Key")
    return api_key

# 2. Setup ChromaDB client and SentenceTransformer embedding model
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

# 3. Models and Helper Functions
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = [] 
    language: str = "auto"

def normalize_arabic(text: str) -> str:
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    return text

# 4. RAG Pipeline Function
def get_rag_response(user_query: str, history: list) -> str:
    clean_query = normalize_arabic(user_query)
    
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', clean_query))
    target_language = "ARABIC" if is_arabic else "ENGLISH"
    
    # Updated Prompt: No catalog, just asks for clean keywords in the original language
    router_prompt = f"""You are an advanced search router and query optimizer.
    Your job is to analyze the user's query and output a strict JSON object with two fields:
    
    1. "intent": Classify the query into EXACTLY one of these categories:
       - "branches" (if looking for store locations, addresses, phone numbers, or working hours)
       - "policy" (if looking for refund rules, returns, exchanges, or warranty)
       - "catalog" (if looking for mattress types, prices, specifications, or pillow configurations)
       
    2. "keywords": Clean, focused search keywords extracted from the user's query in its ORIGINAL LANGUAGE. Remove all conversational filler (like 'hello', 'please', 'how much').
    
    CRITICAL: Output ONLY valid raw JSON. No markdown wrappers, no code blocks."""
    
    try:
        router_output = llm_client.chat.completions.create(
            model=Config.AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": router_prompt},
                {"role": "user", "content": clean_query}
            ],
            temperature=0.0
        ).choices[0].message.content
        
        match = re.search(r'\{.*\}', router_output, re.DOTALL)
        if match:
            cleaned_json = match.group(0)
            route_data = json.loads(cleaned_json)
            
            intent = route_data.get("intent", "catalog").lower()
            optimized_query = route_data.get("keywords", clean_query)
        else:
            raise ValueError("No JSON found in LLM response")

    except Exception as e:
        print("\n" + "!"*60, flush=True)
        print("                 [!] ROUTER CRASHED [!]", flush=True)
        print(f" Error Reason  : {str(e)}", flush=True)
        print("!"*60 + "\n", flush=True)
        
        intent = "fallback"
        optimized_query = clean_query
    
    print("\n" + "="*60, flush=True)
    print("               STAGE 1: ROUTING ENGINE VERDICT", flush=True)
    print("="*60, flush=True)
    print(f" Raw User Query   : {user_query}", flush=True)
    print(f" Classified Intent: {intent.upper()}", flush=True)
    print(f" Search Keywords  : {optimized_query}", flush=True)
    print("="*60, flush=True)

    # 2. Retrieve Context
    query_vector = embedding_model.encode([optimized_query]).tolist()

    search_filter = {}
    if intent == "branches":
        search_filter = {"source": "branches.xlsx"}
    elif intent == "policy":
        search_filter = {"source": "refundpolicy.docx"}
    elif intent == "catalog":
        search_filter = {"source": "productsandprices.pdf"}

    # Execute search with dynamic limits
    if search_filter:
        retrieval_limit = 5 if intent == "branches" else 3
        results = collection.query(
            query_embeddings=query_vector, 
            n_results=retrieval_limit, 
            where=search_filter
        )
    else:
        results = collection.query(
            query_embeddings=query_vector, 
            n_results=5
        )
        
    context_string = "\n\n".join(results['documents'][0])

    # 3. Generate Final Answer
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

# 5. API Endpoints
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    try:
        answer = get_rag_response(request.query, request.history)
        print(f"\nDEBUG: Final Answer from LLM:\n{answer}\n", flush=True)
        return {
            "status": "success",
            "query": request.query,
            "response": answer
        }
    except Exception as e:
        print(f"API Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
