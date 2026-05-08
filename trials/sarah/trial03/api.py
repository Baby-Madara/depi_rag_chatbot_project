import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# 1. Initialize API App
app = FastAPI(title="CityFoam Customer Support API", version="1.1")

# ==========================================
# NEW: SECURITY SETUP
# ==========================================
# In a real production environment, you would load this from a hidden .env file!
CITYFOAM_SECRET_KEY = "cf_secure_key_2026" 
API_KEY_NAME = "X-API-Key"

# This tells FastAPI to look for 'X-API-Key' in the headers of every request
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    """Validates the key. If it doesn't match, it blocks the request."""
    if api_key != CITYFOAM_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Access Denied: Invalid API Key")
    return api_key


# 2. Setup AI and Database Clients
print("Loading Embedding Model...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Connecting to ChromaDB...")
db_path = os.path.join(os.getcwd(), "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(name="customer_support_knowledge")


AZURE_ENDPOINT = "https://YOUR_RESOURCE_NAME.openai.azure.com/"
AZURE_API_KEY = "YOUR_API_KEY"
DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME" # e.g., "gpt-4o"


print("Connecting to Azure OpenAI...")
llm_client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview"
)

# 3. Define the Request Data Structure
class ChatRequest(BaseModel):
    query: str
    language: str = "auto"

# 4. The Core Generation Logic
def get_rag_response(user_query: str):
    query_vector = embedding_model.encode([user_query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=5)
    
    context = ""
    for i, text in enumerate(results['documents'][0]):
        context += f"[Document {i+1}]: {text}\n\n"
        
    prompt = f"""You are a professional Customer Support AI for "CityFoam", a mattress and furniture company in Egypt.
You must answer the user's question based ONLY on the context provided below. 
If the answer is not in the context, politely say "I do not have that information, but I can transfer you to a human agent." (or the Arabic equivalent).
Do not make up prices, policies, or branch locations.

CRITICAL LANGUAGE RULE:
- If the CUSTOMER QUESTION is in Arabic, you MUST write your entire answer in polite, professional Arabic.
- If the CUSTOMER QUESTION is in English, you MUST write your entire answer in English.

=========================================
CONTEXT FROM KNOWLEDGE BASE:
{context}
=========================================
CUSTOMER QUESTION: {user_query}
YOUR ANSWER:"""

    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

# 5. Define the API Endpoints
@app.get("/")
def health_check():
    return {"status": "CityFoam Support API is securely running."}

# Notice the new 'Depends(verify_api_key)' parameter!
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