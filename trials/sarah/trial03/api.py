import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# 1. Initialize API App
app = FastAPI(title="CityFoam Customer Support API", version="1.0")

# 2. Setup AI and Database Clients (Loads once when server starts)
print("Loading Embedding Model...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Connecting to ChromaDB...")
db_path = os.path.join(os.getcwd(), "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(name="customer_support_knowledge")


AZURE_ENDPOINT = "https://YOUR_RESOURCE_NAME.openai.azure.com/"
AZURE_API_KEY = "YOUR_API_KEY"
DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME"

print("Connecting to Azure OpenAI...")
llm_client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview"
)

# 3. Define the Request Data Structure
class ChatRequest(BaseModel):
    query: str
    language: str = "auto" # Optional flag if the frontend wants to force a language

# 4. The Core Generation Logic (From our Notebook)
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
    return {"status": "CityFoam Support API is running perfectly."}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        answer = get_rag_response(request.query)
        return {
            "status": "success",
            "query": request.query,
            "response": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))