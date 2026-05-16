# CityFoam RAG Customer Support API Hierarichy:

## Members

| Name | GitHub Account | Contributions |
| :--- | :--- | :--- |
| **Mohamed Abd El-Fattah** | @ | - |
| **Sarah Arafa** | @saraharafa | • Ingestion, Semantic Search & Embeddings<br>• Contextual Retrieval (OpenAI Azure)<br>• Docker<br>• ACR<br>• Web App Service |
| **Ahmed Farahat** | @Baby-Madara | • Deployment on Azure<br>• Docker<br>• ACR<br>• UI/Frontend<br>• Web App Service |
| **Nora** | @ | - |
| **Basel** | @ | - |
| **Mahmoud Osama Ahmed** | @ | - |

cityfoam-rag-api/

│
├── data/                   
├── chroma_db/             
│
├── src/                    
│   ├── __init__.py         
│   ├── config.py           
│   ├── ingest.py          
│   └── api.py             
│
├── .env                    
├── .env.example           
├── .gitignore              
├── requirements.txt       
└── README.md   

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline and API built to handle customer support inquiries for CityFoam. This system ingests mixed-format corporate data (PDFs, Excel, Word), processes it into vector embeddings via ChromaDB, and serves a bilingual (Arabic/English) chatbot powered by Azure OpenAI.

## Architecture
- **Data Ingestion:** `unstructured` for parsing PDFs, Word docs, and Excel files.
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` via `sentence-transformers`.
- **Vector Database:** Persistent `ChromaDB` storage.
- **LLM Engine:** Azure OpenAI (`gpt-4o` or similar).
- **Backend:** `FastAPI` with secure API key authentication.

## Setup & Installation

**1. Clone the repository and navigate to the root directory.**

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

**3. Install dependencies:**
pip install -r requirements.txt

**4. Configure Environment Variables:**
#copy and fill in you secure credentials
cp .env.example .env

##USage

**Step 1: Ingest Data through
python src/ingest.py
**Step 2: Run the API
#If you want to run the FASTAPI Server use:
uvicorn src.api:app --host 0.0.0.0 --port 8000
#To use the Built-in Swagger UI at the localhost/docs or via Postman:
curl -X POST "http://localhost:8000/chat" \
     -H "X-API-Key: your_secure_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{"query": "أين يوجد فرعكم في الإسكندرية؟"}'

***
###Done
