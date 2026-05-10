# CityFoam RAG Customer Support API Hierarichy:

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
source venv/bin/activate  # On Windows: venv\Scripts\activate