Not a problem! We can easily scrub that out to keep the repository looking as clean and production-ready as possible.

Here is the updated `README.md` with the `debug_camelot.py` file completely removed from the directory structure.

---

```markdown
# CityFoam Interactive RAG Support System

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline and interactive web interface built to handle customer support inquiries for CityFoam. This system ingests mixed-format corporate data (PDFs, Excel, Word), processes it into vector embeddings, and serves a bilingual (Arabic/English) chatbot powered by Azure OpenAI. 

## 👥 Team Members

| Name | GitHub Account | Contributions |
| :--- | :--- | :--- |
| **Mohamed Abd El-Fattah** | @ | - |
| **Sarah Arafa** | @saraharafa | • Ingestion, Semantic Search & Embeddings<br>• Contextual Retrieval (OpenAI Azure)<br>• Docker<br>• ACR<br>• Web App Service |
| **Ahmed Farahat** | @Baby-Madara | • Deployment on Azure<br>• Docker<br>• ACR<br>• UI/Frontend<br>• Web App Service |
| **Nora** | @ | - |
| **Basel** | @ | - |
| **Mahmoud Osama Ahmed** | @ | - |

---

## 📂 Project Structure

```text
cityfoam-rag-api/
│
├── chroma_db/               # Persistent local vector database storage
├── data/                    # Raw corporate files (PDFs, Excel, Word)
├── deliverables/            # Project analysis and evaluation
│   ├── eda_analysis.ipynb   # Exploratory Data Analysis & Semantic Clustering
│   └── evaluation_dataset.csv # Golden dataset for RAG accuracy evaluation
│
├── src/                     # Backend Python Source Code
│   ├── __init__.py         
│   ├── api.py               # FastAPI server and LLM routing
│   ├── config.py            # Environment configurations
│   └── ingest.py            # Data ingestion and vectorization pipeline
│
├── static/                  # Frontend UI Assets
│   ├── favicon.ico
│   ├── index.html           # Bootstrap landing page & chat widget
│   ├── main.js              # API interaction and typing animations
│   └── style.css            # Custom floating UI styling
│
├── .dockerignore            # Docker exclusion rules to keep images lightweight
├── .env                     # Secure environment variables (Not in repo)
├── .env.example             # Template for required environment variables
├── .gitignore              
├── Dockerfile               # Containerization blueprint optimized for PyTorch CPU
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies

```

---

## ⚙️ Architecture & Tech Stack

* **Data Ingestion:** * `pymupdf4llm`: Converts complex, gridless PDF tables into Markdown to preserve row/column relationships for accurate LLM reading.
* `pandas`: Processes spreadsheet data (Branch locations, operating hours).
* `unstructured`: Parses raw Word documents (Refund policies).


* **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` via `sentence-transformers` for robust Arabic/English semantic mapping.
* **Vector Database:** `ChromaDB` for persistent local vector storage.
* **LLM Engine:** Azure OpenAI (`gpt-4o` or similar).
* **Backend:** `FastAPI` providing high-performance REST endpoints.
* **Frontend UI:** Vanilla JavaScript, CSS, and Bootstrap 5 for a responsive, interactive floating chat widget simulating a real e-commerce experience.
* **DevOps & Cloud:** `Docker` for containerization, **Azure Container Registry (ACR)** for image hosting, and **Azure Web App Service** for production deployment.

---

## 🚀 Setup & Local Installation

**1. Clone the repository and navigate to the root directory:**

```bash
git clone [https://github.com/your-repo/cityfoam-rag-api.git](https://github.com/your-repo/cityfoam-rag-api.git)
cd cityfoam-rag-api

```

**2. Create and activate a virtual environment:**

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

```

**3. Install dependencies:**

```bash
pip install -r requirements.txt

```

**4. Configure Environment Variables:**
Copy the template and fill in your secure Azure credentials.

```bash
cp .env.example .env

```

---

## 💻 Local Usage

**Step 1: Ingest the Data**
Run the ingestion script to parse the `data/` folder, convert PDFs to Markdown, and build the ChromaDB vector space.

```bash
python src/ingest.py

```

**Step 2: Run the API Server**
Start the FastAPI server locally.

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

```

**Step 3: Interact with the System**

* **Web UI:** Open your browser and navigate to `http://localhost:8000` to interact with the floating CityFoam chat widget.
* **API Testing:** Use the Built-in Swagger UI at `http://localhost:8000/docs`.

---

## 🐳 Cloud Deployment (Docker & Azure)

This project is fully containerized and deployed to Microsoft Azure for high availability.

**1. Containerization:**
The application uses a multi-stage `Dockerfile` configured to pull the lightweight CPU-only version of PyTorch to drastically reduce the image size and prevent deployment timeouts.

**2. Local Docker Build & Test:**

```bash
docker build -t cityfoam-api .
docker run -p 8000:8000 --env-file .env cityfoam-api

```

**3. Azure Deployment Pipeline:**

* **Azure Container Registry (ACR):** The Docker image is securely tagged and pushed to an ACR repository.
* **Azure App Service Plan:** A Linux-based App Service Plan was provisioned to handle the compute requirements of the RAG pipeline.
* **Azure Web App:** The Web App is configured to continuously pull the latest image from ACR and expose the FastAPI endpoints (`EXPOSE 8000`) to the public web.

---

## 🔍 Key Engineering Highlights

1. **PDF-to-Markdown Pivot:** Standard OCR libraries failed to extract gridless pricing tables accurately. We pivoted to `pymupdf4llm` to convert unstructured PDFs directly into Markdown. This preserved spatial relationships, allowing the LLM to successfully read across rows to find exact prices and dimensions without hallucinating.
2. **Colloquial Arabic Intent Mapping:** Implemented an LLM query optimizer to handle the gap between conversational Egyptian Arabic ("بكام المرتبة دي") and formal database keywords ("السعر", "مرتبة"), dramatically increasing retrieval accuracy.
3. **Vector Space Validation:** Conducted Principal Component Analysis (PCA) on the ChromaDB embeddings to visually verify semantic clustering, ensuring product catalogs, legal policies, and geographic branch locations were strictly isolated in the vector space to prevent context bleed.
4. **Optimized Cloud Deployment:** Successfully navigated deployment limits by configuring the Dockerfile to bypass standard PyTorch installations in favor of optimized CPU wheels, allowing the heavy NLP models to run efficiently within Azure Web App constraints.

```

```