import os
import pandas as pd
import logging
import re
import chromadb
import pymupdf4llm  # <--- NEW: The ultimate PDF-to-Markdown tool
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import Config

# Configure logging to see progress in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. TEXT UTILITIES
# ==========================================
def normalize_arabic(text):
    """Standardizes Arabic characters to eliminate common spelling mismatches."""
    if not text: return ""
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    return text

# ==========================================
# 2. PROCESSING PIPELINE
# ==========================================
def process_data_folder():
    """Scans the data folder and extracts content based on file type."""
    logger.info(f"Scanning folder: {Config.DATA_DIR}")
    final_documents = []
    
    # We increase chunk size slightly for Markdown tables so they don't get cut in half
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

    if not os.path.exists(Config.DATA_DIR):
        logger.error(f"Data folder '{Config.DATA_DIR}' does not exist.")
        return []

    for filename in os.listdir(Config.DATA_DIR):
        file_path = os.path.join(Config.DATA_DIR, filename)
        if os.path.isdir(file_path) or filename.startswith('.'): continue
            
        ext = filename.split('.')[-1].lower()
        logger.info(f"Processing {filename}...")

        # --- ROUTE 1: Spreadsheets (Pandas) ---
        if ext in ['xlsx', 'xls', 'csv']:
            try:
                df = pd.read_csv(file_path) if ext == 'csv' else pd.read_excel(file_path)
                for _, row in df.fillna("N/A").iterrows():
                    row_text = f"--- Database Record ({filename}) ---\n"
                    row_text += "\n".join([f"{normalize_arabic(str(k))}: {normalize_arabic(str(v))}" for k, v in row.items()])
                    final_documents.append({"text": row_text.strip(), "metadata": {"source": filename, "type": "spreadsheet"}})
            except Exception as e:
                logger.error(f"Error processing spreadsheet {filename}: {e}")

        # --- ROUTE 2: PDFs (Automated Markdown Conversion) ---
        elif ext == 'pdf':
            try:
                # 1. Convert the entire PDF into a clean Markdown string instantly
                md_text = pymupdf4llm.to_markdown(file_path)
                
                # 2. Clean the Arabic text
                clean_text = normalize_arabic(md_text)
                
                # 3. Split it into chunks
                chunks = text_splitter.split_text(clean_text)
                for i, chunk in enumerate(chunks):
                    final_documents.append({"text": chunk, "metadata": {"source": filename, "type": "pdf_as_markdown", "chunk_index": i}})
            except Exception as e:
                logger.error(f"Error processing PDF {filename}: {e}")

        # --- ROUTE 3: Word Docs (Unstructured) ---
        elif ext in ['docx', 'txt']:
            try:
                elements = partition(filename=file_path)
                full_text = "\n\n".join([normalize_arabic(el.text) for el in elements if hasattr(el, 'text')])
                chunks = text_splitter.split_text(full_text)
                for i, chunk in enumerate(chunks):
                    final_documents.append({"text": chunk, "metadata": {"source": filename, "type": "document", "chunk_index": i}})
            except Exception as e:
                logger.error(f"Error processing document {filename}: {e}")

    return final_documents

def build_vector_db(documents):
    """Embeds and saves documents to ChromaDB."""
    if not documents:
        logger.warning("No documents to ingest.")
        return

    logger.info("Connecting to ChromaDB and embedding documents...")
    client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)
    
    try:
        client.delete_collection(name=Config.COLLECTION_NAME)
    except Exception: pass
    
    collection = client.create_collection(name=Config.COLLECTION_NAME)
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    texts = [doc["text"] for doc in documents]
    embeddings = embedding_model.encode(texts).tolist()
    
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[doc["metadata"] for doc in documents],
        ids=[f"id_{i}" for i in range(len(documents))]
    )
    logger.info(f"Successfully stored {len(documents)} chunks in {Config.COLLECTION_NAME}.")


def main():
    """الدالة الرئيسية المتاحة للاستدعاء من الـ API لإعادة التدريب الفوري"""
    logger.info("Starting knowledge base ingestion pipeline...")
    docs = process_data_folder()
    build_vector_db(docs)
    logger.info("Ingestion pipeline finished successfully!")

if __name__ == "__main__":
    # هذا السطر يضمن تشغيل الملف يدوياً بشكل طبيعي من التيرمنال
    main()