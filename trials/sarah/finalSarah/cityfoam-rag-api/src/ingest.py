import os
import pandas as pd
import io
import logging
import re
import camelot
import chromadb
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

def fix_arabic_pdf_text(text):
    """Fixes backwards Arabic text from PDFs while preserving English and Numbers."""
    if not text or str(text).lower() == 'nan':
        return ""
    text = str(text).replace('\n', ' ')
    reversed_text = text[::-1]
    
    def flip_back(match):
        return match.group(0)[::-1]
        
    fixed_text = re.sub(r'[A-Za-z0-9\%\\,\\.\\(\\)\-]+', flip_back, reversed_text)
    return fixed_text.strip()

# ==========================================
# 2. PROCESSING PIPELINE
# ==========================================
def process_data_folder():
    """Scans the data folder and extracts content based on file type."""
    logger.info(f"Scanning folder: {Config.DATA_DIR}")
    final_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    if not os.path.exists(Config.DATA_DIR):
        logger.error(f"Data folder '{Config.DATA_DIR}' does not exist.")
        return []

    for filename in os.listdir(Config.DATA_DIR):
        file_path = os.path.join(Config.DATA_DIR, filename)
        if os.path.isdir(file_path) or filename.startswith('.'): continue
            
        ext = filename.split('.')[-1].lower()
        logger.info(f"Processing {filename}...")

        # --- ROUTE 1: PDFs (Camelot Engine for Tables) ---
        if ext == 'pdf':
            try:
                tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
                for table in tables:
                    df = table.df
                    if df.empty: continue
                    headers = df.iloc[0]
                    for _, row in df[1:].iterrows():
                        chunk_text = f"--- Catalog Entry ({filename}) ---\n"
                        for col_name, cell_value in zip(headers, row):
                            clean_col = normalize_arabic(fix_arabic_pdf_text(col_name))
                            clean_cell = normalize_arabic(fix_arabic_pdf_text(cell_value))
                            chunk_text += f"{clean_col}: {clean_cell}\n"
                        final_documents.append({"text": chunk_text.strip(), "metadata": {"source": filename, "type": "pdf_table"}})
            except Exception as e:
                logger.error(f"Error processing PDF {filename}: {e}")

        # --- ROUTE 2: Spreadsheets (Pandas) ---
        elif ext in ['xlsx', 'xls', 'csv']:
            try:
                df = pd.read_csv(file_path) if ext == 'csv' else pd.read_excel(file_path)
                for _, row in df.fillna("N/A").iterrows():
                    row_text = f"--- Database Record ({filename}) ---\n"
                    row_text += "\n".join([f"{normalize_arabic(str(k))}: {normalize_arabic(str(v))}" for k, v in row.items()])
                    final_documents.append({"text": row_text.strip(), "metadata": {"source": filename, "type": "spreadsheet"}})
            except Exception as e:
                logger.error(f"Error processing spreadsheet {filename}: {e}")

        # --- ROUTE 3: Word/Text (Recursive Splitting) ---
        elif ext in ['docx', 'txt']:
            try:
                elements = partition(filename=file_path)
                full_text = "\n\n".join([normalize_arabic(el.text) for el in elements if hasattr(el, 'text')])
                chunks = text_splitter.split_text(full_text)
                for i, chunk in enumerate(chunks):
                    final_documents.append({"text": chunk, "metadata": {"source": filename, "type": "text", "chunk_index": i}})
            except Exception as e:
                logger.error(f"Error processing document {filename}: {e}")


    all_text_content= "\n\n".join([doc["text"] for doc in final_documents])

    with open('my_file.txt', 'w', encoding='utf-8') as file:
        file.write(all_text_content)
        print(f"Done! Saved {len(final_documents)} chunks to my_file.txt")
    return final_documents
def build_vector_db(documents):
    """Embeds and saves documents to ChromaDB."""
    if not documents:
        logger.warning("No documents to ingest.")
        return

    logger.info("Connecting to ChromaDB and embedding documents...")
    client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)
    
    # Reset collection for a clean build
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

if __name__ == "__main__":
    docs = process_data_folder()
    build_vector_db(docs)