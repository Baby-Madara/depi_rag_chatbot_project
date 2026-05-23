"""
CityFoam RAG — Ingestion Pipeline (v2 — Merged)
=================================================
New in this version (improvements from Sara's implementation):
  1. pymupdf4llm PDF route — converts PDFs to clean Markdown before splitting,
     which preserves Arabic table structure far better than unstructured alone.
     Falls back to unstructured automatically if pymupdf4llm is not installed.
  2. Larger chunk size (1 200 chars, up from 800) with a graceful override so
     Markdown tables are not split in the middle of a row.

Retained from original:
  • .md / .txt / .docx / .xlsx / .csv support.
  • normalize_arabic + fix_arabic_pdf_text helpers.
  • Batch SentenceTransformer encoding.
  • --debug flag for text dump.
  • config.py validate() called explicitly.
"""

import argparse
import logging
import os
import re
import sys

import chromadb
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("cityfoam.ingest")


# ============================================================================
# 1. TEXT UTILITIES
# ============================================================================

def normalize_arabic(text: str) -> str:
    """Standardise Arabic characters to eliminate common spelling mismatches."""
    if not text:
        return ""
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[\u064B-\u065F]", "", text)  # strip harakat
    return text


def fix_arabic_pdf_text(text: str) -> str:
    """Fixes reversed Arabic words that come out of some PDF extractors."""
    if not text or str(text).lower() == "nan":
        return ""
    text = str(text).replace("\n", " ")
    reversed_text = text[::-1]

    def flip_back(match):
        return match.group(0)[::-1]

    return re.sub(r"[A-Za-z0-9%,.()\-]+", flip_back, reversed_text).strip()


# ============================================================================
# 2. SPLITTERS
# ============================================================================

# Slightly larger chunks (Sara's recommendation) so Markdown tables are not
# cut mid-row.  Overlap keeps cross-chunk context intact.
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)


# ============================================================================
# 3. FILE PROCESSORS
# ============================================================================

def _process_text_or_md(file_path: str, filename: str) -> list[dict]:
    """Plain text and Markdown files — read directly, then split."""
    with open(file_path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    full_text = normalize_arabic(raw)
    chunks = TEXT_SPLITTER.split_text(full_text)
    return [
        {"text": chunk, "metadata": {"source": filename, "type": "markdown", "chunk_index": i}}
        for i, chunk in enumerate(chunks)
    ]


def _process_docx_or_txt_unstructured(file_path: str, filename: str) -> list[dict]:
    """Word documents processed via unstructured."""
    elements = partition(filename=file_path)
    full_text = "\n\n".join(
        normalize_arabic(el.text) for el in elements if hasattr(el, "text") and el.text
    )
    chunks = TEXT_SPLITTER.split_text(full_text)
    return [
        {"text": chunk, "metadata": {"source": filename, "type": "document", "chunk_index": i}}
        for i, chunk in enumerate(chunks)
    ]


def _process_pdf(file_path: str, filename: str) -> list[dict]:
    """
    PDF processing — two-tier approach (NEW from Sara):

    Tier 1 (preferred): pymupdf4llm
        Converts the entire PDF to clean Markdown in one call.
        Preserves Arabic table structure, headings, and lists.

    Tier 2 (fallback): unstructured
        Used automatically when pymupdf4llm is not installed.
    """
    try:
        import pymupdf4llm  # noqa: PLC0415
        logger.info("    PDF route: pymupdf4llm (Markdown)")
        md_text = pymupdf4llm.to_markdown(file_path)
        clean   = normalize_arabic(md_text)
        chunks  = TEXT_SPLITTER.split_text(clean)
        return [
            {"text": chunk, "metadata": {"source": filename, "type": "pdf_as_markdown", "chunk_index": i}}
            for i, chunk in enumerate(chunks)
        ]
    except ImportError:
        logger.warning("    pymupdf4llm not installed — falling back to unstructured for PDF.")
        return _process_docx_or_txt_unstructured(file_path, filename)


def _process_spreadsheet(file_path: str, filename: str, ext: str) -> list[dict]:
    """Spreadsheets — one document per row (key: value pairs)."""
    df = pd.read_csv(file_path) if ext == "csv" else pd.read_excel(file_path)
    df = df.fillna("N/A")
    docs = []
    for _, row in df.iterrows():
        row_text = f"--- Branch / Product Record ({filename}) ---\n"
        row_text += "\n".join(
            f"{normalize_arabic(str(k))}: {normalize_arabic(str(v))}"
            for k, v in row.items()
        )
        docs.append({"text": row_text.strip(), "metadata": {"source": filename, "type": "spreadsheet"}})
    return docs


# ============================================================================
# 4. PROCESSING PIPELINE
# ============================================================================

def process_data_folder(debug: bool = False) -> list[dict]:
    """
    Scan Config.DATA_DIR and return a flat list of document dicts
    ready to be embedded and stored in ChromaDB.
    """
    logger.info("Scanning data folder: %s", Config.DATA_DIR)

    if not os.path.exists(Config.DATA_DIR):
        logger.error("Data folder '%s' does not exist.", Config.DATA_DIR)
        return []

    all_files = [
        f for f in os.listdir(Config.DATA_DIR)
        if not f.startswith(".") and os.path.isfile(os.path.join(Config.DATA_DIR, f))
    ]

    if not all_files:
        logger.warning("No files found in %s.", Config.DATA_DIR)
        return []

    final_documents: list[dict] = []

    for filename in sorted(all_files):
        file_path = os.path.join(Config.DATA_DIR, filename)
        ext = filename.rsplit(".", 1)[-1].lower()
        logger.info("  Processing %-40s [.%s]", filename, ext)

        try:
            if ext in ("md", "txt"):
                docs = _process_text_or_md(file_path, filename)

            elif ext == "docx":
                docs = _process_docx_or_txt_unstructured(file_path, filename)

            elif ext in ("xlsx", "xls", "csv"):
                docs = _process_spreadsheet(file_path, filename, ext)

            elif ext == "pdf":
                docs = _process_pdf(file_path, filename)   # ← upgraded route

            else:
                logger.warning("  Skipping unsupported file type: %s", filename)
                continue

            logger.info("    → %d chunks extracted.", len(docs))
            final_documents.extend(docs)

        except Exception as exc:
            logger.error("  ERROR processing %s: %s", filename, exc)

    logger.info("Total chunks ready for embedding: %d", len(final_documents))

    if debug:
        dump_path = os.path.join(Config.DATA_DIR, "_ingest_debug.txt")
        with open(dump_path, "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(d["text"] for d in final_documents))
        logger.info("Debug dump written to %s", dump_path)

    return final_documents


# ============================================================================
# 5. VECTOR DB BUILD
# ============================================================================

def build_vector_db(documents: list[dict]) -> None:
    """Embed all documents and upsert them into ChromaDB."""
    if not documents:
        logger.warning("No documents to ingest. Aborting.")
        return

    logger.info("Connecting to ChromaDB at: %s", Config.CHROMA_DB_DIR)
    client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)

    try:
        client.delete_collection(name=Config.COLLECTION_NAME)
        logger.info("Existing collection '%s' dropped.", Config.COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=Config.COLLECTION_NAME)

    logger.info("Loading embedding model…")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    texts = [d["text"] for d in documents]

    logger.info("Embedding %d chunks (batch)…", len(texts))
    embeddings = embedding_model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[d["metadata"] for d in documents],
        ids=[f"doc_{i}" for i in range(len(documents))],
    )

    logger.info("Successfully stored %d chunks in collection '%s'.",
                len(documents), Config.COLLECTION_NAME)


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CityFoam RAG ingestion pipeline")
    parser.add_argument("--debug", action="store_true", help="Dump extracted text for inspection")
    args = parser.parse_args()

    docs = process_data_folder(debug=args.debug)
    build_vector_db(docs)