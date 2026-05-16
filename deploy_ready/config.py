"""
CityFoam — Application Configuration
======================================
Reads all settings from the .env file.

Change vs. original:
  • Config.validate() is NO LONGER called automatically at import time.
    This prevented Ollama-only setups (no Azure keys) from starting.
    Call Config.validate() explicitly inside server.py when LLM_PROVIDER == "azure".
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Security ────────────────────────────────────────────────────────────
    CITYFOAM_SECRET_KEY = os.getenv("CITYFOAM_SECRET_KEY", "change_me_in_production")

    # ── Azure OpenAI ────────────────────────────────────────────────────────
    AZURE_ENDPOINT        = os.getenv("AZURE_ENDPOINT")
    AZURE_API_KEY         = os.getenv("AZURE_API_KEY")
    AZURE_API_VERSION     = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")

    # ── Ollama ───────────────────────────────────────────────────────────────
    OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ── Paths & ChromaDB ─────────────────────────────────────────────────────
    DATA_DIR        = os.getenv("DATA_DIR", "./data")
    CHROMA_DB_DIR   = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cityfoam_rag")

    # ── Frontend ─────────────────────────────────────────────────────────────
    COMPANY_NAME   = os.getenv("COMPANY_NAME", "CityFoam")
    THEME          = os.getenv("THEME", "dark")
    ENABLE_SIDEBAR = os.getenv("ENABLE_SIDEBAR", "true").lower() == "true"
    PORT           = int(os.getenv("PORT", 8000))

    @classmethod
    def validate_azure(cls) -> None:
        """Call this explicitly when LLM_PROVIDER is 'azure'."""
        if not cls.AZURE_API_KEY or not cls.AZURE_ENDPOINT:
            raise ValueError(
                "CRITICAL: AZURE_API_KEY and AZURE_ENDPOINT must both be set in .env "
                "when LLM_PROVIDER=azure."
            )