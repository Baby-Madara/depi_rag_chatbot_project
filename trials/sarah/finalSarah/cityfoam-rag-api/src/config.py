import os
import json
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

class Config:
    # Security
    CITYFOAM_SECRET_KEY = os.getenv("CITYFOAM_SECRET_KEY", "default_dev_key")
    
    # Azure OpenAI
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "cityfoam-gpt")
    
    # Paths & DB Configurations
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cityfoam_rag")

    # ==========================================
    # PSO-Optimized Parameters
    # (defaults used if no pso_best_params.json exists)
    # ==========================================
    PSO_PARAMS_FILE = os.getenv("PSO_PARAMS_FILE", "./pso_best_params.json")

    # Retrieval hyperparameters
    TOP_K: int                   = int(os.getenv("PSO_TOP_K", 5))
    SIMILARITY_THRESHOLD: float  = float(os.getenv("PSO_SIMILARITY_THRESHOLD", 0.0))
    CHUNK_WEIGHT: float          = float(os.getenv("PSO_CHUNK_WEIGHT", 1.0))

    # Ranking weights (loaded from PSO output, must sum to 1.0)
    WEIGHT_SEMANTIC: float       = float(os.getenv("PSO_WEIGHT_SEMANTIC",  0.7))
    WEIGHT_KEYWORD: float        = float(os.getenv("PSO_WEIGHT_KEYWORD",   0.1))
    WEIGHT_RECENCY: float        = float(os.getenv("PSO_WEIGHT_RECENCY",   0.1))
    WEIGHT_AUTHORITY: float      = float(os.getenv("PSO_WEIGHT_AUTHORITY", 0.1))

    @classmethod
    def load_pso_params(cls, verbose: bool = True):
        """
        Loads optimized PSO parameters from pso_best_params.json dynamically.
        By keeping this dynamic, the API will instantly absorb new weights updated
        by the background scheduler without requiring a server restart.
        """
        if os.path.exists(cls.PSO_PARAMS_FILE):
            try:
                with open(cls.PSO_PARAMS_FILE, "r", encoding="utf-8") as f:
                    params = json.load(f)

                cls.TOP_K                = int(params.get("top_k",                cls.TOP_K))
                cls.SIMILARITY_THRESHOLD = float(params.get("similarity_threshold", cls.SIMILARITY_THRESHOLD))
                cls.CHUNK_WEIGHT         = float(params.get("chunk_weight",         cls.CHUNK_WEIGHT))
                cls.WEIGHT_SEMANTIC      = float(params.get("weight_semantic",      cls.WEIGHT_SEMANTIC))
                cls.WEIGHT_KEYWORD       = float(params.get("weight_keyword",       cls.WEIGHT_KEYWORD))
                cls.WEIGHT_RECENCY       = float(params.get("weight_recency",       cls.WEIGHT_RECENCY))
                cls.WEIGHT_AUTHORITY     = float(params.get("weight_authority",     cls.WEIGHT_AUTHORITY))

                if verbose:
                    print(f"[Config] ✓ Live PSO params synced from {cls.PSO_PARAMS_FILE}")
                    print(f"         top_k={cls.TOP_K} | threshold={cls.SIMILARITY_THRESHOLD}")
                    print(f"         weights → semantic={cls.WEIGHT_SEMANTIC} | keyword={cls.WEIGHT_KEYWORD} | "
                          f"recency={cls.WEIGHT_RECENCY} | authority={cls.WEIGHT_AUTHORITY}")
            except Exception as e:
                # Fallback silently to current RAM values if file is being written to prevent edge-case IO crashes
                if verbose:
                    print(f"[Config] ⚠ Temporary issue reading {cls.PSO_PARAMS_FILE}: {e}. Keeping current memory values.")
        else:
            if verbose:
                print(f"[Config] ⚠ No PSO params file found at '{cls.PSO_PARAMS_FILE}' — using defaults.")

    @classmethod
    def validate(cls):
        """Checks if all critical variables are present before the app starts."""
        if not cls.AZURE_API_KEY or not cls.AZURE_ENDPOINT:
            raise ValueError("CRITICAL: Azure API Key or Endpoint is missing from .env!")

# Run the validation immediately when this file is imported
Config.validate()