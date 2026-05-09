import os
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
    
    @classmethod
    def validate(cls):
        """Checks if all critical variables are present before the app starts."""
        if not cls.AZURE_API_KEY or not cls.AZURE_ENDPOINT:
            raise ValueError("CRITICAL: Azure API Key or Endpoint is missing from .env!")

# Run the validation immediately when this file is imported
Config.validate()