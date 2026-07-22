import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Google Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please add it to your .env file."
    )

# Project folders
DOCUMENTS_PATH = "documents"
UPLOADS_PATH = "uploads"
VECTOR_DB_PATH = "vector_store"