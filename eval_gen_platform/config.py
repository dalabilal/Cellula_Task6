import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/free")  

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL = "small"  


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 60 * 60 * 24     
SEMANTIC_DISTANCE = 0.1       


CHROMA_DIR = "chroma_db"
UPLOAD_DIR = "uploads"


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

MAX_LOOPS = 4                
MAX_FILE_MB = 25

NOT_AVAILABLE = "The required information is not available in the provided sources."
DISCLAIMER = (
    f"after {MAX_LOOPS} attempts. Please double-check it against your sources."
)
