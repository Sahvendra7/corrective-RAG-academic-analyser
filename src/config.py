import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables once centrally
load_dotenv()

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEXT_DIR = PROCESSED_DIR / "texts"
CHUNK_DIR = PROCESSED_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
META_FILE = PROCESSED_DIR / "metadata.json"

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.1-flash-lite"
TOP_K = 3
MAX_RETRIES = 2
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
MIN_CHUNK_SIZE = 50

def setup_dirs():
    """Ensure data directories exist. Must be called explicitly by entry points."""
    for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, TEXT_DIR, CHUNK_DIR, EMBEDDINGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
