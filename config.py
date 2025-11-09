"""Configuration settings for Edalo pipeline."""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
TEST_DATA_DIR = BASE_DIR / "tests" / "test_data"

# Create directories if they don't exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TEST_DATA_DIR.mkdir(exist_ok=True, parents=True)

# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Duplicate detection
FUZZY_MATCH_THRESHOLD = 0.85

# Vector DB settings
VECTOR_DB_PATH = OUTPUT_DIR / "chroma_db"
VECTOR_DB_COLLECTION = "edalo_documents"

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

# Output formats
OUTPUT_JSONL = OUTPUT_DIR / "output.jsonl"
OUTPUT_SQLITE = OUTPUT_DIR / "output.db"
OUTPUT_METRICS = OUTPUT_DIR / "metrics.json"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = OUTPUT_DIR / "edalo.log"

