import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")          
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")       # Optional fallback

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "openai"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

# --- Embeddings ---
# Using HuggingFace locally — no API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector Store ---
CHROMA_PERSIST_DIR = "./vectorstore"
COLLECTION_NAME = "rag_docs"

# --- Chunking ---
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50      # overlap between chunks to preserve context

# --- Retrieval ---
TOP_K = 4               # how many chunks to retrieve per query

# --- Paths ---
DATA_DIR = "./data"
