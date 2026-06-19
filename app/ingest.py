"""
ingest.py
---------
Pipeline:
  1. Load raw documents from ./data
  2. Split into chunks (with overlap)
  3. Embed each chunk using HuggingFace sentence-transformers
  4. Store in ChromaDB (persisted to disk)

Run this once (or re-run whenever you add new documents):
  python -m app.ingest
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
)
from app.utils import clean_text, load_documents_from_dir


def run_ingestion() -> Chroma:
    """
    Full ingestion pipeline. Returns the populated Chroma vector store.
    """
    print("=== RAG Ingestion Pipeline ===\n")

    # ── Step 1: Load documents ──────────────────────────────────────────────
    print("Step 1: Loading documents...")
    documents = load_documents_from_dir(DATA_DIR)

    # Clean text content in-place
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # ── Step 2: Chunk documents ─────────────────────────────────────────────
    print(f"\nStep 2: Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on natural boundaries first
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  → {len(documents)} document(s) split into {len(chunks)} chunks")

    # ── Step 3: Embed & store ───────────────────────────────────────────────
    print(f"\nStep 3: Embedding with '{EMBEDDING_MODEL}'...")
    print("  (First run downloads ~90MB model — cached after that)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},       # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"\nStep 4: Storing in ChromaDB at '{CHROMA_PERSIST_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    print(f"\n✅ Ingestion complete. {len(chunks)} chunks stored in vector store.")
    return vectorstore


if __name__ == "__main__":
    run_ingestion()
