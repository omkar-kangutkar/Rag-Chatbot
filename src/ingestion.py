"""
ingestion.py — Load documents, chunk them, embed them, store in ChromaDB.

Why chunking?  LLMs have context limits. We split docs into ~500-token chunks
so we can retrieve only the relevant pieces at query time.

Why embeddings?  Embeddings turn text into vectors (lists of numbers) that
capture semantic meaning — "car" and "automobile" end up close together in
vector space, so similarity search finds conceptually related chunks.
"""

import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"           # Where the vector DB is persisted to disk
EMBED_MODEL = "all-MiniLM-L6-v2"    # 384-dim sentence transformer, fast & free
CHUNK_SIZE   = 500                   # tokens per chunk
CHUNK_OVERLAP = 50                   # overlap prevents losing context at boundaries


def load_documents(data_dir: str = "./data") -> List[Document]:
    """Load PDFs and .txt files from a directory. Returns a list of Documents."""
    docs = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    # Load PDFs
    pdf_files = list(data_path.glob("*.pdf"))
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
        print(f"  Loaded PDF: {pdf_path.name} ({len(docs)} pages so far)")

    # Load text files
    txt_files = list(data_path.glob("*.txt"))
    for txt_path in txt_files:
        loader = TextLoader(str(txt_path), encoding="utf-8")
        docs.extend(loader.load())
        print(f"  Loaded TXT: {txt_path.name}")

    if not docs:
        raise ValueError(f"No PDF or .txt files found in '{data_dir}'.")

    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on:
      1. Paragraphs (\n\n)  2. Lines (\n)  3. Spaces  4. Characters
    This keeps chunks semantically coherent where possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks (avg {CHUNK_SIZE} chars each)")
    return chunks


def build_vector_store(chunks: List[Document], persist: bool = True) -> Chroma:
    """
    Embed chunks and store them in ChromaDB.

    HuggingFaceEmbeddings runs locally — no API key needed.
    ChromaDB stores vectors on disk so we only embed once.
    """
    print(f"  Loading embedding model: {EMBED_MODEL} (first run downloads ~90MB)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
    )

    print("  Embedding chunks and writing to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR if persist else None,
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")
    return vectorstore


def load_vector_store() -> Chroma:
    """Load an existing ChromaDB from disk (avoids re-embedding every time)."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def vector_store_exists() -> bool:
    """Check if a ChromaDB has already been built."""
    return Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())


def ingest(data_dir: str = "./data") -> Chroma:
    """Full ingestion pipeline: load → chunk → embed → store. Returns vectorstore."""
    print("\n📂 Loading documents...")
    docs = load_documents(data_dir)
    print(f"\n✂️  Chunking {len(docs)} document pages...")
    chunks = chunk_documents(docs)
    print("\n🔢 Building vector store...")
    vectorstore = build_vector_store(chunks)
    print("\n✅ Ingestion complete!\n")
    return vectorstore


if __name__ == "__main__":
    # Run this script directly to build the vector store from ./data
    ingest()
