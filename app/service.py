"""Shared operations for the API and Streamlit interface."""

from app import chain
from app.config import DATA_DIR, TOP_K
from app.retriever import reset_vectorstore_cache


def ask(question: str, top_k: int = TOP_K):
    """Retrieve relevant documents and generate an answer."""
    return chain.ask(question, top_k=top_k)


def run_ingestion(data_dir: str = DATA_DIR):
    """Index documents and refresh the retrieval cache."""
    from app.ingest import run_ingestion as ingest

    vectorstore = ingest(data_dir=data_dir)
    reset_vectorstore_cache()
    return vectorstore