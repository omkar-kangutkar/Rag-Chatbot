"""
retriever.py
------------
Loads the persisted ChromaDB vector store and exposes a retriever
that returns the top-k most relevant chunks for a given query.

This is kept separate from the chain so you can:
  - Swap vector stores easily (FAISS, Pinecone, etc.)
  - Test retrieval independently from LLM generation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import Chroma

from app.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)

# Module-level cache so we don't reload on every request
_vectorstore: Chroma | None = None
def reset_vectorstore_cache() -> None:
    """Reload the vector store on the next retrieval."""
    global _vectorstore
    _vectorstore = None

def get_vectorstore() -> Chroma:
    """Return a cached store, or load and validate it before caching."""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    count = vectorstore._collection.count()
    if count == 0:
        raise RuntimeError(
            "Vector store is empty. Run ingestion first:\n"
            "  python -m app.ingest"
        )

    _vectorstore = vectorstore
    print(f"[retriever] Loaded vector store with {count} chunks.")
    return _vectorstore

def retrieve(query: str, top_k: int = TOP_K) -> List[Document]:
    """
    Retrieve the top-k most relevant document chunks for a query.

    Args:
        query:  The user's question string.
        top_k:  Number of chunks to return (default from config).

    Returns:
        List of LangChain Document objects with page_content + metadata.
    """
    vs = get_vectorstore()

    # similarity_search_with_score returns (Document, score) tuples
    results_with_scores = vs.similarity_search_with_score(query, k=top_k)

    # Filter out very low-relevance chunks (cosine distance > 1.5)
    filtered = [
        doc for doc, score in results_with_scores if score < 1.5
    ]

    return filtered


def retrieve_with_scores(query: str, top_k: int = TOP_K):
    """
    Same as retrieve() but also returns relevance scores.
    Useful for debugging and building confidence-aware responses.
    """
    vs = get_vectorstore()
    return vs.similarity_search_with_score(query, k=top_k)
