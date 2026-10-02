"""Load documents, create stable chunks and store their embeddings."""

import hashlib
import json

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


def chunk_id(chunk) -> str:
    """Identify a chunk independently of its filename or upload order."""
    identity = {
        "document_id": chunk.metadata["document_id"],
        "page": chunk.metadata.get("page"),
        "start_index": chunk.metadata["start_index"],
        "text": chunk.page_content,
    }
    encoded = json.dumps(identity, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_ingestion(data_dir: str = DATA_DIR) -> Chroma:
    """Index documents using stable IDs for repeated identical uploads."""
    documents = load_documents_from_dir(data_dir)

    for document in documents:
        document.page_content = clean_text(document.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No readable text found in the supplied documents.")

    # Remove identical chunks within this ingestion request.
    unique_chunks = {}
    for chunk in chunks:
        unique_chunks.setdefault(chunk_id(chunk), chunk)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=list(unique_chunks.values()),
        ids=list(unique_chunks),
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    print(f"Ingestion complete: {len(unique_chunks)} unique chunks processed.")
    return vectorstore


if __name__ == "__main__":
    run_ingestion()