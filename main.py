"""
main.py
-------
FastAPI backend exposing the RAG chatbot as a REST API.

Endpoints:
  POST /ingest       - Trigger document ingestion (build vector store)
  POST /chat         - Ask a question, get an answer + sources
  GET  /health       - Health check
  GET  /docs         - Auto-generated Swagger UI (FastAPI built-in)

Run with:
  uvicorn main:app --reload --port 8000
"""

import os
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import CHROMA_PERSIST_DIR, TOP_K
from app.service import ask, run_ingestion

# ── Lifespan: pre-load vector store on startup ───────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the retriever on startup if vector store exists."""
    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            from app.retriever import get_vectorstore
            get_vectorstore()
            print("[startup] Vector store loaded and ready.")
        except Exception as e:
            print(f"[startup] Warning: Could not load vector store: {e}")
            print("[startup] Run POST /ingest to build it.")
    else:
        print("[startup] No vector store found. Run POST /ingest first.")
    yield


# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot. Ingest documents, then ask questions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, example="What is this document about?")
    top_k: int = Field(default=TOP_K, ge=1, le=10, description="Maximum number of chunks to retrieve before relevance filtering")

class SourceItem(BaseModel):
    source: str
    excerpt: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    latency_ms: float

class IngestResponse(BaseModel):
    status: str
    message: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    chunk_count: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API and vector store are ready."""
    try:
        from app.retriever import get_vectorstore
        vs = get_vectorstore()
        count = vs._collection.count()
        return HealthResponse(status="ok", vector_store_ready=True, chunk_count=count)
    except Exception:
        return HealthResponse(status="ok", vector_store_ready=False, chunk_count=0)


@app.post("/ingest", response_model=IngestResponse, tags=["Setup"])
def ingest_documents():
    """
    Trigger the ingestion pipeline.
    
    - Loads all files from ./data directory
    - Chunks, embeds, and stores in ChromaDB
    - Must be run before /chat will work
    
    Supported file types: .txt, .pdf, .md
    """
    start = time.time()
    try:
        run_ingestion()
        elapsed = round((time.time() - start) * 1000, 2)
        return IngestResponse(
            status="success",
            message="Documents ingested and vector store built successfully.",
            latency_ms=elapsed,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Ask a question about your ingested documents.
    
    Flow:
    1. Embeds your question
    2. Retrieves top-k relevant chunks from ChromaDB
    3. Sends chunks + question to LLM
    4. Returns answer + source excerpts
    """
    start = time.time()
    try:
        result = ask(request.question, top_k=request.top_k)
        elapsed = round((time.time() - start) * 1000, 2)

        return ChatResponse(
            question=request.question,
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
            latency_ms=elapsed,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
