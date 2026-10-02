# RAG Chatbot

A production-style Retrieval-Augmented Generation (RAG) chatbot. Ask questions over your own documents using embeddings, vector search, and an LLM.

## Architecture

```
Your Documents (.txt / .pdf / .md)
        │
        ▼
   [ingest.py]
   Text Chunking (RecursiveTextSplitter)
        │
        ▼
   HuggingFace Embeddings (all-MiniLM-L6-v2)
        │
        ▼
   ChromaDB Vector Store (persisted to disk)
        │
   User Query ──► Embed Query ──► Similarity Search ──► Top-K Chunks
                                                              │
                                                              ▼
                                                    RAG Prompt Template
                                                              │
                                                              ▼
                                                    Groq LLM (Llama 3)
                                                              │
                                                              ▼
                                                    Answer + Sources
```

## Stack

| Component | Tool |
|---|---|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, free) |
| Vector Store | ChromaDB (persisted to disk) |
| LLM | Groq API — Llama 3 8B (free, fast) |
| RAG Framework | LangChain |
| API | FastAPI |
| UI | Streamlit |

## Setup

### 1. Clone and install

```bash
git clone https://github.com/omkar-kangutkar/Rag-Chatbot.git
cd Rag-Chatbot
pip install -r requirements.txt
```

### 2. Get a free Groq API key

Go to [console.groq.com](https://console.groq.com), sign up, create an API key.

### 3. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Add your documents

Drop `.txt`, `.pdf`, or `.md` files into the `./data/` folder.
A sample ML document is already included to test with.

## Running

### Step 1 — Start the API

```bash
uvicorn main:app --reload --port 8000
```

### Step 2 — Ingest your documents

```bash
# Option A: via curl
curl -X POST http://localhost:8000/ingest

# Option B: via the Streamlit UI sidebar button
```

This runs once. Re-run only when you add new documents.

### Step 3 — Ask questions

```bash
# via curl
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the types of machine learning?", "top_k": 4}'

# Response:
# {
#   "question": "What are the types of machine learning?",
#   "answer": "Based on the document, there are three types...",
#   "sources": [{"source": "sample_ml_doc.txt", "excerpt": "..."}],
#   "latency_ms": 843.2
# }
```

### Step 4 — (Optional) Run the UI

```bash
streamlit run ui.py
```

Open [http://localhost:8501](http://localhost:8501)

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check API + vector store status |
| `/ingest` | POST | Trigger document ingestion pipeline |
| `/chat` | POST | Ask a question, get answer + sources |
| `/docs` | GET | Interactive Swagger UI |

## Key Design Decisions

### Retrieval settings

`POST /chat` accepts an optional `top_k` integer from 1 to 10. Omit it to use
the configured default of 4. The value controls the maximum number of chunks
requested from vector search; relevance filtering may return fewer chunks.
Explicit `null`, out-of-range values and fractional values are rejected with
HTTP 422. If no relevant chunks are returned, the API returns an insufficient
evidence response without calling the LLM.

### Automated tests

Run the API and retrieval tests without API keys, embedding downloads or an
existing vector database:

```bash
python -m venv .venv-test
source .venv-test/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The tests exercise the HTTP endpoint through answer orchestration and vector
search with a fake store. They also test LangChain prompt/output handling with
a fake model. They verify parameter forwarding, validation, relevance filtering,
empty results and an unready store. They do not measure real retrieval quality,
live provider compatibility or model answer correctness. GitHub Actions runs
this suite on pushes and pull requests.

**Why local embeddings?** `all-MiniLM-L6-v2` runs on CPU, no API key needed, and is fast enough for most use cases. Swap to OpenAI embeddings in `config.py` for better quality at a cost.

**Why Groq?** Free tier, very fast (low latency), runs Llama 3 which is competitive with GPT-3.5. Great for demos.

**Why ChromaDB?** Simple setup, persists to disk, no server needed. For production scale, swap to Pinecone or Weaviate by changing the retriever.

**Chunk size 500 / overlap 50:** Works well for most documents. Increase chunk size for technical docs, decrease for dense factual content.

## Resume Bullet

> Built a RAG chatbot using HuggingFace embeddings, ChromaDB vector search, and LLM prompting via LangChain and Groq; deployed as a REST API with FastAPI and a Streamlit UI.
