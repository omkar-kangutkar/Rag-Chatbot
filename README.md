# RAG Chatbot

A document question-answering project using local embeddings, ChromaDB vector search and an LLM through Groq. FastAPI and Streamlit share the same backend operations.

## Architecture

FastAPI (`main.py`) and Streamlit (`app.py`) call the shared
operations in `app/service.py`.

- **Ingestion:** `app/ingest.py` loads documents, splits them into
  chunks, computes local embeddings and stores them in ChromaDB.
  After successful ingestion, the service resets the retrieval cache.
- **Retrieval:** `app/retriever.py` searches the persisted vector
  store for relevant chunks using the requested `top_k`.
- **Answer generation:** `app/chain.py` builds a prompt from the
  retrieved chunks, calls the configured LLM and returns an answer
  with source excerpts.

The main Streamlit interface accepts PDF and TXT uploads.
The API indexes documents from the configured `data/` folder.

## Stack

| Component | Tool |
|---|---|
| Language | Python; local setup tested with Python 3.11 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, computed locally |
| Vector store | ChromaDB, persisted locally |
| LLM provider | Groq API; model configured through `.env` |
| Tested model | `openai/gpt-oss-20b` |
| RAG framework | LangChain |
| API | FastAPI |
| Main interface | Streamlit (`app.py`) |
| Automated testing | pytest |
| Continuous integration | GitHub Actions |

## Setup

The instructions below use Windows Command Prompt and Python 3.11.

### 1. Clone the repository

```cmd
git clone https://github.com/omkar-kangutkar/Rag-Chatbot.git
cd Rag-Chatbot
```

If you already have the repository, open a terminal in its folder.

### 2. Create an environment and install dependencies

```cmd
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
```

The project pins HTTPX to 0.27.2 for compatibility with the Groq SDK.

### 3. Configure the model

Create a file named `.env` in the project root, beside `main.py`.
If it already exists, update it rather than overwriting it.

```dotenv
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=your_actual_groq_api_key
```

Create an API key in the [Groq Console](https://console.groq.com/).
Replace the placeholder with your own key. Never commit `.env`.

## Run the Streamlit application

```cmd
.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501.

1. Upload a PDF or TXT file.
2. Click **Add documents** and wait for indexing.
3. Ask a question about the document.
4. Expand **Retrieved sources** to inspect the supporting excerpts.

The first ingestion may download the embedding model.
Adding documents appends to the existing knowledge base.
After restarting, click **Load existing knowledge base** to reuse it.

Embeddings run locally. Retrieved document text is sent to the
configured LLM provider when generating an answer.

The generated `vectorstore/` directory is ignored by Git.

## Run the FastAPI application

FastAPI and the main Streamlit application use the shared operations
in `app/service.py`. Streamlit's `app.py` does not require the API
server to be running.

To start the API:

```cmd
.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs to try the endpoints:

1. Put documents in the `data/` folder.
2. Call `POST /ingest` to index them.
3. Call `POST /chat` with a request such as:

```json
{
  "question": "What are the types of machine learning?",
  "top_k": 4
}
```

The response includes the answer, retrieved source excerpts and
request latency.

The alternative `ui.py` interface connects to the API and requires
the API server to be running.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check API and vector-store status |
| `/ingest` | POST | Index documents from the configured data folder |
| `/chat` | POST | Return an answer, source excerpts and latency |
| `/docs` | GET | Interactive API documentation |

## Retrieval Settings

`POST /chat` accepts an optional `top_k` integer from 1 to 10.
Omitting it uses the configured default of 4.

The setting controls the maximum number of chunks requested.
Relevance filtering may return fewer chunks. Explicit `null`,
fractional values and out-of-range values receive HTTP 422.

If retrieval returns no relevant chunks, the application returns
an insufficient-evidence response without calling the LLM.

## Automated Tests

Run these commands in Windows Command Prompt:

```cmd
py -3.11 -m venv .venv-test
.venv-test\Scripts\python.exe -m pip install -r requirements-test.txt
.venv-test\Scripts\python.exe -m pytest -q
```

The suite runs without API keys, embedding-model downloads or an
existing vector database.

Tests cover API validation, retrieval parameter forwarding,
relevance filtering, empty results, prompt/output handling,
shared-service forwarding and cache reset after ingestion.

GitHub Actions runs the suite on pushes and pull requests.
These tests use fake stores and models; they do not measure live
retrieval quality or provider compatibility.

## Manual Verification

The following checks passed locally:

- A question about an uploaded TXT document returned the expected
  answer and displayed a retrieved source.
- A question asking for a founding year absent from the test
  document returned an insufficient-information response.

These checks demonstrate the tested examples, not a guarantee
that every generated answer will be correct.