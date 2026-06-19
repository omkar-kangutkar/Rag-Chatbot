"""
app.py — Streamlit UI for the RAG chatbot.

Run with:  streamlit run app.py

The app lets users:
  1. Upload PDF/TXT documents
  2. Chat with those documents
  3. See exactly which chunks the answer came from (transparency!)
"""

import os
import sys
import tempfile
import streamlit as st
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion import ingest, load_vector_store, vector_store_exists, CHROMA_DIR
from rag_chain import build_rag_chain, retrieve_chunks, answer_question


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🔍",
    layout="wide",
)

# ── Session state init ─────────────────────────────────────────────────────────
# Streamlit re-runs the whole script on every interaction.
# st.session_state persists values across re-runs.
if "messages" not in st.session_state:
    st.session_state.messages = []          # Chat history
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None     # ChromaDB instance
if "chain" not in st.session_state:
    st.session_state.chain = None           # RAG chain
if "retriever" not in st.session_state:
    st.session_state.retriever = None       # Retriever (for showing sources)
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False    # Whether we have a vector store ready


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Document Setup")
    st.markdown("---")

    # LLM Provider selector
    provider = st.selectbox(
        "LLM Provider",
        ["groq", "openai"],
        help="Groq is free & fast. OpenAI requires a paid key.",
    )
    os.environ["LLM_PROVIDER"] = provider

    # API Key input
    api_key_label = "Groq API Key" if provider == "groq" else "OpenAI API Key"
    api_key = st.text_input(api_key_label, type="password", placeholder="sk-...")
    if api_key:
        env_key = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
        os.environ[env_key] = api_key

    st.markdown("---")
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Your documents are processed locally.",
    )

    # Number of retrieved chunks
    k_chunks = st.slider(
        "Chunks to retrieve (k)",
        min_value=1, max_value=8, value=4,
        help="How many document chunks to use as context per question."
    )

    build_btn = st.button("🔨 Build Knowledge Base", use_container_width=True)

    if build_btn:
        if not uploaded_files:
            st.error("Please upload at least one document first.")
        else:
            # Save uploads to temp dir, run ingestion
            with st.spinner("Processing documents..."):
                try:
                    # Clear existing DB so we start fresh
                    import shutil
                    if Path(CHROMA_DIR).exists():
                        shutil.rmtree(CHROMA_DIR)

                    # Write uploaded files to ./data
                    data_dir = Path("./data")
                    data_dir.mkdir(exist_ok=True)
                    for f in data_dir.iterdir():  # clear old files
                        f.unlink()
                    for uploaded in uploaded_files:
                        dest = data_dir / uploaded.name
                        dest.write_bytes(uploaded.read())

                    # Run the ingestion pipeline
                    vectorstore = ingest("./data")
                    chain, retriever = build_rag_chain(vectorstore, k=k_chunks)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.chain = chain
                    st.session_state.retriever = retriever
                    st.session_state.docs_loaded = True
                    st.session_state.messages = []  # reset chat on new docs

                    st.success(f"✅ Knowledge base built from {len(uploaded_files)} file(s)!")

                except Exception as e:
                    st.error(f"Error during ingestion: {e}")

    # Load existing DB if available and not already loaded
    if not st.session_state.docs_loaded and vector_store_exists():
        with st.spinner("Loading existing knowledge base..."):
            try:
                vectorstore = load_vector_store()
                chain, retriever = build_rag_chain(vectorstore, k=k_chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.chain = chain
                st.session_state.retriever = retriever
                st.session_state.docs_loaded = True
            except Exception:
                pass  # No valid DB yet

    st.markdown("---")

    if st.session_state.docs_loaded:
        st.success("📚 Knowledge base ready")
    else:
        st.info("Upload documents and click **Build Knowledge Base** to start.")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("🔍 RAG Chatbot")
st.markdown(
    "Ask questions about your uploaded documents. "
    "The bot retrieves relevant passages then uses an LLM to answer."
)

# Show pipeline explanation (collapses after first use)
with st.expander("How does this work?", expanded=not st.session_state.docs_loaded):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Ingestion**\nDocuments → chunks → embeddings → ChromaDB")
    with col2:
        st.markdown("**2. Retrieval**\nQuestion → embed → similarity search → top-k chunks")
    with col3:
        st.markdown("**3. Generation**\nPrompt (chunks + question) → LLM → answer")

st.markdown("---")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show sources for assistant messages if stored
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📎 Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    source_name = os.path.basename(src["source"])
                    page = src.get("page", "")
                    page_str = f" • Page {page}" if page != "" else ""
                    st.markdown(f"**Chunk {i}** — `{source_name}`{page_str}")
                    st.text(src["content"][:400] + ("..." if len(src["content"]) > 400 else ""))
                    st.markdown("---")


# Chat input
if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.docs_loaded,
):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Retrieve source chunks first (for display)
                source_docs = retrieve_chunks(st.session_state.retriever, prompt)
                sources = [
                    {
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", ""),
                        "content": doc.page_content,
                    }
                    for doc in source_docs
                ]

                # Generate the answer
                answer = answer_question(st.session_state.chain, prompt)

                st.markdown(answer)

                # Show sources inline
                with st.expander(f"📎 Sources ({len(sources)} chunks retrieved)"):
                    for i, src in enumerate(sources, 1):
                        source_name = os.path.basename(src["source"])
                        page_str = f" • Page {src['page']}" if src["page"] != "" else ""
                        st.markdown(f"**Chunk {i}** — `{source_name}`{page_str}")
                        st.text(src["content"][:400] + ("..." if len(src["content"]) > 400 else ""))
                        st.markdown("---")

                # Persist to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                err_msg = f"Error: {e}"
                if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                    err_msg = "❌ API key error. Please enter a valid key in the sidebar."
                st.error(err_msg)
