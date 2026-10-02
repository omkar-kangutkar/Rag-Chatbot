"""Streamlit interface using the shared RAG backend."""

import tempfile
from pathlib import Path

import streamlit as st

from app.config import TOP_K
from app.retriever import get_vectorstore
from app.service import ask, run_ingestion


st.set_page_config(page_title="RAG Chatbot", page_icon="📄")

st.title("Document Question Answering")
st.caption("Upload documents and ask questions using the shared RAG backend.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ready" not in st.session_state:
    st.session_state.ready = False


def display_sources(sources):
    if sources:
        with st.expander("Retrieved sources"):
            for source in sources:
                st.markdown(f"**{Path(source['source']).name}**")
                st.write(source["excerpt"])


with st.sidebar:
    st.header("Documents")
    st.caption(
        "Embeddings are computed locally. Retrieved document text "
        "is sent to the configured LLM provider when you ask a question."
    )

    uploads = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    top_k = st.slider(
        "Maximum chunks to retrieve",
        min_value=1,
        max_value=10,
        value=TOP_K,
    )

    st.caption(
        "Adding documents keeps previously indexed documents "
        "in the shared knowledge base."
    )

    if st.button("Add documents"):
        if not uploads:
            st.warning("Choose at least one document first.")
        else:
            with st.spinner("Indexing documents..."):
                try:
                    with tempfile.TemporaryDirectory() as folder:
                        for index, upload in enumerate(uploads):
                            filename = f"{index}_{Path(upload.name).name}"
                            destination = Path(folder) / filename
                            destination.write_bytes(upload.getvalue())

                        run_ingestion(data_dir=folder)

                    st.session_state.ready = True
                    st.session_state.messages = []
                    st.success("Documents indexed.")
                except Exception:
                    st.error(
                        "Indexing failed. Check the terminal for details."
                    )
                    import logging
                    logging.exception("Document ingestion failed")

    if st.button("Load existing knowledge base"):
        try:
            get_vectorstore()
            st.session_state.ready = True
            st.success("Knowledge base loaded.")
        except Exception:
            st.session_state.ready = False
            st.warning("Could not load the store. Try adding documents.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("LLM credentials and model settings are read from .env.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        display_sources(message.get("sources", []))


question = st.chat_input(
    "Ask about your documents",
    disabled=not st.session_state.ready,
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating an answer..."):
            try:
                result = ask(question, top_k=top_k)
                st.write(result["answer"])
                display_sources(result["sources"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )
            except Exception:
                st.error(
                    "The request failed. Check your model configuration "
                    "and the terminal output."
                )
                import logging
                logging.exception("Question answering failed")
                