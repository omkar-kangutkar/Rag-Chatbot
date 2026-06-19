"""
ui.py
-----
Minimal Streamlit frontend. The real work is in the backend.

Run with:
  streamlit run ui.py
(Make sure main.py / uvicorn is running on port 8000 first)
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.title("RAG Chatbot")
st.caption("Ask questions about your uploaded documents.")

# Sidebar — ingest trigger
with st.sidebar:
    st.header("Setup")
    if st.button("Ingest Documents", use_container_width=True):
        with st.spinner("Ingesting..."):
            res = requests.post(f"{API_URL}/ingest")
            if res.ok:
                st.success(res.json()["message"])
            else:
                st.error(res.json().get("detail", "Ingestion failed"))

    st.divider()
    health = requests.get(f"{API_URL}/health").json()
    st.metric("Chunks in store", health["chunk_count"])
    st.metric("Store ready", "✅" if health["vector_store_ready"] else "❌")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = requests.post(f"{API_URL}/chat", json={"question": prompt})

        if res.ok:
            data = res.json()
            st.write(data["answer"])

            if data["sources"]:
                with st.expander(f"Sources ({len(data['sources'])})"):
                    for s in data["sources"]:
                        st.markdown(f"**{s['source']}**")
                        st.caption(s["excerpt"])

            st.caption(f"⏱ {data['latency_ms']}ms")
            st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
        else:
            err = res.json().get("detail", "Something went wrong")
            st.error(err)
