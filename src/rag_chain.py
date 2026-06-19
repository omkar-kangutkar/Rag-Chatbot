"""
rag_chain.py — The retrieval-augmented generation pipeline.

The RAG flow:
  1. User asks a question
  2. We embed the question using the same model we used at ingestion
  3. ChromaDB finds the top-k most similar chunks (vector similarity search)
  4. We inject those chunks into a prompt as "context"
  5. The LLM answers based only on that context (grounded, not hallucinated)

This is the core pattern you'll explain in every ML interview.
"""

import os
from typing import List, Tuple, Optional
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()


# ── Prompt template ──────────────────────────────────────────────────────────
# This is the key to a good RAG system. The prompt:
#   - Grounds the LLM in the retrieved context
#   - Prevents hallucination by saying "only use the context below"
#   - Handles the "I don't know" case gracefully
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on provided documents.

Use ONLY the context below to answer the question.
If the answer isn't in the context, say "I don't have enough information in the uploaded documents to answer that."
Do not make up information.

Context:
{context}

Question: {question}

Answer:""")


def get_llm(provider: Optional[str] = None):
    """
    Return an LLM client. Supports Groq (free) or OpenAI (paid).
    Groq is recommended for beginners — free tier, fast inference.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "groq")

    if provider == "groq":
        from langchain_groq import ChatGroq  # pip install langchain-groq
        return ChatGroq(
            model="llama3-8b-8192",          # Free, fast model on Groq
            temperature=0.1,                  # Low temp = more factual answers
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'groq' or 'openai'.")


def format_docs(docs: List[Document]) -> str:
    """Format retrieved chunks into a single context string for the prompt."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        page_str = f", page {page}" if page != "" else ""
        formatted.append(f"[Source {i}: {os.path.basename(source)}{page_str}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore: Chroma, k: int = 4):
    """
    Build the full RAG chain using LangChain's LCEL (LangChain Expression Language).

    The chain (written as a pipeline with |):
      question → retriever → format_docs → prompt → LLM → parse output

    k = number of chunks to retrieve per query. Higher k = more context,
    but also more noise. 3-5 is a good default.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    llm = get_llm()

    # LCEL chain — reads like a pipeline
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def retrieve_chunks(retriever, question: str) -> List[Document]:
    """Return the raw retrieved chunks for a question (used to show sources in UI)."""
    return retriever.invoke(question)


def answer_question(chain, question: str) -> str:
    """Run the full RAG pipeline and return the answer string."""
    return chain.invoke(question)
