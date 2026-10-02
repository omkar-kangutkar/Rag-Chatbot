"""
chain.py
--------
Builds the full RAG chain:
  retrieved chunks → prompt template → LLM → answer

Architecture:
  User query
      │
      ▼
  Retriever (ChromaDB) → top-k chunks
      │
      ▼
  Prompt Template (context + question)
      │
      ▼
  LLM (Groq / OpenAI)
      │
      ▼
  Answer + source metadata
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from langchain_core.documents import Document

from app.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    TOP_K,
)
from app.retriever import retrieve


# ── Prompt Template ──────────────────────────────────────────────────────────
# The prompt asks the model to ground answers in the retrieved context.
# Prompt instructions alone do not guarantee factual or supported answers.

RAG_PROMPT_TEXT = """
You are a helpful assistant that answers questions strictly based on the provided context.

Rules:
- Only use information from the context below to answer
- If the context doesn't contain enough information, say: "I don't have enough information in the provided documents to answer this."
- Be concise and precise
- Cite which document/source you used when possible

Context:
{context}

Question: {question}

Answer:"""


def _get_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_template(RAG_PROMPT_TEXT)


def _format_context(docs: List[Document]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt.
    Includes source metadata so the LLM can cite it.
    """
    sections = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        sections.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def _get_llm():
    """
    Return the LLM based on config.
    Groq is recommended — it's free, fast, and uses Llama 3.
    """
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0.2,    # low = more factual, less creative
            max_tokens=1024,
        )

    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env")
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.2,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'groq' or 'openai'.")


def build_rag_chain(top_k: int = TOP_K):
    """
    Build and return the LangChain RAG chain.
    
    Chain flow:
      {"context": retriever_fn, "question": passthrough} 
          → prompt → llm → output_parser
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    llm = _get_llm()

    chain = (
        {
            "context": lambda x: _format_context(retrieve(x["question"], top_k=top_k)),
            "question": RunnablePassthrough() | (lambda x: x["question"]),
        }
        | _get_prompt()
        | llm
        | StrOutputParser()
    )

    return chain


def _generate_answer(question: str, docs: List[Document]) -> str:
    """Generate an answer; retrieval can be tested without model dependencies."""
    from langchain_core.output_parsers import StrOutputParser

    prompt = _get_prompt().format_messages(
        context=_format_context(docs), question=question
    )
    response = _get_llm().invoke(prompt)
    return StrOutputParser().invoke(response)


def ask(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Main entry point: ask a question, get an answer + sources.
    
    Returns:
        {
            "answer": str,
            "sources": [{"source": str, "excerpt": str}, ...]
        }
    """
    # Retrieve relevant chunks
    docs = retrieve(question, top_k=top_k)

    if not docs:
        return {
            "answer": "I couldn't find any relevant information in the documents.",
            "sources": [],
        }

    # Build context and run through chain
    answer = _generate_answer(question, docs)

    # Build source list for transparency
    sources = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append({
                "source": src,
                "excerpt": doc.page_content[:200] + "...",
            })

    return {
        "answer": answer,
        "sources": sources,
    }
