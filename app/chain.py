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

from typing import Any, Dict, List

from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
)
from app.retriever import retrieve


# ── Prompt Template ──────────────────────────────────────────────────────────
# This is the core of RAG — the LLM only uses the provided context, 
# not its training data. This grounds the answers in YOUR documents.

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions strictly based on the provided context.

Rules:
- Only use information from the context below to answer
- If the context doesn't contain enough information, say: "I don't have enough information in the provided documents to answer this."
- Be concise and precise
- Cite which document/source you used when possible

Context:
{context}

Question: {question}

Answer:""")


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


def build_rag_chain():
    """
    Build and return the LangChain RAG chain.
    
    Chain flow:
      {"context": retriever_fn, "question": passthrough} 
          → prompt → llm → output_parser
    """
    llm = _get_llm()

    chain = (
        {
            "context": lambda x: _format_context(retrieve(x["question"])),
            "question": RunnablePassthrough() | (lambda x: x["question"]),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def ask(question: str) -> Dict[str, Any]:
    """
    Main entry point: ask a question, get an answer + sources.
    
    Returns:
        {
            "answer": str,
            "sources": [{"source": str, "excerpt": str}, ...]
        }
    """
    # Retrieve relevant chunks
    docs = retrieve(question)

    if not docs:
        return {
            "answer": "I couldn't find any relevant information in the documents.",
            "sources": [],
        }

    # Build context and run through chain
    llm = _get_llm()
    context = _format_context(docs)

    prompt = RAG_PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(prompt)
    answer = StrOutputParser().invoke(response)

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
