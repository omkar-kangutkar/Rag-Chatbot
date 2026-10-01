"""Exercise HTTP -> answer orchestration -> vector search without external models."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main
from app import chain, retriever
from app.config import TOP_K


@pytest.fixture
def client():
    # Do not pre-warm real model/vector dependencies in these isolated tests.
    return TestClient(main.app)


@pytest.fixture
def backend(monkeypatch):
    docs = [
        SimpleNamespace(page_content=f"Evidence {i}", metadata={"source": f"doc{i}.txt"})
        for i in range(10)
    ]
    store = Mock()
    store.similarity_search_with_score.side_effect = (
        lambda query, k: [(doc, 0.1) for doc in docs[:k]]
    )
    monkeypatch.setattr(retriever, "get_vectorstore", lambda: store)
    generate = Mock(return_value="An answer supported by the test documents.")
    monkeypatch.setattr(chain, "_generate_answer", generate)
    return store, generate


@pytest.mark.parametrize("top_k", [1, 2, 7, 10])
def test_requested_top_k_reaches_vector_search(client, backend, top_k):
    store, generate = backend
    response = client.post("/chat", json={"question": "What is ML?", "top_k": top_k})
    assert response.status_code == 200
    store.similarity_search_with_score.assert_called_once_with("What is ML?", k=top_k)
    assert len(generate.call_args.args[1]) == top_k
    assert len(response.json()["sources"]) == top_k
    assert response.json()["latency_ms"] >= 0


def test_omitted_top_k_uses_configured_default(client, backend):
    store, _ = backend
    response = client.post("/chat", json={"question": "What is ML?"})
    assert response.status_code == 200
    store.similarity_search_with_score.assert_called_once_with("What is ML?", k=TOP_K)


@pytest.mark.parametrize("top_k", [0, -1, 11, None, 1.5, "invalid"])
def test_invalid_top_k_is_rejected_before_retrieval(client, backend, top_k):
    store, generate = backend
    response = client.post("/chat", json={"question": "What is ML?", "top_k": top_k})
    assert response.status_code == 422
    store.similarity_search_with_score.assert_not_called()
    generate.assert_not_called()


def test_empty_retrieval_returns_refusal_without_calling_llm(client, backend):
    store, generate = backend
    store.similarity_search_with_score.side_effect = None
    store.similarity_search_with_score.return_value = []
    response = client.post("/chat", json={"question": "What is missing?", "top_k": 2})
    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert "couldn't find" in response.json()["answer"]
    generate.assert_not_called()


def test_unready_store_returns_service_unavailable(client, backend):
    store, generate = backend
    store.similarity_search_with_score.side_effect = RuntimeError("Vector store is empty")
    response = client.post("/chat", json={"question": "What is ML?"})
    assert response.status_code == 503
    generate.assert_not_called()


def test_relevance_filter_may_return_fewer_than_top_k(client, backend):
    store, generate = backend
    relevant = SimpleNamespace(page_content="Useful evidence", metadata={"source": "good.txt"})
    unrelated = SimpleNamespace(page_content="Unrelated", metadata={"source": "other.txt"})
    store.similarity_search_with_score.side_effect = None
    store.similarity_search_with_score.return_value = [(relevant, 0.1), (unrelated, 2.0)]
    response = client.post("/chat", json={"question": "What is ML?", "top_k": 2})
    assert response.status_code == 200
    assert [s["source"] for s in response.json()["sources"]] == ["good.txt"]
    assert generate.call_args.args[1] == [relevant]


def test_generation_includes_retrieved_context_and_parses_answer(monkeypatch):
    from langchain_core.messages import AIMessage

    llm = Mock()
    llm.invoke.return_value = AIMessage(content="A supported answer")
    monkeypatch.setattr(chain, "_get_llm", lambda: llm)
    doc = SimpleNamespace(page_content="Grounding text", metadata={"source": "guide.txt"})
    assert chain._generate_answer("What is ML?", [doc]) == "A supported answer"
    messages = llm.invoke.call_args.args[0]
    assert "Grounding text" in messages[0].content
    assert "guide.txt" in messages[0].content
    assert "What is ML?" in messages[0].content


def test_composable_chain_honours_top_k(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    doc = SimpleNamespace(page_content="Grounding text", metadata={"source": "guide.txt"})
    retrieve = Mock(return_value=[doc])
    monkeypatch.setattr(chain, "retrieve", retrieve)
    monkeypatch.setattr(chain, "_get_llm", lambda: RunnableLambda(lambda prompt: "An answer"))
    result = chain.build_rag_chain(top_k=2).invoke({"question": "What is ML?"})
    assert result == "An answer"
    retrieve.assert_called_once_with("What is ML?", top_k=2)
