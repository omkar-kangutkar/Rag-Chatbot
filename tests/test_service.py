import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from app import retriever, service
from app.config import DATA_DIR, TOP_K


def test_service_forwards_question_and_top_k(monkeypatch):
    backend = Mock(return_value={"answer": "Test answer", "sources": []})
    monkeypatch.setattr(service.chain, "ask", backend)

    result = service.ask("What is ML?", top_k=2)

    backend.assert_called_once_with("What is ML?", top_k=2)
    assert result["answer"] == "Test answer"


def test_service_uses_default_top_k(monkeypatch):
    backend = Mock()
    monkeypatch.setattr(service.chain, "ask", backend)

    service.ask("What is ML?")

    backend.assert_called_once_with("What is ML?", top_k=TOP_K)


@pytest.mark.parametrize("folder", [DATA_DIR, "uploaded_documents"])
def test_ingestion_refreshes_cache(monkeypatch, folder):
    fake_module = ModuleType("app.ingest")
    store = object()
    fake_module.run_ingestion = Mock(return_value=store)
    monkeypatch.setitem(sys.modules, "app.ingest", fake_module)
    reset = Mock()
    monkeypatch.setattr(service, "reset_vectorstore_cache", reset)

    result = service.run_ingestion(data_dir=folder)

    fake_module.run_ingestion.assert_called_once_with(data_dir=folder)
    reset.assert_called_once_with()
    assert result is store


def test_failed_ingestion_does_not_reset_cache(monkeypatch):
    fake_module = ModuleType("app.ingest")
    fake_module.run_ingestion = Mock(
        side_effect=ValueError("No documents found")
    )
    monkeypatch.setitem(sys.modules, "app.ingest", fake_module)
    reset = Mock()
    monkeypatch.setattr(service, "reset_vectorstore_cache", reset)

    with pytest.raises(ValueError, match="No documents found"):
        service.run_ingestion()

    reset.assert_not_called()


def test_cache_reset_clears_cached_store(monkeypatch):
    monkeypatch.setattr(retriever, "_vectorstore", object())

    retriever.reset_vectorstore_cache()

    assert retriever._vectorstore is None