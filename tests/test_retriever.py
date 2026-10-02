"""Regression tests for vector-store loading and caching."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from app import retriever


def test_empty_store_is_not_cached_and_can_recover(monkeypatch):
    empty_store = Mock()
    empty_store._collection.count.return_value = 0

    populated_store = Mock()
    populated_store._collection.count.return_value = 1

    chroma_factory = Mock(
        side_effect=[empty_store, empty_store, populated_store]
    )

    vectorstores_module = ModuleType("langchain_community.vectorstores")
    vectorstores_module.Chroma = chroma_factory

    embeddings_module = ModuleType("langchain_huggingface")
    embeddings_module.HuggingFaceEmbeddings = Mock()

    monkeypatch.setitem(
        sys.modules,
        "langchain_community.vectorstores",
        vectorstores_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface",
        embeddings_module,
    )
    monkeypatch.setattr(retriever, "_vectorstore", None)

    # Both attempts must reject an empty store.
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Vector store is empty"):
            retriever.get_vectorstore()

    # A later populated store must load successfully and be cached.
    assert retriever.get_vectorstore() is populated_store
    assert retriever.get_vectorstore() is populated_store
    assert chroma_factory.call_count == 3
    
