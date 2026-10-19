import pytest

import retriever as retrieval


def test_default_mode_is_hybrid(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    assert retrieval.get_mode() == "hybrid"


def test_invalid_mode_rejected(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    with pytest.raises(ValueError, match="RETRIEVAL_MODE"):
        retrieval.get_mode()


def test_mode_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", " Dense ")
    assert retrieval.get_mode() == "dense"


def test_retrieve_dispatches_per_mode(monkeypatch):
    monkeypatch.setattr(retrieval, "search", lambda q, limit: [(1, "d", 0.1)])
    monkeypatch.setattr(retrieval, "hybrid_search", lambda q, limit: [(2, "h", 0.03)])
    monkeypatch.setattr(
        retrieval, "search_with_rerank", lambda q, limit: [(3, "r", 0.2, 9.5)]
    )
    assert retrieval.retrieve("q", mode="dense") == [(1, "d", 0.1)]
    assert retrieval.retrieve("q", mode="hybrid") == [(2, "h", 0.03)]
    # rerank keeps the reranker score, not the dense distance
    assert retrieval.retrieve("q", mode="rerank") == [(3, "r", 9.5)]


def test_retrieve_reads_mode_from_env(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "dense")
    monkeypatch.setattr(retrieval, "search", lambda q, limit: [(1, "d", 0.1)])
    assert retrieval.retrieve("q") == [(1, "d", 0.1)]


def test_get_pages_empty_skips_database():
    assert retrieval.get_pages([]) == {}
