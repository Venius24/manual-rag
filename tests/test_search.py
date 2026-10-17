import importlib
import sys

import numpy as np
import pytest

import search as search_mod


class FakeCursor:
    def __init__(self, rows, log):
        self.rows, self.log = rows, log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params):
        self.log.append((sql, params))

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, rows, log):
        self.rows, self.log = rows, log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return FakeCursor(self.rows, self.log)


@pytest.fixture
def fake_db(monkeypatch):
    state = {"rows": [], "log": [], "encoded": []}

    monkeypatch.setattr(
        search_mod.psycopg, "connect", lambda url: FakeConn(state["rows"], state["log"])
    )
    monkeypatch.setattr(search_mod, "register_vector", lambda conn: None)

    def encode(text, normalize_embeddings=False):
        state["encoded"].append((text, normalize_embeddings))
        return np.ones(4, dtype=np.float32)

    monkeypatch.setattr(search_mod.model, "encode", encode, raising=False)
    return state


def test_search_returns_rows_in_db_order(fake_db):
    fake_db["rows"] = [(7, "a", 0.1), (3, "b", 0.2)]
    assert search_mod.search("q", limit=2) == [(7, "a", 0.1), (3, "b", 0.2)]


def test_search_empty_corpus_returns_empty_list(fake_db):
    assert search_mod.search("anything") == []


def test_search_uses_normalized_embedding_and_limit(fake_db):
    search_mod.search("what is a derivative", limit=9)
    assert fake_db["encoded"] == [("what is a derivative", True)]
    sql, params = fake_db["log"][0]
    assert "ORDER BY embedding <=>" in sql
    assert params[-1] == 9


def test_search_empty_and_non_ascii_query_do_not_crash(fake_db):
    search_mod.search("")
    search_mod.search("производная функции ∫")
    assert [t for t, _ in fake_db["encoded"]] == ["", "производная функции ∫"]


def test_search_very_long_query_passed_through(fake_db):
    q = "word " * 100_000
    search_mod.search(q)
    assert fake_db["encoded"][0][0] == q


def test_search_with_rerank_fetches_20_candidates(fake_db, monkeypatch):
    seen = {}

    def fake_search(query, limit=5):
        seen["limit"] = limit
        return [(1, "x", 0.5)]

    def fake_rerank(query, candidates, limit=5):
        seen["rerank"] = (query, candidates, limit)
        return ["ranked"]

    monkeypatch.setattr(search_mod, "search", fake_search)
    monkeypatch.setitem(sys.modules, "reranker", type(sys)("reranker"))
    sys.modules["reranker"].rerank = fake_rerank
    assert search_mod.search_with_rerank("q", limit=3) == ["ranked"]
    assert seen["limit"] == 20
    assert seen["rerank"] == ("q", [(1, "x", 0.5)], 3)


def test_rerank_orders_by_cross_encoder_score(monkeypatch):
    import sentence_transformers

    class FakeCE:
        def __init__(self, name):
            pass

        def predict(self, pairs):
            return [0.1 if "bad" in p[1] else 0.9 for p in pairs]

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCE)
    sys.modules.pop("reranker", None)
    try:
        reranker = importlib.import_module("reranker")
        out = reranker.rerank("q", [(1, "bad", 0.1), (2, "good", 0.9), (3, "bad2", 0.3)], limit=2)
        assert [r[0] for r in out] == [2, 1]
        assert out[0][3] == pytest.approx(0.9)
        assert reranker.rerank("q", [], limit=5) == []
    finally:
        sys.modules.pop("reranker", None)
