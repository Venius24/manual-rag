import httpx
import openai
import psycopg
import pytest
from fastapi.testclient import TestClient

import main


def _fake_retrieval(monkeypatch, results, pages=None):
    monkeypatch.setattr(main, "retrieve", lambda q, limit=5, mode=None: results)
    monkeypatch.setattr(main, "get_pages", lambda ids: pages or {})


@pytest.fixture
def client():
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_assembles_context_and_sources(client, monkeypatch):
    results = [(11, "Product rule text", 0.12), (4, "Chain rule text", 0.34)]
    _fake_retrieval(monkeypatch, results)
    captured = {}

    def fake_generate(question, context):
        captured["question"], captured["context"] = question, context
        return "Use the product rule."

    monkeypatch.setattr(main, "generate", fake_generate)
    body = client.post("/ask", json={"question": "How to differentiate products?"}).json()

    assert captured["context"] == "[Chunk 11]\nProduct rule text\n\n[Chunk 4]\nChain rule text"
    assert body["answer"] == "Use the product rule."
    assert [s["chunk_id"] for s in body["sources"]] == [11, 4]
    assert body["sources"][0]["score"] == pytest.approx(0.12)


def test_sources_report_page_score_type_and_mode(client, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "rerank")
    _fake_retrieval(monkeypatch, [(7, "t", 4.2), (8, "u", 1.0)], pages={7: 12})
    body = client.post("/ask", json={"question": "q"}).json()
    assert body["retrieval_mode"] == "rerank"
    assert body["sources"][0] == {
        "chunk_id": 7, "page": 12, "score": 4.2, "score_type": "reranker", "text": "t"
    }
    assert body["sources"][1]["page"] is None


def test_ask_requests_top_5(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "retrieve", lambda q, limit=5, mode=None: seen.setdefault("limit", limit) and [])
    monkeypatch.setattr(main, "get_pages", lambda ids: {})
    monkeypatch.setattr(main, "generate", lambda q, c: "x")
    client.post("/ask", json={"question": "q"})
    assert seen["limit"] == 5


def test_ask_with_empty_corpus_passes_empty_context(client, monkeypatch):
    _fake_retrieval(monkeypatch, [])
    ctx = {}
    monkeypatch.setattr(main, "generate", lambda q, c: ctx.setdefault("c", c) or "none")
    body = client.post("/ask", json={"question": "anything"}).json()
    assert ctx["c"] == "" and body["sources"] == []


def test_ask_non_ascii_question(client, monkeypatch):
    _fake_retrieval(monkeypatch, [(1, "Производная ∫", 0.0)])
    monkeypatch.setattr(main, "generate", lambda q, c: f"{q}|{c}")
    r = client.post("/ask", json={"question": "Что такое производная?"}).json()
    assert r["answer"].startswith("Что такое производная?|[Chunk 1]\nПроизводная")


def test_ask_rejects_empty_and_oversized_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422


def test_ask_missing_question_is_422(client):
    assert client.post("/ask", json={}).status_code == 422


def test_ask_returns_503_when_database_is_down(client, monkeypatch):
    def boom(q, limit=5, mode=None):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(main, "retrieve", boom)
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 503
    assert "Database" in r.json()["detail"]


def test_ask_returns_503_when_llm_fails(client, monkeypatch):
    _fake_retrieval(monkeypatch, [(1, "t", 0.1)])

    def boom(q, c):
        raise openai.APITimeoutError(request=httpx.Request("POST", "https://api.deepseek.com"))

    monkeypatch.setattr(main, "generate", boom)
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 503
    assert "Language model" in r.json()["detail"]
