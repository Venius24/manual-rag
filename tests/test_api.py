import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_assembles_context_and_sources(client, monkeypatch):
    results = [(11, "Product rule text", 0.12), (4, "Chain rule text", 0.34)]
    monkeypatch.setattr(main, "search", lambda q, limit=5: results)
    captured = {}

    def fake_generate(question, context):
        captured["question"], captured["context"] = question, context
        return "Use the product rule."

    monkeypatch.setattr(main, "generate", fake_generate)
    body = client.post("/ask", json={"question": "How to differentiate products?"}).json()

    assert captured["context"] == "[Chunk 11]\nProduct rule text\n\n[Chunk 4]\nChain rule text"
    assert body["answer"] == "Use the product rule."
    assert [s["chunk_id"] for s in body["sources"]] == [11, 4]
    assert body["sources"][0]["distance"] == pytest.approx(0.12)


def test_ask_requests_top_5(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "search", lambda q, limit=5: seen.setdefault("limit", limit) and [])
    monkeypatch.setattr(main, "generate", lambda q, c: "x")
    client.post("/ask", json={"question": "q"})
    assert seen["limit"] == 5


def test_ask_with_empty_corpus_passes_empty_context(client, monkeypatch):
    monkeypatch.setattr(main, "search", lambda q, limit=5: [])
    ctx = {}
    monkeypatch.setattr(main, "generate", lambda q, c: ctx.setdefault("c", c) or "none")
    body = client.post("/ask", json={"question": "anything"}).json()
    assert ctx["c"] == "" and body["sources"] == []


def test_ask_non_ascii_and_empty_question(client, monkeypatch):
    monkeypatch.setattr(main, "search", lambda q, limit=5: [(1, "Производная ∫", 0.0)])
    monkeypatch.setattr(main, "generate", lambda q, c: f"{q}|{c}")
    r = client.post("/ask", json={"question": "Что такое производная?"}).json()
    assert r["answer"].startswith("Что такое производная?|[Chunk 1]\nПроизводная")
    assert client.post("/ask", json={"question": ""}).status_code == 200


def test_ask_missing_question_is_422(client):
    assert client.post("/ask", json={}).status_code == 422
