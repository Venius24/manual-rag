import hybrid_search as hs


def _patch(monkeypatch, vec, kw):
    monkeypatch.setattr(hs, "search", lambda q, limit=20: vec)
    monkeypatch.setattr(hs, "keyword_search", lambda q, limit=20: kw)


def test_both_empty_returns_empty(monkeypatch):
    _patch(monkeypatch, [], [])
    assert hs.hybrid_search("q") == []


def test_chunk_in_both_lists_wins(monkeypatch):
    vec = [(1, "a", 0.1), (2, "b", 0.2)]
    kw = [(2, "b", 3.0), (3, "c", 1.0)]
    _patch(monkeypatch, vec, kw)
    out = hs.hybrid_search("q", limit=3)
    assert [r[0] for r in out] == [2, 1, 3]
    assert out[0][2] == 1 / 62 + 1 / 61


def test_rrf_scores_are_descending_and_limit_applied(monkeypatch):
    vec = [(i, str(i), 0.0) for i in range(1, 11)]
    _patch(monkeypatch, vec, [])
    out = hs.hybrid_search("q", limit=4)
    assert [r[0] for r in out] == [1, 2, 3, 4]
    scores = [r[2] for r in out]
    assert scores == sorted(scores, reverse=True)


def test_keyword_only_results_still_returned(monkeypatch):
    _patch(monkeypatch, [], [(9, "z", 0.5)])
    assert hs.hybrid_search("q") == [(9, "z", 1 / 61)]
