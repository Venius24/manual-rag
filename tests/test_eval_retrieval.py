import pytest

from ingest import chunk_text
from retrieval import (BM25, chunk_spans, first_relevant_rank, is_relevant,
                       rank_desc, rrf_fuse, summarize, tokenize)


def test_chunk_spans_match_chunk_text():
    text = "".join(chr(97 + i % 26) for i in range(1000))
    spans = chunk_spans(len(text), 100, 30)
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert [text[s:e] for s, e in spans] == chunks


def test_chunk_spans_empty():
    assert chunk_spans(0, 100, 10) == []


def test_bm25_prefers_rare_matching_term():
    docs = ["the cat sat", "the dog barked loudly", "the the the the"]
    scores = BM25(docs).scores("dog")
    assert rank_desc(scores, 3)[0] == 1
    assert scores[0] == 0 and scores[2] == 0


def test_bm25_empty_corpus_and_empty_query():
    assert len(BM25([]).scores("x")) == 0
    assert BM25(["a b"]).scores("").tolist() == [0.0]


def test_bm25_non_ascii_and_case():
    docs = ["Производная функции", "интеграл"]
    assert rank_desc(BM25(docs).scores("ПРОИЗВОДНАЯ"), 2)[0] == 0
    assert tokenize("Hello, Мир!") == ["hello", "мир"]


def test_bm25_long_document_is_normalised():
    short = "derivative rule"
    long = "derivative rule " + "filler " * 5000
    s = BM25([short, long]).scores("derivative rule")
    assert s[0] > s[1]


def test_rank_desc_ties_break_by_index():
    assert rank_desc([1.0, 3.0, 3.0, 2.0], 3) == [1, 2, 3]


def test_rank_desc_limit_larger_than_corpus():
    assert rank_desc([0.1], 10) == [0]


def test_rrf_item_in_both_lists_wins():
    assert rrf_fuse([[1, 2], [2, 3]], limit=3) == [2, 1, 3]


def test_rrf_empty_and_limit():
    assert rrf_fuse([[], []]) == []
    assert rrf_fuse([[1, 2, 3, 4]], limit=2) == [1, 2]


def test_is_relevant_threshold():
    assert is_relevant((0, 100), [(10, 60)])
    assert is_relevant((50, 150), [(10, 110)])  # 60/100 covered
    assert not is_relevant((90, 190), [(10, 110)])  # 20/100 covered
    assert not is_relevant((0, 10), [])


def test_first_relevant_rank_and_miss():
    chunks = [(0, 100), (100, 200), (200, 300)]
    assert first_relevant_rank([2, 1, 0], chunks, [(110, 150)]) == 2
    assert first_relevant_rank([0, 2], chunks, [(110, 150)]) is None


def test_summarize():
    s = summarize([1, 2, None, 6, 12])
    assert s["recall@1"] == pytest.approx(0.2)
    assert s["recall@3"] == pytest.approx(0.4)
    assert s["recall@10"] == pytest.approx(0.6)
    assert s["mrr@10"] == pytest.approx((1 + 0.5 + 1 / 6) / 5)
    assert summarize([])["recall@5"] == 0.0
