import pytest

from ingest import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []


def test_short_text_single_chunk():
    assert chunk_text("hello", chunk_size=100, overlap=10) == ["hello"]


def test_chunk_size_and_overlap_respected():
    text = "".join(chr(ord("a") + i % 26) for i in range(1000))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    # consecutive chunks share exactly `overlap` characters
    for a, b in zip(chunks, chunks[1:]):
        assert a[-20:] == b[:20]


def test_chunks_cover_whole_text():
    text = "0123456789" * 57
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    step = 40
    rebuilt = chunks[0] + "".join(c[10:] for c in chunks[1:])
    assert rebuilt.startswith(text[: len(rebuilt)]) and len(rebuilt) >= len(text)
    assert chunks[-1].endswith(text[-1])
    assert len(chunks) == len(range(0, len(text), step))


def test_zero_overlap_partitions_text():
    text = "abcdefghij" * 10
    assert "".join(chunk_text(text, chunk_size=30, overlap=0)) == text


def test_non_ascii_text_chunks_by_codepoint():
    text = "Нейронная сеть 微分 ∫ f'(x) dx 🙂" * 20
    chunks = chunk_text(text, chunk_size=40, overlap=5)
    assert chunks[0] == text[:40]
    assert "".join([chunks[0]] + [c[5:] for c in chunks[1:]]).startswith(text)


def test_very_long_input_chunk_count():
    text = "x" * 1_000_000
    chunks = chunk_text(text)  # defaults 1500 / 300
    assert len(chunks) == len(range(0, 1_000_000, 1200))
    assert len(chunks[0]) == 1500


def test_overlap_not_smaller_than_chunk_size_is_rejected():
    # step <= 0 would make range() raise a bare ValueError / loop nothing useful
    with pytest.raises(ValueError):
        chunk_text("abc" * 100, chunk_size=10, overlap=10)


def test_overlap_larger_than_chunk_size_is_rejected():
    # regression: negative step made range() empty, so all text was silently dropped
    with pytest.raises(ValueError):
        chunk_text("abc" * 100, chunk_size=10, overlap=20)


def test_extract_text_reads_all_pages(tmp_path):
    import pymupdf

    from ingest import extract_text

    path = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for body in ("First page", "Second page"):
        doc.new_page().insert_text((72, 72), body)
    doc.save(path)
    text = extract_text(str(path))
    assert "First page" in text and "Second page" in text
    assert text.index("First") < text.index("Second")
    assert "\x00" not in text


def test_normalize_ligatures_expands_words():
    from ingest import normalize_ligatures

    assert normalize_ligatures("Deﬁnition of inﬂection, eﬃcient") == "Definition of inflection, efficient"
    assert normalize_ligatures("plain ascii") == "plain ascii"
    assert normalize_ligatures("") == ""


def test_extract_text_ligatures_opt_in(monkeypatch):
    import ingest

    class Page:
        def get_text(self):
            return "deﬁnition \x00of inﬂection"

    monkeypatch.setattr(ingest.pymupdf, "open", lambda path: [Page()])
    assert ingest.extract_text("x.pdf") == "deﬁnition of inﬂection"
    assert ingest.extract_text("x.pdf", fix_ligatures=True) == "definition of inflection"


def test_chunk_pages_tracks_start_page():
    from ingest import chunk_pages

    pages = ["a" * 100, "b" * 100, "c" * 100]
    out = chunk_pages(pages, chunk_size=60, overlap=20)  # chunk starts 0,40,...,280
    assert [p for _, p in out] == [1, 1, 1, 2, 2, 3, 3, 3]
    assert "".join(c for c, _ in out)[:60] == ("a" * 60)


def test_chunk_pages_empty():
    from ingest import chunk_pages

    assert chunk_pages([]) == []
    assert chunk_pages([""]) == []
