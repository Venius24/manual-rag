from search import search
from keyword_search import keyword_search


def hybrid_search(query: str, limit=5):
    vector_results = search(query, limit=20)
    keyword_results = keyword_search(query, limit=20)

    scores = {}
    chunks = {}

    # BGE-M3
    for rank, (chunk_id, text, distance) in enumerate(
        vector_results,
        start=1
    ):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank)
        chunks[chunk_id] = text

    # PostgreSQL FTS
    for rank, (chunk_id, text, fts_score) in enumerate(
        keyword_results,
        start=1
    ):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank)
        chunks[chunk_id] = text

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for chunk_id, rrf_score in ranked[:limit]:
        results.append(
            (chunk_id, chunks[chunk_id], rrf_score)
        )

    return results


if __name__ == "__main__":
    query = input("Вопрос: ")

    results = hybrid_search(query)

    for i, (chunk_id, text, score) in enumerate(results, start=1):
        print(f"\n===== RESULT {i} =====")
        print(f"Chunk ID: {chunk_id}")
        print(f"RRF score: {score:.4f}")
        print()
        print(text[:1000])