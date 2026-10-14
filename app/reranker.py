from sentence_transformers import CrossEncoder


reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")


def rerank(query: str, results: list, limit=5):
    pairs = []

    for chunk_id, text, distance in results:
        pairs.append([query, text])

    scores = reranker.predict(pairs)

    scored_results = []

    for (chunk_id, text, distance), score in zip(results, scores):
        scored_results.append(
            (chunk_id, text, distance, float(score))
        )

    scored_results.sort(
        key=lambda x: x[3],
        reverse=True
    )

    return scored_results[:limit]