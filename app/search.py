import psycopg
from pgvector.psycopg import register_vector

from db import DB_URL
from embeddings import model


def search(query: str, limit=5):

    # превращаем вопрос в 1024 числа
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    text,
                    embedding <=> %s AS distance
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, limit)
            )

            return cur.fetchall()


def search_with_rerank(query: str, limit=5):
    from reranker import rerank

    candidates = search(query, limit=20)

    return rerank(
        query,
        candidates,
        limit=limit
    )


if __name__ == "__main__":

    query = input("Вопрос: ")

    results = search(query)

    for i, (chunk_id, text, distance) in enumerate(results, start=1):

        print(f"\n===== RESULT {i} =====")
        print(f"Chunk ID: {chunk_id}")
        print(f"Distance: {distance:.4f}")
        print()
        print(text[:1000])