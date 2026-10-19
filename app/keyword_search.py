import psycopg

from db import DB_URL


def keyword_search(query: str, limit=5):

    with psycopg.connect(DB_URL) as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    id,
                    text,
                    ts_rank(tsv, websearch_to_tsquery('english', %s)) AS rank
                FROM chunks
                WHERE tsv @@ websearch_to_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, query, limit)
            )

            return cur.fetchall()


if __name__ == "__main__":

    query = input("Вопрос: ")

    results = keyword_search(query)

    for i, (chunk_id, text, rank) in enumerate(results, start=1):

        print(f"\n===== RESULT {i} =====")
        print(f"Chunk ID: {chunk_id}")
        print(f"FTS rank: {rank:.4f}")
        print()
        print(text[:1000])