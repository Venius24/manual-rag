import psycopg
from pgvector.psycopg import register_vector

from ingest import extract_text, chunk_text
from embeddings import model


PDF_PATH = "data/Calculus-Lecture.pdf"

DB_URL = "postgresql://postgres:postgres@127.0.0.1:5433/rag"


# 1. Читаем PDF
text = extract_text(PDF_PATH, fix_ligatures=True)


# 2. Режем его
chunks = chunk_text(text)

print("Чанков:", len(chunks))


# 3. Делаем embeddings СРАЗУ для всех чанков
embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embeddings готовы:", embeddings.shape)


# 4. Подключаемся к PostgreSQL
with psycopg.connect(DB_URL) as conn:

    register_vector(conn)

    with conn.cursor() as cur:

        # чтобы повторный запуск не создавал дубликаты
        cur.execute("DELETE FROM chunks")

        # 5. Сохраняем каждый chunk + embedding
        for chunk, embedding in zip(chunks, embeddings):

            cur.execute(
                """
                INSERT INTO chunks (text, embedding)
                VALUES (%s, %s)
                """,
                (chunk, embedding)
            )

    conn.commit()


print("Готово! Чанки записаны в PostgreSQL.")