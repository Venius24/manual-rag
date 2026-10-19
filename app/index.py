import sys

import psycopg
from pgvector.psycopg import register_vector

from db import DB_URL, init_schema
from ingest import extract_pages, chunk_pages
from embeddings import model


PDF_PATH = "data/Calculus-Lecture.pdf"


def main(pdf_path=PDF_PATH):
    # 1. Читаем PDF
    pages = extract_pages(pdf_path, fix_ligatures=True)

    # 2. Режем его (запоминая страницу начала каждого чанка)
    chunk_list = chunk_pages(pages)
    chunks = [chunk for chunk, _ in chunk_list]
    chunk_pages_ = [page for _, page in chunk_list]

    print("Чанков:", len(chunks))

    # 3. Делаем embeddings СРАЗУ для всех чанков
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embeddings готовы:", embeddings.shape)

    # 4. Подключаемся к PostgreSQL и создаём таблицу, если её ещё нет
    with psycopg.connect(DB_URL) as conn:

        init_schema(conn)
        register_vector(conn)

        with conn.cursor() as cur:

            # чтобы повторный запуск не создавал дубликаты
            cur.execute("DELETE FROM chunks")

            # 5. Сохраняем каждый chunk + embedding
            for chunk, page, embedding in zip(chunks, chunk_pages_, embeddings):

                cur.execute(
                    """
                    INSERT INTO chunks (text, page, embedding)
                    VALUES (%s, %s, %s)
                    """,
                    (chunk, page, embedding)
                )

        conn.commit()

    print("Готово! Чанки записаны в PostgreSQL.")


if __name__ == "__main__":
    main(*sys.argv[1:2])
