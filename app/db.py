import os


# Local default matches docker-compose.yml (Postgres exposed on 5433).
DB_URL = os.getenv(
    "DB_URL",
    "postgresql://postgres:postgres@127.0.0.1:5433/rag"
)

EMBEDDING_DIM = 1024


def init_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM}) NOT NULL
            )
            """
        )
    conn.commit()
