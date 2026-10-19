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
        cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page INTEGER")
        # Generated column + GIN index for full-text search. Both are
        # idempotent; on an existing table Postgres backfills the column.
        cur.execute(
            """
            ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv)"
        )
    conn.commit()
