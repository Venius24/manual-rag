import os

import psycopg

from db import DB_URL
from search import search, search_with_rerank
from hybrid_search import hybrid_search


MODES = ("dense", "hybrid", "rerank")
DEFAULT_MODE = "hybrid"

# What the score of each mode means (and which direction is better).
SCORE_TYPES = {
    "dense": "cosine_distance",   # lower is better
    "hybrid": "rrf",              # higher is better (reciprocal rank fusion)
    "rerank": "reranker",         # higher is better (cross-encoder logit)
}


def get_mode():
    mode = os.getenv("RETRIEVAL_MODE", DEFAULT_MODE).strip().lower()

    if mode not in MODES:
        raise ValueError(
            f"RETRIEVAL_MODE must be one of {', '.join(MODES)}, got {mode!r}"
        )

    return mode


def retrieve(query: str, limit=5, mode=None):
    """Return [(chunk_id, text, score)] ranked best-first for the given mode."""
    mode = mode or get_mode()

    if mode == "dense":
        return search(query, limit=limit)

    if mode == "hybrid":
        return hybrid_search(query, limit=limit)

    if mode == "rerank":
        # the reranker also returns the dense distance; keep the reranker score
        return [
            (chunk_id, text, score)
            for chunk_id, text, distance, score
            in search_with_rerank(query, limit=limit)
        ]

    raise ValueError(f"unknown retrieval mode {mode!r}")


def get_pages(chunk_ids):
    """Map chunk id -> page number (None if the chunk was indexed without one)."""
    if not chunk_ids:
        return {}

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, page FROM chunks WHERE id = ANY(%s)",
                (list(chunk_ids),)
            )

            return dict(cur.fetchall())
