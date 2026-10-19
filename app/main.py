import logging
from contextlib import asynccontextmanager

import openai
import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from db import DB_URL, init_schema
from retriever import retrieve, get_pages, get_mode, SCORE_TYPES
from llm import generate


logger = logging.getLogger(__name__)

get_mode()  # fail fast on an invalid RETRIEVAL_MODE


@asynccontextmanager
async def lifespan(app):
    # Bring an existing database up to date (FTS column/index, page column).
    try:
        with psycopg.connect(DB_URL, connect_timeout=5) as conn:
            init_schema(conn)
    except psycopg.Error:
        logger.warning("could not initialise schema at startup", exc_info=True)
    yield

app = FastAPI(
    title="Study RAG API",
    version="1.0.0",
    lifespan=lifespan
)


MAX_QUESTION_CHARS = 2000


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/ask")
def ask(request: AskRequest):

    mode = get_mode()

    try:
        results = retrieve(
            request.question,
            limit=5,
            mode=mode
        )
        pages = get_pages([chunk_id for chunk_id, _, _ in results])
    except psycopg.Error:
        logger.exception("database unavailable")
        raise HTTPException(503, "Database is unavailable, try again later")

    context_parts = []

    for chunk_id, text, score in results:
        context_parts.append(
            f"[Chunk {chunk_id}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    try:
        answer = generate(
            request.question,
            context
        )
    except openai.OpenAIError:
        logger.exception("LLM call failed")
        raise HTTPException(503, "Language model is unavailable, try again later")

    sources = []

    for chunk_id, text, score in results:
        sources.append({
            "chunk_id": chunk_id,
            "page": pages.get(chunk_id),
            "score": float(score),
            "score_type": SCORE_TYPES[mode],
            "text": text
        })

    return {
        "question": request.question,
        "retrieval_mode": mode,
        "answer": answer,
        "sources": sources
    }