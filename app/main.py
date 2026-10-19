from fastapi import FastAPI
from pydantic import BaseModel, Field

from search import search
from llm import generate


app = FastAPI(
    title="Study RAG API",
    version="1.0.0"
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

    results = search(
        request.question,
        limit=5
    )

    context_parts = []

    for chunk_id, text, distance in results:
        context_parts.append(
            f"[Chunk {chunk_id}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    answer = generate(
        request.question,
        context
    )

    sources = []

    for chunk_id, text, distance in results:
        sources.append({
            "chunk_id": chunk_id,
            "distance": float(distance),
            "text": text
        })

    return {
        "question": request.question,
        "answer": answer,
        "sources": sources
    }