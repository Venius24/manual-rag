# Manual RAG
n[![tests](https://github.com/Venius24/manual-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/Venius24/manual-rag/actions/workflows/tests.yml)

A Retrieval-Augmented Generation service built from scratch, without LangChain or
LlamaIndex: FastAPI, PostgreSQL/pgvector, BGE-M3 embeddings and DeepSeek, plus a
reproducible retrieval evaluation that compares dense, BM25, hybrid and reranked
retrieval.

## Architecture

```text
PDF ─► PyMuPDF ─► chunking (1500 chars, 300 overlap) ─► BGE-M3 embeddings (1024-d)
                                                              │
                                                              ▼
                                                    PostgreSQL + pgvector
                                                              │
question ─► BGE-M3 embedding ─► cosine search, top-5 ─────────┘
                                        │
                                        ▼
                         DeepSeek, answer from context only ─► answer + sources (FastAPI)
```

The service uses dense retrieval. Hybrid (dense + Postgres full-text search, RRF)
and a cross-encoder reranker are implemented in `app/hybrid_search.py` and
`app/search.py::search_with_rerank`. Both scored higher on top-rank quality in the
evaluation below, but they are not the default yet: the evaluated hybrid uses
BM25 rather than the app's Postgres full-text search, and the reranker adds
seconds of CPU latency per query. Switching the default is the next step.

## Stack

Python · FastAPI · PostgreSQL · pgvector · psycopg · BAAI/bge-m3 · bge-reranker-v2-m3 ·
Sentence Transformers · PyMuPDF · DeepSeek API · Docker Compose · pytest

## Results (reproducible retrieval evaluation)

All numbers below come from `eval/run_eval.py` runs whose raw output is committed
in `eval/results/`. Corpus: one PDF, *Calculus I* lecture notes (228 pages,
~165k characters, 138 chunks at 1500/300). Eval set: 50 hand-written questions,
44 answerable from a specific passage + 6 unanswerable (`eval/questions.jsonl`,
frozen before any experiment; construction in `eval/README.md`). A chunk counts as
relevant if it covers at least half of the gold evidence span. Metrics are over
the 44 answerable questions; MRR is cut off at rank 10.

Baseline pipeline as shipped (text extracted as-is), run `2026-10-17`, seed 0:

| Retriever | Chunking | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|
| Dense (BGE-M3), no overlap | 1500/0 | 0.614 | 0.909 | 0.955 | 1.000 | 0.761 |
| Dense (BGE-M3) | 800/200 | 0.682 | 0.932 | 0.977 | 1.000 | 0.805 |
| Dense (BGE-M3), app default | 1500/300 | 0.705 | 0.932 | 0.955 | 1.000 | 0.825 |
| BM25 | 1500/300 | 0.591 | 0.773 | 0.795 | 0.909 | 0.698 |
| Hybrid (dense + BM25, RRF k=60) | 1500/300 | 0.795 | 0.886 | 0.955 | 0.977 | 0.857 |
| Dense + bge-reranker-v2-m3 (top-20) | 1500/300 | 0.818 | 0.909 | 0.932 | 1.000 | 0.880 |
| Hybrid + reranker (top-20) | 1500/300 | 0.818 | 0.932 | 0.932 | 1.000 | 0.882 |

After expanding PDF ligatures (`ﬁ`→`fi`) before chunking (reranker rows not re-run):

| Retriever | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.727 | 0.886 | 0.955 | 1.000 | 0.831 |
| BM25 | 0.659 | 0.795 | 0.841 | 0.932 | 0.747 |
| Hybrid | 0.841 | 0.955 | 0.977 | 0.977 | 0.898 |

What this shows: reranking and hybrid fusion both improved top-rank quality
(MRR 0.825 → 0.857 / 0.880) on this set, while recall@5 stayed flat or dipped
slightly; chunk overlap helped at rank 1 as the earlier README claimed. The
ligature fix helps keyword matching (BM25/hybrid) clearly; for dense it is within
noise. See `eval/ANALYSIS.md` for per-question failures. Details on abstention are
there too: top-1 similarity separates answerable from unanswerable questions
only partially (AUROC ≈ 0.95 on 6 negatives, with overlapping ranges).

### Reproduce

```bash
python -m venv venv && venv/Scripts/activate     # Windows; use venv/bin/activate on Linux/macOS
pip install -r requirements-dev.txt
pytest                                           # 45 tests, offline
python eval/run_eval.py --seed 0                 # baseline rows (reranker rows take ~1h on CPU)
python eval/run_eval.py --seed 0 --fix-ligatures --configs dense bm25 hybrid --tag ligature_fix
```

Models (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`) are downloaded from Hugging Face
on first use. No database or API key is needed for tests or eval.

### Limits — what these numbers do not prove

- One 228-page document and 44 answerable questions: one question is 0.023 recall,
  and most differences between rows are a handful of questions. No confidence
  intervals were computed; treat gaps under ~0.05 as noise.
- Gold labels were written by an AI agent and mechanically checked (evidence text
  exists in the PDF) but not reviewed by a human; only the primary passage of each
  fact is labelled, which understates every method.
- The eval runs an in-memory re-implementation of the pgvector search (same
  chunking code, cosine ranking). The "BM25" row is not the app's PostgreSQL
  full-text search, which was not run (no Postgres in the eval environment).
- The set was frozen before any experiment, but the
  ligature fix was chosen after seeing failures on this same set, so its
  before/after gain is optimistic.
- Retrieval only. The generation scores below come from an earlier 10-question set
  and were not re-run on this one; abstention behaviour of the generator was not
  evaluated.

### How this was built

The pipeline, the API and the first reranker/hybrid experiments were written by
hand. The test suite, the 50-question eval set, the harness and the error
analysis were added with an AI coding agent (Claude Code) and checked by the
tests; every reported number comes from the committed result files.

## Generation Evaluation

Final RAG answers on the earlier 10-question set (`evals/questions.json`, script `app/evaluate_answer.py`) were evaluated using an LLM-as-a-Judge approach.

The judge scored:

- Correctness
- Groundedness
- Relevance

Results:

| Metric | Score |
|---|---:|
| Correctness | **4.90 / 5** |
| Groundedness | **5.00 / 5** |
| Relevance | **4.90 / 5** |

The generation evaluation currently uses DeepSeek both as the answer generator and as the judge.

Because the judge is not independent from the generator, these scores should be interpreted as development metrics rather than an unbiased benchmark.

## API

Run the service and open:

```text
http://127.0.0.1:8000/docs
```

### POST /ask

Request:

```json
{
  "question": "How do I differentiate two functions that are multiplied together?"
}
```

Response:

```json
{
  "question": "How do I differentiate two functions that are multiplied together?",
  "answer": "Use the Product Rule...",
  "sources": [
    {
      "chunk_id": 240,
      "distance": 0.41,
      "text": "..."
    }
  ]
}
```

### GET /health

Returns the current API status.

## Running with Docker

```bash
cp .env.example .env        # then set DEEPSEEK_API_KEY
docker compose up --build -d
python app/index.py         # creates the table and indexes data/Calculus-Lecture.pdf
```

`app/index.py` connects to `DB_URL` (default: the compose Postgres on
`127.0.0.1:5433`) and creates the `vector` extension and the `chunks` table if
they do not exist. The API is at `http://127.0.0.1:8000`, Swagger at `/docs`.
Inside Docker the API reaches Postgres at `postgres:5432`.

For a quick terminal session without the API: `python app/rag.py`.

## Project structure

```text
manual-rag/
├── app/
│   ├── main.py              # FastAPI: /ask, /health
│   ├── db.py                # DB_URL and schema
│   ├── ingest.py            # PDF extraction and chunking
│   ├── index.py             # embed chunks and write them to pgvector
│   ├── embeddings.py        # BGE-M3
│   ├── search.py            # dense search, dense + reranker
│   ├── keyword_search.py    # Postgres full-text search
│   ├── hybrid_search.py     # RRF fusion
│   ├── reranker.py          # bge-reranker-v2-m3
│   ├── llm.py               # DeepSeek client
│   ├── rag.py               # terminal Q&A
│   └── evaluate_answer.py   # LLM-as-a-judge generation eval
├── eval/                    # frozen 50-question set, harness, results, analysis
├── evals/questions.json     # earlier 10-question set used by evaluate_answer.py
├── tests/                   # offline pytest suite
├── data/Calculus-Lecture.pdf
├── Dockerfile
└── docker-compose.yml
```
