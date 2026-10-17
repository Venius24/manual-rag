# Manual RAG

A Retrieval-Augmented Generation system built from scratch without LangChain or LlamaIndex.

The project implements the RAG pipeline manually without LangChain or LlamaIndex and includes retrieval evaluation, generation evaluation, and experiments with different retrieval strategies.

RAG service built from scratch with FastAPI, PostgreSQL/pgvector, BGE-M3, DeepSeek and evaluation pipelines.

## Architecture

```text
PDF
 ↓
PyMuPDF
 ↓
Chunking
 ↓
BGE-M3 embeddings
 ↓
PostgreSQL + pgvector
 ↓
Cosine similarity search
 ↓
Top-5 chunks
 ↓
DeepSeek
 ↓
Answer + sources
 ↓
FastAPI
```

## Stack

- Python
- FastAPI
- PostgreSQL
- pgvector
- Docker / Docker Compose
- BAAI/bge-m3
- Sentence Transformers
- DeepSeek API
- PyMuPDF
- psycopg

## RAG Pipeline

Documents are extracted from PDF files and split into overlapping chunks.

Current chunking configuration:

```text
chunk_size = 1500 characters
overlap    = 300 characters
```

Each chunk is embedded using `BAAI/bge-m3`.

Embedding dimension:

```text
1024
```

Embeddings and source text are stored in PostgreSQL using pgvector.

At query time:

```text
Question
   ↓
BGE-M3 embedding
   ↓
pgvector cosine search
   ↓
Top-5 chunks
   ↓
DeepSeek
   ↓
Answer grounded in retrieved context
```

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
- The runs were tuned on nothing: the set was frozen before experiments, but the
  ligature fix was chosen after seeing failures on this same set, so its
  before/after gain is optimistic.
- Retrieval only. The generation scores in the section below were produced by the
  original author with an LLM judge and were not re-run; abstention behaviour of
  the generator was not evaluated.
- The earlier "Retrieval Evaluation" table below (10-question set) is the original
  author's and is not reproduced by any committed script; it disagrees with the
  results above on reranking/hybrid, and the newer, larger set should be preferred.

### How this was built

The tests, eval set, harness and analysis on branch `feat/eval-and-tests` were
developed with an AI coding agent (Claude Code) and verified by the test suite and
the eval harness; all reported numbers come from the committed result files.

## Retrieval Evaluation (original, 10 questions — legacy)

A small paraphrased evaluation dataset was created to test whether the retriever can find relevant information when the question does not exactly match the wording of the source document.

Metrics:

- Recall@5
- Mean Reciprocal Rank (MRR)

### Experiments

| Experiment | Chunk size | Overlap | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BGE-M3 baseline | 1500 | 0 | 0.60 | 0.39 |
| BGE-M3 + overlap | 1500 | 300 | **0.80** | **0.41** |
| + BGE reranker | 1500 | 300 | 0.50 | 0.35 |
| + Hybrid Search (FTS + RRF) | 1500 | 300 | 0.70 | 0.38 |

The simplest BGE-M3 retrieval pipeline with overlapping chunks produced the best results on the current evaluation set.

Both reranking and hybrid retrieval were tested but rejected because they reduced measured retrieval quality.

### Evaluation limitation

The current retrieval evaluation uses expected text matches inside retrieved chunks.

Therefore Recall@5 and MRR should be treated as proxy metrics rather than a complete semantic relevance evaluation.

A larger manually annotated evaluation dataset would provide a more reliable measurement.

## Generation Evaluation

Final RAG answers were evaluated using an LLM-as-a-Judge approach.

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

Create a `.env` file:

```env
DEEPSEEK_API_KEY=your_key_here
```

Start the application:

```bash
docker compose up --build
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

PostgreSQL is exposed locally on port `5433`.

Inside Docker, the API communicates with PostgreSQL using:

```text
postgres:5432
```

## Project Structure

```text
my-rag/
├── app/
│   ├── main.py
│   ├── ingest.py
│   ├── index.py
│   ├── embeddings.py
│   ├── search.py
│   ├── llm.py
│   ├── rag.py
│   ├── evaluate.py
│   ├── evaluate_answers.py
│   ├── reranker.py
│   ├── keyword_search.py
│   └── hybrid_search.py
│
├── data/
│   └── lecture.pdf
│
├── evals/
│   └── questions.json        # original 10-question set
├── eval/                     # frozen 50-question set, harness, results, analysis
├── tests/                    # offline pytest suite
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Key Findings

Increasing retrieval complexity did not automatically improve quality.

Adding chunk overlap improved Recall@5 from `0.60` to `0.80`.

However:

```text
Reranker:
0.80 → 0.50 Recall@5

Hybrid Search:
0.80 → 0.70 Recall@5
```

Both approaches were therefore rejected for the final pipeline.

This project uses evaluation-driven development: retrieval changes are measured before being accepted into the final RAG pipeline.