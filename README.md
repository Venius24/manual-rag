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

## Retrieval Evaluation

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
│   └── questions.json
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
