"""Retrieval evaluation: recall@{1,3,5,10} and MRR@10 for several retrievers.

Run from the repo root:

    python eval/run_eval.py --seed 0

Writes eval/results/<tag>_<utc-date>.json (config, per-question ranks, metrics).
No database or API key is needed; models are loaded from the local HF cache
(or downloaded once).
"""
import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "eval"))

from ingest import chunk_text, extract_text, normalize_ligatures  # noqa: E402
from retrieval import (BM25, chunk_spans, first_relevant_rank, rank_desc,  # noqa: E402
                       rrf_fuse, summarize)

PDF = ROOT / "data" / "Calculus-Lecture.pdf"
QUESTIONS = ROOT / "eval" / "questions.jsonl"
CACHE = ROOT / "eval" / "cache"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
KS = (1, 3, 5, 10)
POOL = 20  # candidate pool size, same as app/search.py and app/hybrid_search.py

# name -> (chunk_size, overlap, method)
CONFIGS = {
    "dense_1500_0": (1500, 0, "dense"),
    "dense_800_200": (800, 200, "dense"),
    "dense": (1500, 300, "dense"),
    "bm25": (1500, 300, "bm25"),
    "hybrid": (1500, 300, "hybrid"),
    "dense+rerank": (1500, 300, "dense+rerank"),
    "hybrid+rerank": (1500, 300, "hybrid+rerank"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_questions():
    with open(QUESTIONS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_cached(model, texts, key):
    CACHE.mkdir(exist_ok=True)
    h = hashlib.sha256((EMBED_MODEL + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = CACHE / f"{key}_{h}.npy"
    if path.exists():
        return np.load(path)
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=8, show_progress_bar=False)
    np.save(path, vecs)
    return vecs


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def auroc(pos, neg):
    """P(score_pos > score_neg), ties count half."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--fix-ligatures", action="store_true",
                    help="expand PDF ligatures (fi, fl, ...) before chunking")
    ap.add_argument("--tag", default="retrieval")
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    from sentence_transformers import CrossEncoder, SentenceTransformer

    text = extract_text(str(PDF))
    questions = load_questions()
    if args.fix_ligatures:
        # gold spans are offsets into the raw text; map them into the normalised
        # text (ligature expansion is local, so prefix length gives the new offset)
        raw = text
        text = normalize_ligatures(raw)
        for q in questions:
            q["spans"] = [[len(normalize_ligatures(raw[:s])), len(normalize_ligatures(raw[:e]))]
                          for s, e in q["spans"]]
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = None
    qvecs_all = embed_cached(embedder, [q["question"] for q in questions], "questions")
    qvec = {q["id"]: v for q, v in zip(questions, qvecs_all)}

    results = {}
    for name in args.configs:
        size, overlap, method = CONFIGS[name]
        chunks = chunk_text(text, size, overlap)
        ranges = chunk_spans(len(text), size, overlap)
        assert [text[s:e] for s, e in ranges] == chunks
        dense_m = None
        if "dense" in method or "hybrid" in method:
            dense_m = embed_cached(embedder, chunks, f"chunks_{size}_{overlap}")
        bm25 = BM25(chunks) if ("bm25" in method or "hybrid" in method) else None

        per_q, ranks, top1_scores = [], [], {}
        for q in questions:
            cand = []
            if dense_m is not None:
                dscore = dense_m @ qvec[q["id"]]
                dense_rank = rank_desc(dscore, POOL)
                top1_scores[q["id"]] = float(dscore[dense_rank[0]])
            if bm25 is not None:
                bscore = bm25.scores(q["question"])
                bm25_rank = [i for i in rank_desc(bscore, POOL) if bscore[i] > 0]
            if method.startswith("dense"):
                cand = dense_rank
            elif method == "bm25":
                cand = bm25_rank
            else:  # hybrid*
                cand = rrf_fuse([dense_rank, bm25_rank], limit=POOL)
            if method.endswith("+rerank"):
                if reranker is None:
                    reranker = CrossEncoder(RERANK_MODEL)
                sc = reranker.predict([[q["question"], chunks[i]] for i in cand])
                order = sorted(range(len(cand)), key=lambda j: (-float(sc[j]), j))
                cand = [cand[j] for j in order]
            top10 = cand[:10]
            if q["answerable"]:
                r = first_relevant_rank(top10, ranges, [tuple(s) for s in q["spans"]])
                ranks.append(r)
                per_q.append({"id": q["id"], "rank": r, "top10_chunks": top10,
                              "top10_ranges": [ranges[i] for i in top10]})
        metrics = summarize(ranks, KS)
        entry = {"chunk_size": size, "overlap": overlap, "method": method,
                 "n_chunks": len(chunks), "n_answerable": len(answerable),
                 "metrics": metrics, "per_question": per_q}
        if top1_scores:
            a = [top1_scores[q["id"]] for q in answerable]
            u = [top1_scores[q["id"]] for q in unanswerable]
            entry["abstention_dense_top1_cosine"] = {
                "mean_answerable": float(np.mean(a)), "mean_unanswerable": float(np.mean(u)),
                "max_unanswerable": float(max(u)), "min_answerable": float(min(a)),
                "auroc_answerable_vs_not": auroc(a, u)}
        results[name] = entry
        print(f"{name:16s} " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()), flush=True)

    out = {
        "date_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed,
        "git_commit": git_commit(),
        "command": "python eval/run_eval.py " + " ".join(sys.argv[1:]),
        "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL,
        "candidate_pool": POOL, "ks": list(KS),
        "relevance_rule": "chunk covers >=50% of a gold evidence span",
        "corpus_sha256": sha256(PDF), "questions_sha256": sha256(QUESTIONS),
        "corpus_chars": len(text), "fix_ligatures": args.fix_ligatures,
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "device": "cuda" if torch.cuda.is_available() else "cpu"},
        "results": results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.tag}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
