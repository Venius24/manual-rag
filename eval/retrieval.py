"""In-memory retrievers and metrics used by the eval harness.

The production app keeps chunks in PostgreSQL/pgvector. For evaluation the same
chunking code (`app/ingest.py`) and the same ranking logic are reproduced in
memory so the eval runs without a database. Dense ranking is cosine similarity
over normalised BGE-M3 vectors, i.e. the same ordering as pgvector's `<=>`.
"""
import math
import re
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def chunk_spans(text_len: int, chunk_size: int, overlap: int):
    """(start, end) character range of every chunk produced by ingest.chunk_text."""
    step = chunk_size - overlap
    return [(i, min(i + chunk_size, text_len)) for i in range(0, text_len, step)]


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean() if len(docs) else 0.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str):
        q = tokenize(query)
        out = np.zeros(len(self.tf))
        if not len(self.tf):
            return out
        for i, tf in enumerate(self.tf):
            norm = 1 - self.b + self.b * self.len[i] / (self.avg or 1.0)
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm)
            out[i] = s
        return out


def rank_desc(scores, limit):
    """Indices of the top `limit` scores, ties broken by lower index (deterministic)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:limit]


def rrf_fuse(rankings, k=60, limit=10):
    """Reciprocal Rank Fusion, as in app/hybrid_search.py."""
    fused = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos)
    return [i for i, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]


def is_relevant(chunk, spans, min_frac=0.5):
    """True if the chunk covers >= min_frac of any gold span."""
    cs, ce = chunk
    for s, e in spans:
        overlap = min(ce, e) - max(cs, s)
        if overlap >= min_frac * (e - s):
            return True
    return False


def first_relevant_rank(ranking, chunks, spans):
    """1-based rank of the first relevant chunk in `ranking`, or None."""
    for pos, idx in enumerate(ranking, start=1):
        if is_relevant(chunks[idx], spans):
            return pos
    return None


def summarize(ranks, ks=(1, 3, 5, 10)):
    """recall@k and MRR@max(ks) from a list of first-relevant ranks (None = miss)."""
    n = len(ranks)
    cutoff = max(ks)
    out = {f"recall@{k}": (sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0)
           for k in ks}
    out[f"mrr@{cutoff}"] = (sum(1 / r for r in ranks if r is not None and r <= cutoff) / n
                            if n else 0.0)
    return out
