import json
from pathlib import Path

from search import search_with_rerank


ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "evals" / "questions.json"

with open(EVAL_FILE, encoding="utf-8") as f:
    dataset = json.load(f)


hits = 0
reciprocal_ranks = []


for item in dataset:

    question = item["question"]
    expected = item["expected_text"].lower()

    results = search_with_rerank(question, limit=5)

    if "multiplied together" in question:
        print("\n===== DEBUG PRODUCT RULE =====")

        for i, (chunk_id, text, distance, reranker_score) in enumerate(
            results,
            start=1
        ):
            print(f"\n--- RANK {i} ---")
            print(f"Chunk: {chunk_id}")
            print(f"Vector distance: {distance:.4f}")
            print(f"Reranker score: {reranker_score:.4f}")
            print(text[:700])

    rank = None

    for i, (chunk_id, text, distance, reranker_score) in enumerate(results, start=1):

        if expected in text.lower():
            rank = i
            break

    if rank is not None:
        hits += 1

        reciprocal_rank = 1 / rank
        reciprocal_ranks.append(reciprocal_rank)

        print(
            f"✅ rank={rank} | RR={reciprocal_rank:.2f} | {question}"
        )

    else:
        reciprocal_ranks.append(0)

        print(
            f"❌ rank=- | RR=0.00 | {question}"
        )


recall_at_5 = hits / len(dataset)
mrr = sum(reciprocal_ranks) / len(dataset)


print()
print("====================")
print(f"Recall@5: {recall_at_5:.2f}")
print(f"MRR:      {mrr:.2f}")
print(f"Found:    {hits}/{len(dataset)}")