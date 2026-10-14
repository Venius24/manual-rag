import json
from pathlib import Path

from search import search
from llm import generate, client


ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "evals" / "questions.json"


def judge_answer(question: str, context: str, answer: str):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": """
You are evaluating a RAG system.

Evaluate the answer using ONLY the provided context.

Give three scores from 1 to 5:

correctness:
5 = fully correct
1 = incorrect

groundedness:
5 = fully supported by the context
1 = contains unsupported claims

relevance:
5 = directly answers the question
1 = irrelevant

Return ONLY valid JSON in this exact format:

{
  "correctness": 1,
  "groundedness": 1,
  "relevance": 1
}
"""
            },
            {
                "role": "user",
                "content": f"""
QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""
            }
        ],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    # На случай если модель завернула JSON в ```json
    text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


with open(EVAL_FILE, encoding="utf-8") as f:
    dataset = json.load(f)


correctness_scores = []
groundedness_scores = []
relevance_scores = []


for number, item in enumerate(dataset, start=1):

    question = item["question"]

    # Наш лучший retrieval:
    # BGE-M3 + chunk 1500 + overlap 300
    results = search(question, limit=5)

    context_parts = []

    for chunk_id, text, distance in results:
        context_parts.append(
            f"[Chunk {chunk_id}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    # DeepSeek отвечает
    answer = generate(question, context)

    # DeepSeek оценивает ответ
    scores = judge_answer(
        question,
        context,
        answer
    )

    correctness_scores.append(scores["correctness"])
    groundedness_scores.append(scores["groundedness"])
    relevance_scores.append(scores["relevance"])

    print()
    print("=" * 60)
    print(f"QUESTION {number}/{len(dataset)}")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nJUDGE:")
    print(
        f"Correctness:  {scores['correctness']}/5"
    )
    print(
        f"Groundedness: {scores['groundedness']}/5"
    )
    print(
        f"Relevance:    {scores['relevance']}/5"
    )


avg_correctness = sum(correctness_scores) / len(correctness_scores)
avg_groundedness = sum(groundedness_scores) / len(groundedness_scores)
avg_relevance = sum(relevance_scores) / len(relevance_scores)


print()
print("=" * 60)
print("FINAL GENERATION METRICS")
print("=" * 60)

print(f"Correctness:  {avg_correctness:.2f}/5")
print(f"Groundedness: {avg_groundedness:.2f}/5")
print(f"Relevance:    {avg_relevance:.2f}/5")