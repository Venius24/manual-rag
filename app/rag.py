from search import search
from llm import generate


question = input("Вопрос: ")

results = search(question, limit=5)

context_parts = []

for chunk_id, text, distance in results:
    context_parts.append(
        f"[Chunk {chunk_id}]\n{text}"
    )

context = "\n\n".join(context_parts)

answer = generate(question, context)

print("\n===== ANSWER =====")
print(answer)

print("\n===== SOURCES =====")

for chunk_id, text, distance in results:
    print(
        f"Chunk {chunk_id} | distance={distance:.4f}"
    )