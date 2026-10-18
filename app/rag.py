from search import search
from llm import generate


def main():
    question = input("Вопрос: ")

    results = search(question, limit=5)

    context = "\n\n".join(
        f"[Chunk {chunk_id}]\n{text}"
        for chunk_id, text, distance in results
    )

    answer = generate(question, context)

    print("\n===== ANSWER =====")
    print(answer)

    print("\n===== SOURCES =====")

    for chunk_id, text, distance in results:
        print(
            f"Chunk {chunk_id} | distance={distance:.4f}"
        )


if __name__ == "__main__":
    main()
