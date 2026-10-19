from retriever import retrieve, get_mode, SCORE_TYPES
from llm import generate


def main():
    question = input("Вопрос: ")

    mode = get_mode()
    results = retrieve(question, limit=5, mode=mode)

    context = "\n\n".join(
        f"[Chunk {chunk_id}]\n{text}"
        for chunk_id, text, score in results
    )

    answer = generate(question, context)

    print("\n===== ANSWER =====")
    print(answer)

    print("\n===== SOURCES =====")

    for chunk_id, text, score in results:
        print(
            f"Chunk {chunk_id} | {SCORE_TYPES[mode]}={score:.4f}"
        )


if __name__ == "__main__":
    main()
