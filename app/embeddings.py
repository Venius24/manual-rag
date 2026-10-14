from sentence_transformers import SentenceTransformer


model = SentenceTransformer("BAAI/bge-m3")


def embed(text: str):
    return model.encode(text).tolist()


if __name__ == "__main__":
    vector = embed("What is the derivative of a function?")

    print("Размерность:", len(vector))
    print("Первые 10 чисел:")
    print(vector[:10])