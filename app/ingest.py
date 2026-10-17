import pymupdf


LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}


def normalize_ligatures(text: str):
    """Expand typographic ligatures (e.g. 'ﬁ' -> 'fi') that PDFs embed in words."""
    for lig, plain in LIGATURES.items():
        text = text.replace(lig, plain)
    return text


def extract_text(pdf_path: str, fix_ligatures=False):
    doc = pymupdf.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    text = text.replace("\x00", "")

    return normalize_ligatures(text) if fix_ligatures else text


def chunk_text(text: str, chunk_size=1500, overlap=300):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    text = extract_text("data/Calculus-Lecture.pdf")
    chunks = chunk_text(text)

    print("Всего символов:", len(text))
    print("Всего чанков:", len(chunks))
    print(chunks[0])