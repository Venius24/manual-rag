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


def extract_pages(pdf_path: str, fix_ligatures=False):
    """Return the text of each PDF page (index 0 = page 1)."""
    doc = pymupdf.open(pdf_path)

    pages = []

    for page in doc:
        text = page.get_text().replace("\x00", "")
        pages.append(normalize_ligatures(text) if fix_ligatures else text)

    return pages


def extract_text(pdf_path: str, fix_ligatures=False):
    return "".join(extract_pages(pdf_path, fix_ligatures))


def chunk_text(text: str, chunk_size=1500, overlap=300):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


def chunk_pages(pages: list, chunk_size=1500, overlap=300):
    """Chunk the joined pages; return (chunk, page) where page (1-based) is
    the page the chunk starts on."""
    text = "".join(pages)
    chunks = chunk_text(text, chunk_size, overlap)

    starts = []
    offset = 0
    for page_text in pages:
        starts.append(offset)
        offset += len(page_text)

    step = chunk_size - overlap
    result = []
    page_index = 0

    for n, chunk in enumerate(chunks):
        start = n * step
        while page_index + 1 < len(starts) and starts[page_index + 1] <= start:
            page_index += 1
        result.append((chunk, page_index + 1))

    return result


if __name__ == "__main__":
    text = extract_text("data/Calculus-Lecture.pdf")
    chunks = chunk_text(text)

    print("Всего символов:", len(text))
    print("Всего чанков:", len(chunks))
    print(chunks[0])