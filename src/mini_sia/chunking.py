import re
from collections.abc import Iterable

from mini_sia.models import TextSection


_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def chunk_sections(
    sections: Iterable[TextSection],
    *,
    chunk_size_words: int,
    overlap_words: int,
) -> list[TextSection]:
    """Split sections into stable word windows without crossing page boundaries."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_size_words")

    chunks: list[TextSection] = []
    step = chunk_size_words - overlap_words
    for section in sections:
        words = normalize_text(section.text).split()
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size_words]
            if not window:
                continue
            chunks.append(TextSection(text=" ".join(window), page=section.page))
            if start + chunk_size_words >= len(words):
                break
    return chunks

