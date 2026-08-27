import pytest

from mini_sia.chunking import chunk_sections
from mini_sia.models import TextSection


def test_chunking_uses_overlap_and_preserves_page() -> None:
    words = [f"word{index}" for index in range(12)]
    chunks = chunk_sections(
        [TextSection(text=" ".join(words), page=3)],
        chunk_size_words=5,
        overlap_words=2,
    )

    assert [chunk.page for chunk in chunks] == [3, 3, 3, 3]
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert chunks[-1].text.endswith("word11")


def test_chunking_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_words"):
        chunk_sections([TextSection(text="hello")], chunk_size_words=5, overlap_words=5)

