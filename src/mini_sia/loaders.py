from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from mini_sia.models import TextSection


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


class UnsupportedDocumentError(ValueError):
    pass


def load_sections(filename: str, content: bytes) -> list[TextSection]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise UnsupportedDocumentError(f"Unsupported file type; expected one of: {supported}")

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        sections = [
            TextSection(text=page.extract_text() or "", page=page_number)
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnsupportedDocumentError("Text documents must be UTF-8 encoded") from exc
        sections = [TextSection(text=text)]

    non_empty = [section for section in sections if section.text.strip()]
    if not non_empty:
        raise UnsupportedDocumentError("Document contains no extractable text")
    return non_empty

