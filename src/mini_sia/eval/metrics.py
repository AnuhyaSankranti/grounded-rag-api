import re
from collections.abc import Sequence

from mini_sia.models import AskResponse, RetrievedChunk


def retrieval_recall(retrieved: Sequence[RetrievedChunk], expected_sources: set[str]) -> float:
    if not expected_sources:
        return 1.0
    observed = {item.chunk.filename for item in retrieved}
    return len(observed & expected_sources) / len(expected_sources)


def reciprocal_rank(retrieved: Sequence[RetrievedChunk], expected_sources: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item.chunk.filename in expected_sources:
            return 1.0 / rank
    return 0.0


def context_precision(retrieved: Sequence[RetrievedChunk], expected_sources: set[str]) -> float:
    """Source-level average precision, rewarding relevant evidence ranked early."""
    if not retrieved or not expected_sources:
        return 0.0
    hits = 0
    precision_sum = 0.0
    seen: set[str] = set()
    for rank, item in enumerate(retrieved, start=1):
        filename = item.chunk.filename
        if filename in expected_sources and filename not in seen:
            hits += 1
            precision_sum += hits / rank
            seen.add(filename)
    return precision_sum / len(expected_sources)


def answer_coverage(answer: str, expected_phrases: Sequence[str]) -> float:
    if not expected_phrases:
        return 1.0
    normalized = _normalize(answer)
    matches = sum(_normalize(phrase) in normalized for phrase in expected_phrases)
    return matches / len(expected_phrases)


def citation_validity(response: AskResponse) -> float:
    citations = [int(value) for value in re.findall(r"\[(\d+)]", response.answer)]
    if not response.sources:
        return 1.0 if not citations else 0.0
    if not citations:
        return 0.0
    valid = sum(1 <= citation <= len(response.sources) for citation in citations)
    return valid / len(citations)


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))
