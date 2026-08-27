import asyncio
import hashlib
import re
import time
from collections.abc import Sequence

from mini_sia.chunking import chunk_sections
from mini_sia.config import Settings
from mini_sia.loaders import load_sections
from mini_sia.models import AskResponse, Chunk, IngestResponse, Source
from mini_sia.providers import AnswerProvider, EmbeddingProvider
from mini_sia.store import SQLiteHybridStore


class DocumentTooLargeError(ValueError):
    pass


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        store: SQLiteHybridStore,
        embeddings: EmbeddingProvider,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embeddings = embeddings

    async def ingest(self, filename: str, content: bytes) -> IngestResponse:
        if len(content) > self._settings.max_upload_mb * 1024 * 1024:
            raise DocumentTooLargeError(
                f"Document exceeds {self._settings.max_upload_mb} MB upload limit"
            )
        content_sha = hashlib.sha256(content).hexdigest()
        document_id = hashlib.sha256(f"{filename}:{content_sha}".encode()).hexdigest()[:24]
        sections = load_sections(filename, content)
        text_chunks = chunk_sections(
            sections,
            chunk_size_words=self._settings.chunk_size_words,
            overlap_words=self._settings.chunk_overlap_words,
        )
        chunks = [
            Chunk(
                id=hashlib.sha256(
                    f"{document_id}:{position}:{section.page}:{section.text}".encode()
                ).hexdigest()[:32],
                document_id=document_id,
                filename=filename,
                text=section.text,
                position=position,
                page=section.page,
            )
            for position, section in enumerate(text_chunks)
        ]
        embeddings: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            batch = [chunk.text for chunk in chunks[start : start + batch_size]]
            embeddings.extend(await self._embeddings.embed(batch))
        await asyncio.to_thread(
            self._store.upsert_document,
            document_id,
            filename,
            content_sha,
            chunks,
            embeddings,
        )
        return IngestResponse(
            document_id=document_id,
            filename=filename,
            chunks_indexed=len(chunks),
        )


class RagService:
    def __init__(
        self,
        settings: Settings,
        store: SQLiteHybridStore,
        embeddings: EmbeddingProvider,
        answer_provider: AnswerProvider,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embeddings = embeddings
        self._answer_provider = answer_provider

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ):
        query_embedding = (await self._embeddings.embed([question]))[0]
        return await asyncio.to_thread(
            self._store.search,
            question,
            query_embedding,
            top_k=top_k or self._settings.top_k,
            vector_weight=self._settings.vector_weight,
            document_ids=document_ids,
        )

    async def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> AskResponse:
        started = time.perf_counter()
        context = await self.retrieve(question, top_k=top_k, document_ids=document_ids)
        if not context:
            answer = "I could not find that information in the indexed documents."
        else:
            answer = await self._answer_provider.answer(question, context)
            answer = _remove_invalid_citations(answer, len(context))
        sources = [
            Source(
                citation=index,
                document_id=item.chunk.document_id,
                filename=item.chunk.filename,
                chunk_id=item.chunk.id,
                page=item.chunk.page,
                score=round(item.score, 4),
                snippet=_snippet(item.chunk.text),
            )
            for index, item in enumerate(context, start=1)
        ]
        return AskResponse(
            answer=answer,
            sources=sources,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


def _remove_invalid_citations(answer: str, source_count: int) -> str:
    def replace(match: re.Match[str]) -> str:
        citation = int(match.group(1))
        return match.group(0) if 1 <= citation <= source_count else ""

    return re.sub(r"\[(\d+)]", replace, answer).strip()


def _snippet(text: str, limit: int = 280) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"

