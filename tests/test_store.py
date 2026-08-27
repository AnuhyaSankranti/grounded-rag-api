import asyncio
from pathlib import Path

from mini_sia.models import Chunk
from mini_sia.providers import HashEmbeddingProvider
from mini_sia.store import SQLiteHybridStore


def test_hybrid_search_ranks_relevant_chunk_first(tmp_path: Path) -> None:
    store = SQLiteHybridStore(tmp_path / "store.db")
    store.initialize()
    provider = HashEmbeddingProvider()
    chunks = [
        Chunk("c1", "d1", "aws.md", "Glue jobs detect playback anomalies", 0),
        Chunk("c2", "d1", "aws.md", "Redis caches API metadata", 1),
    ]
    embeddings = asyncio.run(provider.embed([chunk.text for chunk in chunks]))
    store.upsert_document("d1", "aws.md", "sha", chunks, embeddings)
    query_embedding = asyncio.run(provider.embed(["playback anomaly detection"]))[0]

    results = store.search(
        "playback anomaly detection",
        query_embedding,
        top_k=2,
        vector_weight=0.65,
    )

    assert results[0].chunk.id == "c1"
    assert results[0].score >= results[1].score


def test_document_filter_excludes_other_documents(tmp_path: Path) -> None:
    store = SQLiteHybridStore(tmp_path / "store.db")
    store.initialize()
    provider = HashEmbeddingProvider()
    first = Chunk("c1", "d1", "one.md", "anomaly detection", 0)
    second = Chunk("c2", "d2", "two.md", "anomaly detection", 0)
    store.upsert_document("d1", "one.md", "one", [first], asyncio.run(provider.embed([first.text])))
    store.upsert_document(
        "d2", "two.md", "two", [second], asyncio.run(provider.embed([second.text]))
    )
    query = asyncio.run(provider.embed(["anomaly detection"]))[0]

    results = store.search(
        "anomaly detection", query, top_k=5, vector_weight=0.5, document_ids=["d2"]
    )

    assert [result.chunk.document_id for result in results] == ["d2"]
