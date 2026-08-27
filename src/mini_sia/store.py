import json
import math
import re
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from mini_sia.models import Chunk, RetrievedChunk


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class SQLiteHybridStore:
    """Small-footprint hybrid store using FTS5 and persisted JSON vectors."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    page INTEGER,
                    embedding TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    content,
                    tokenize='porter unicode61'
                );
                """
            )
            connection.commit()

    def upsert_document(
        self,
        document_id: str,
        filename: str,
        content_sha256: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding")
        with closing(self._connect()) as connection:
            old_ids = [
                row["id"]
                for row in connection.execute(
                    "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
                )
            ]
            if old_ids:
                connection.executemany(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    [(chunk_id,) for chunk_id in old_ids],
                )
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute(
                """
                INSERT INTO documents(id, filename, content_sha256)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    filename = excluded.filename,
                    content_sha256 = excluded.content_sha256
                """,
                (document_id, filename, content_sha256),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                connection.execute(
                    """
                    INSERT INTO chunks(
                        id, document_id, filename, content, position, page, embedding
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.filename,
                        chunk.text,
                        chunk.position,
                        chunk.page,
                        json.dumps(list(embedding)),
                    ),
                )
                connection.execute(
                    "INSERT INTO chunks_fts(chunk_id, content) VALUES (?, ?)",
                    (chunk.id, chunk.text),
                )
            connection.commit()

    def search(
        self,
        query: str,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        vector_weight: float,
        document_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        with closing(self._connect()) as connection:
            rows = self._candidate_rows(connection, document_ids)
            lexical = self._lexical_scores(connection, query, top_k * 5, document_ids)

        vector_raw = {
            row["id"]: cosine_similarity(query_embedding, json.loads(row["embedding"]))
            for row in rows
        }
        vector_scores = _normalize_scores(vector_raw)
        lexical_scores = _normalize_scores(lexical)

        ranked: list[RetrievedChunk] = []
        for row in rows:
            vector_score = vector_scores.get(row["id"], 0.0)
            lexical_score = lexical_scores.get(row["id"], 0.0)
            score = vector_weight * vector_score + (1.0 - vector_weight) * lexical_score
            if score <= 0:
                continue
            chunk = Chunk(
                id=row["id"],
                document_id=row["document_id"],
                filename=row["filename"],
                text=row["content"],
                position=row["position"],
                page=row["page"],
            )
            ranked.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    vector_score=vector_score,
                    lexical_score=lexical_score,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.chunk.position, item.chunk.id))
        return ranked[:top_k]

    @staticmethod
    def _candidate_rows(
        connection: sqlite3.Connection, document_ids: Sequence[str] | None
    ) -> list[sqlite3.Row]:
        if not document_ids:
            return list(connection.execute("SELECT * FROM chunks"))
        placeholders = ",".join("?" for _ in document_ids)
        return list(
            connection.execute(
                f"SELECT * FROM chunks WHERE document_id IN ({placeholders})",  # noqa: S608
                tuple(document_ids),
            )
        )

    @staticmethod
    def _lexical_scores(
        connection: sqlite3.Connection,
        query: str,
        limit: int,
        document_ids: Sequence[str] | None,
    ) -> dict[str, float]:
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        if not tokens:
            return {}
        match_query = " OR ".join(f'"{token}"' for token in tokens[:30])
        sql = (
            "SELECT c.id, bm25(chunks_fts) AS rank "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.chunk_id "
            "WHERE chunks_fts MATCH ?"
        )
        params: list[object] = [match_query]
        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            sql += f" AND c.document_id IN ({placeholders})"  # noqa: S608
            params.extend(document_ids)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        rows = connection.execute(sql, tuple(params))
        # SQLite FTS5 bm25 returns smaller (usually more negative) values for
        # better matches. Negating it gives the normal "higher is better" shape.
        return {row["id"]: -float(row["rank"]) for row in rows}


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if math.isclose(minimum, maximum):
        return {key: 1.0 for key in scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}
