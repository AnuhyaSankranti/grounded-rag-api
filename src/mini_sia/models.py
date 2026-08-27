from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class TextSection:
    text: str
    page: int | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    filename: str
    text: str
    position: int
    page: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    vector_score: float
    lexical_score: float


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[str] | None = Field(default=None, max_length=50)


class Source(BaseModel):
    citation: int
    document_id: str
    filename: str
    chunk_id: str
    page: int | None
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    version: str
    providers: dict[str, str]


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None


JsonObject = dict[str, Any]

