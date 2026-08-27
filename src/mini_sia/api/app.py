import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from mini_sia.api.routes import router
from mini_sia.config import Settings, get_settings
from mini_sia.logging import configure_logging
from mini_sia.providers import (
    AnswerProvider,
    EmbeddingProvider,
    ExtractiveAnswerProvider,
    HashEmbeddingProvider,
    OpenAIAnswerProvider,
    OpenAIEmbeddingProvider,
)
from mini_sia.services import IngestionService, RagService
from mini_sia.store import SQLiteHybridStore


logger = logging.getLogger(__name__)
REQUESTS = Counter(
    "mini_sia_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "mini_sia_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)


def create_app(
    settings: Settings | None = None,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    answer_provider: AnswerProvider | None = None,
) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = SQLiteHybridStore(resolved.database_path)
        store.initialize()
        embeddings = embedding_provider or _build_embedding_provider(resolved)
        answers = answer_provider or _build_answer_provider(resolved)
        app.state.settings = resolved
        app.state.store = store
        app.state.embedding_provider = embeddings
        app.state.answer_provider = answers
        app.state.ingestion_service = IngestionService(resolved, store, embeddings)
        app.state.rag_service = RagService(resolved, store, embeddings, answers)
        yield

    app = FastAPI(
        title=resolved.app_name,
        version="0.1.0",
        description="Grounded document Q&A with hybrid retrieval, citations, and evals.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.middleware("http")
    async def request_observability(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - started
            logger.exception(
                "unhandled_request_error",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": round(elapsed * 1000, 2),
                },
            )
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )
        elapsed = time.perf_counter() - started
        response.headers["x-request-id"] = request_id
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        REQUESTS.labels(request.method, route_path, response.status_code).inc()
        LATENCY.labels(request.method, route_path).observe(elapsed)
        logger.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": route_path,
                "status_code": response.status_code,
                "latency_ms": round(elapsed * 1000, 2),
            },
        )
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hash":
        return HashEmbeddingProvider()
    return OpenAIEmbeddingProvider(settings.embedding_model)


def _build_answer_provider(settings: Settings) -> AnswerProvider:
    if settings.llm_provider == "extractive":
        return ExtractiveAnswerProvider()
    return OpenAIAnswerProvider(settings.chat_model, settings.max_answer_tokens)

