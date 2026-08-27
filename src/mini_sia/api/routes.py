from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from mini_sia import __version__
from mini_sia.api.dependencies import get_ingestion_service, get_rag_service
from mini_sia.loaders import UnsupportedDocumentError
from mini_sia.models import AskRequest, AskResponse, HealthResponse, IngestResponse
from mini_sia.services import DocumentTooLargeError, IngestionService, RagService


router = APIRouter(prefix="/v1")


@router.get("/health", response_model=HealthResponse, tags=["operations"])
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        providers={
            "llm": request.app.state.answer_provider.name,
            "embeddings": request.app.state.embedding_provider.name,
        },
    )


@router.post(
    "/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def ingest_document(
    file: Annotated[UploadFile, File(description="UTF-8 text, Markdown, or PDF")],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> IngestResponse:
    filename = file.filename or "upload.txt"
    try:
        content = await file.read()
        return await service.ingest(filename, content)
    except DocumentTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except UnsupportedDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    finally:
        await file.close()


@router.post("/ask", response_model=AskResponse, tags=["questions"])
async def ask_question(
    payload: AskRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> AskResponse:
    return await service.ask(
        payload.question,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )
