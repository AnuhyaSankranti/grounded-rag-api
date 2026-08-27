from fastapi import Request

from mini_sia.services import IngestionService, RagService


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service

