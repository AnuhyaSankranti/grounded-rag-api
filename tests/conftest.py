from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mini_sia.api.app import create_app
from mini_sia.config import Settings
from mini_sia.providers import ExtractiveAnswerProvider, HashEmbeddingProvider


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="test",
        database_path=tmp_path / "test.db",
        llm_provider="extractive",
        embedding_provider="hash",
        chunk_size_words=50,
        chunk_overlap_words=10,
        top_k=3,
    )


@pytest.fixture
def client(settings: Settings):
    app = create_app(
        settings,
        embedding_provider=HashEmbeddingProvider(),
        answer_provider=ExtractiveAnswerProvider(),
    )
    with TestClient(app) as test_client:
        yield test_client

