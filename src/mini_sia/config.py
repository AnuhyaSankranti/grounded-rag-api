from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from MINI_SIA_* environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MINI_SIA_",
        extra="ignore",
    )

    app_name: str = "Mini SIA"
    app_env: Literal["development", "test", "production"] = "development"
    database_path: Path = Path("data/mini_sia.db")
    llm_provider: Literal["openai", "extractive"] = "openai"
    embedding_provider: Literal["openai", "hash"] = "openai"
    chat_model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = Field(default=5, ge=1, le=20)
    vector_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    chunk_size_words: int = Field(default=220, ge=50, le=1000)
    chunk_overlap_words: int = Field(default=40, ge=0, le=300)
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    max_answer_tokens: int = Field(default=500, ge=50, le=4000)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
