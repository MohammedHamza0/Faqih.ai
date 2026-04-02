"""Application configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Ollama Settings ──────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_fallback_models: str = "llama3.2"

    # ── LLM Settings ────────────────────────────────────────
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # ── Neo4j ────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "faqih_neo4j_2024"

    # ── Qdrant ───────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 8333
    qdrant_grpc_port: int = 8334
    qdrant_collection: str = "fiqh_chunks"

    # ── Elasticsearch ────────────────────────────────────────
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "fiqh_chunks"

    # ── Redis ────────────────────────────────────────────────
    redis_url: str = "redis://localhost:8379/0"

    # ── PostgreSQL ───────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://faqih:faqih_pg_2024@localhost:5432/faqih"

    # ── Embedding Model ─────────────────────────────────────
    embedding_model: str = "aubmindlab/bert-base-arabertv2"
    embedding_dim: int = 768


    # ── Retrieval Settings ──────────────────────────────────
    retrieval_top_k: int = 20
    reranker_top_k: int = 8
    rrf_k: int = 60
    semantic_cache_threshold: float = 0.92

    # ── Chunking Settings ───────────────────────────────────
    chunk_min_tokens: int = 400
    chunk_max_tokens: int = 600
    chunk_overlap_tokens: int = 100

    # ── Conversation Memory ─────────────────────────────────
    conversation_window_size: int = 10
    session_ttl_hours: int = 24

    # ── Celery ──────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── API ─────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    admin_api_key: str = "change-this-to-a-secure-key"

    # ── Paths ───────────────────────────────────────────────
    data_dir: Path = Path("data")
    books_dir: Path = Path("data/books")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
