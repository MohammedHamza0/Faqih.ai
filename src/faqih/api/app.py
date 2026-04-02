"""FastAPI application factory with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from faqih.api.routes import admin, chunks, query, sessions
from faqih.config import get_settings
from faqih.models.database import create_db_engine, create_session_factory, init_db
from faqih.services.elasticsearch_client import ElasticsearchService
from faqih.services.embedding import EmbeddingService
from faqih.services.llm import LLMClient
from faqih.services.neo4j_client import Neo4jClient
from faqih.services.qdrant_client import QdrantService
from faqih.services.redis_client import RedisService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service connections lifecycle."""
    settings = get_settings()

    # ── Startup ─────────────────────────────────────────
    # Initialize all service clients
    neo4j = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    await neo4j.connect()

    qdrant = QdrantService(settings.qdrant_host, settings.qdrant_port, settings.qdrant_grpc_port)
    await qdrant.connect()

    es = ElasticsearchService(settings.elasticsearch_url)
    await es.connect()

    redis = RedisService(settings.redis_url)
    await redis.connect()

    # Database
    engine = create_db_engine(settings.database_url)
    await init_db(engine)
    db_session_factory = create_session_factory(engine)

    # Embedding model
    embedding = EmbeddingService(settings.embedding_model)
    embedding.load()

    # LLM client
    api_key = {
        "google": settings.google_api_key,
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
    }.get(settings.llm_provider, settings.google_api_key)
    llm = LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    # Store in app state
    app.state.neo4j = neo4j
    app.state.qdrant = qdrant
    app.state.es = es
    app.state.redis = redis
    app.state.engine = engine
    app.state.db_session_factory = db_session_factory
    app.state.embedding = embedding
    app.state.llm = llm
    app.state.settings = settings

    yield

    # ── Shutdown ────────────────────────────────────────
    await neo4j.close()
    await qdrant.close()
    await es.close()
    await redis.close()
    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Faqih.ai — الفقيه",
        description="Islamic Fiqh Hybrid RAG System with Knowledge Graphs",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])
    app.include_router(query.router, tags=["Query"])
    app.include_router(chunks.router, prefix="/chunks", tags=["Chunks"])
    app.include_router(admin.router, prefix="/admin", tags=["Admin"])

    @app.get("/health")
    async def health():
        return {"status": "healthy", "version": "0.1.0"}

    return app
