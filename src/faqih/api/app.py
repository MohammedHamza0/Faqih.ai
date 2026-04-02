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
from faqih.services.llm import LLMClient, create_provider
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

    # ── LLM Client with failover chain ──────────────────
    # Map provider names to their API keys and default models
    provider_config = {
        "google": (settings.google_api_key, settings.google_model),
        "openai": (settings.openai_api_key, settings.openai_model),
        "groq": (settings.groq_api_key, settings.groq_model),
        "openrouter": (settings.openrouter_api_key, settings.openrouter_model),
        "cohere": (settings.cohere_api_key, settings.cohere_model),
    }

    # Build primary provider
    primary_key, primary_model = provider_config.get(
        settings.llm_provider, ("", settings.llm_model)
    )
    primary = create_provider(
        name=settings.llm_provider,
        api_key=primary_key,
        model=settings.llm_model,  # Use the explicit LLM_MODEL for primary
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    # Build fallback chain (only providers with API keys set)
    fallbacks = []
    fallback_names = [
        n.strip() for n in settings.llm_fallback_providers.split(",") if n.strip()
    ]
    for name in fallback_names:
        if name == settings.llm_provider:
            continue  # Skip primary
        api_key, model = provider_config.get(name, ("", ""))
        if api_key:  # Only add providers with configured keys
            fallbacks.append(
                create_provider(
                    name=name,
                    api_key=api_key,
                    model=model,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                )
            )

    llm = LLMClient(primary=primary, fallbacks=fallbacks)

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
