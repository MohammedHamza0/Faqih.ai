"""Dependency injection for FastAPI routes."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from faqih.cache.semantic_cache import SemanticCache
from faqih.generation.context_assembler import ContextAssembler
from faqih.generation.generator import FiqhGenerator
from faqih.memory.conversation import ConversationMemory
from faqih.retrieval.engine import RetrievalEngine
from faqih.retrieval.graph_search import GraphSearcher
from faqih.retrieval.intent_detector import IntentDetector
from faqih.retrieval.keyword_search import KeywordSearcher
from faqih.retrieval.query_rewriter import QueryRewriter
from faqih.retrieval.vector_search import VectorSearcher
from faqih.services.elasticsearch_client import ElasticsearchService
from faqih.services.embedding import EmbeddingService
from faqih.services.llm import LLMClient
from faqih.services.neo4j_client import Neo4jClient
from faqih.services.qdrant_client import QdrantService
from faqih.services.redis_client import RedisService

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ─── Service Dependencies ──────────────────────────────────


def get_neo4j(request: Request) -> Neo4jClient:
    return request.app.state.neo4j


def get_qdrant(request: Request) -> QdrantService:
    return request.app.state.qdrant


def get_es(request: Request) -> ElasticsearchService:
    return request.app.state.es


def get_redis(request: Request) -> RedisService:
    return request.app.state.redis


def get_embedding(request: Request) -> EmbeddingService:
    return request.app.state.embedding


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


# ─── Component Dependencies ────────────────────────────────


def get_conversation_memory(
    redis: RedisService = Depends(get_redis),
    request: Request = None,
) -> ConversationMemory:
    settings = request.app.state.settings
    return ConversationMemory(redis, window_size=settings.conversation_window_size)


def get_query_rewriter(
    llm: LLMClient = Depends(get_llm),
    embedding: EmbeddingService = Depends(get_embedding),
) -> QueryRewriter:
    return QueryRewriter(llm, embedding)


def get_intent_detector(
    llm: LLMClient = Depends(get_llm),
) -> IntentDetector:
    return IntentDetector(llm)


def get_retrieval_engine(
    request: Request = None,
    qdrant: QdrantService = Depends(get_qdrant),
    neo4j: Neo4jClient = Depends(get_neo4j),
    es: ElasticsearchService = Depends(get_es),
) -> RetrievalEngine:
    settings = request.app.state.settings
    return RetrievalEngine(
        vector_searcher=VectorSearcher(qdrant, settings.qdrant_collection),
        graph_searcher=GraphSearcher(neo4j),
        keyword_searcher=KeywordSearcher(es, settings.elasticsearch_index),
        rrf_k=settings.rrf_k,
        retrieval_top_k=settings.retrieval_top_k,
        reranker_top_k=settings.reranker_top_k,
    )


def get_generator(
    llm: LLMClient = Depends(get_llm),
) -> FiqhGenerator:
    return FiqhGenerator(llm, ContextAssembler())


def get_semantic_cache(
    redis: RedisService = Depends(get_redis),
    embedding: EmbeddingService = Depends(get_embedding),
    request: Request = None,
) -> SemanticCache:
    settings = request.app.state.settings
    return SemanticCache(redis, embedding, threshold=settings.semantic_cache_threshold)


# ─── Auth Dependencies ─────────────────────────────────────


async def verify_admin_key(
    request: Request,
    api_key: str = Security(api_key_header),
):
    """Verify admin API key for protected endpoints."""
    settings = request.app.state.settings
    if api_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
