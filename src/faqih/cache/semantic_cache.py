"""Semantic cache — avoids redundant retrieval for similar queries."""

from __future__ import annotations

import logging
import time

import numpy as np

from faqih.services.embedding import EmbeddingService
from faqih.services.redis_client import RedisService

logger = logging.getLogger(__name__)

CACHE_PREFIX = "semcache:"
CACHE_INDEX_KEY = "semcache:index"


class SemanticCache:
    """
    Caches query results by semantic similarity.

    Before any retrieval, the query is embedded and compared against
    cached queries. If cosine similarity > threshold, returns cached answer.
    """

    def __init__(
        self,
        redis: RedisService,
        embedding: EmbeddingService,
        threshold: float = 0.92,
        ttl_hours: int = 24,
    ):
        self._redis = redis
        self._embedding = embedding
        self._threshold = threshold
        self._ttl = ttl_hours * 3600

    async def get(self, query: str) -> dict | None:
        """
        Check if a semantically similar query has been cached.

        Returns cached response dict or None.
        """
        query_embedding = self._embedding.encode_single(query)

        # Get all cached query embeddings
        cached_keys = await self._redis.lrange(CACHE_INDEX_KEY, 0, -1)
        if not cached_keys:
            return None

        for key in cached_keys:
            cached_data = await self._redis.get_json(f"{CACHE_PREFIX}{key}")
            if not cached_data:
                continue

            cached_embedding = cached_data.get("embedding")
            if not cached_embedding:
                continue

            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity >= self._threshold:
                logger.info("Semantic cache HIT (similarity=%.3f)", similarity)
                return cached_data.get("response")

        return None

    async def set(self, query: str, response: dict, query_embedding: list[float] | None = None):
        """Cache a query-response pair with its embedding."""
        if query_embedding is None:
            query_embedding = self._embedding.encode_single(query)
        cache_key = EmbeddingService.text_hash(query)

        cache_data = {
            "query": query,
            "embedding": query_embedding,
            "response": response,
            "timestamp": time.time(),
        }

        await self._redis.set_json(
            f"{CACHE_PREFIX}{cache_key}",
            cache_data,
            ttl=self._ttl,
        )

        # Add to index
        await self._redis.lpush(CACHE_INDEX_KEY, cache_key)
        # Keep index bounded
        await self._redis.ltrim(CACHE_INDEX_KEY, 0, 999)
        await self._redis.expire(CACHE_INDEX_KEY, self._ttl)

        logger.info("Cached response for query: %s", query[:50])

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        dot = np.dot(a_np, b_np)
        norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
        return float(dot / norm) if norm > 0 else 0.0
