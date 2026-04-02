"""Embedding cache — avoids recomputing embeddings for seen texts."""

from __future__ import annotations

import json
import logging

from faqih.services.embedding import EmbeddingService
from faqih.services.redis_client import RedisService

logger = logging.getLogger(__name__)

EMB_CACHE_PREFIX = "embcache:"


class EmbeddingCache:
    """
    SHA256-keyed embedding cache in Redis.

    Prevents recomputing embeddings for frequently seen texts.
    """

    def __init__(self, redis: RedisService, embedding: EmbeddingService, ttl_hours: int = 72):
        self._redis = redis
        self._embedding = embedding
        self._ttl = ttl_hours * 3600

    async def get_or_compute(self, text: str) -> list[float]:
        """
        Get embedding from cache or compute and cache it.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        cache_key = f"{EMB_CACHE_PREFIX}{EmbeddingService.text_hash(text)}"

        # Try cache
        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("Embedding cache HIT for text: %s", text[:30])
            return json.loads(cached)

        # Compute
        embedding = self._embedding.encode_single(text)

        # Cache
        await self._redis.set(cache_key, json.dumps(embedding), ttl=self._ttl)
        logger.debug("Embedding cache MISS — computed and cached for: %s", text[:30])

        return embedding

    async def batch_get_or_compute(self, texts: list[str]) -> list[list[float]]:
        """Batch version: get from cache or compute missing ones."""
        results: list[list[float] | None] = [None] * len(texts)
        to_compute: list[tuple[int, str]] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = f"{EMB_CACHE_PREFIX}{EmbeddingService.text_hash(text)}"
            cached = await self._redis.get(cache_key)
            if cached:
                results[i] = json.loads(cached)
            else:
                to_compute.append((i, text))

        # Batch compute missing embeddings
        if to_compute:
            indices, uncached_texts = zip(*to_compute)
            embeddings = self._embedding.encode(list(uncached_texts))

            for idx, embedding in zip(indices, embeddings):
                emb_list = embedding.tolist()
                results[idx] = emb_list
                # Cache each computed embedding
                text = texts[idx]
                cache_key = f"{EMB_CACHE_PREFIX}{EmbeddingService.text_hash(text)}"
                await self._redis.set(cache_key, json.dumps(emb_list), ttl=self._ttl)

        logger.info(
            "Embedding batch: %d cached, %d computed",
            len(texts) - len(to_compute), len(to_compute),
        )
        return results  # type: ignore
