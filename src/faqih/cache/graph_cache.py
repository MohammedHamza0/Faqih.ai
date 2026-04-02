"""Graph result cache — caches Cypher query results."""

from __future__ import annotations

import hashlib
import json
import logging

from faqih.services.redis_client import RedisService

logger = logging.getLogger(__name__)

GRAPH_CACHE_PREFIX = "graphcache:"


class GraphCache:
    """
    Caches Neo4j Cypher query results.

    Key is built from entity_ids + traversal_depth.
    TTL defaults to 1 hour.
    """

    def __init__(self, redis: RedisService, ttl_seconds: int = 3600):
        self._redis = redis
        self._ttl = ttl_seconds

    def _build_key(self, entity_ids: list[str], depth: int) -> str:
        """Build a deterministic cache key from entities and depth."""
        sorted_ids = sorted(entity_ids)
        raw = f"{','.join(sorted_ids)}:depth={depth}"
        return f"{GRAPH_CACHE_PREFIX}{hashlib.sha256(raw.encode()).hexdigest()[:24]}"

    async def get(self, entity_ids: list[str], depth: int) -> list[dict] | None:
        """Get cached graph traversal results."""
        key = self._build_key(entity_ids, depth)
        cached = await self._redis.get(key)
        if cached:
            logger.debug("Graph cache HIT")
            return json.loads(cached)
        return None

    async def set(self, entity_ids: list[str], depth: int, results: list[dict]):
        """Cache graph traversal results."""
        key = self._build_key(entity_ids, depth)
        await self._redis.set(key, json.dumps(results, ensure_ascii=False), ttl=self._ttl)
        logger.debug("Graph cache SET (%d results)", len(results))
