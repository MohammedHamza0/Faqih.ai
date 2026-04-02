"""Async prefetch via Celery for related masail."""

from __future__ import annotations

import logging

from celery import Celery

from faqih.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ─── Celery App ─────────────────────────────────────────────

celery_app = Celery(
    "faqih",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(name="faqih.prefetch_related")
def prefetch_related_masail(entity_ids: list[str], depth: int = 2):
    """
    Background task to pre-fetch and cache related masail.

    Triggered after a successful query to warm the cache
    for likely follow-up questions.
    """
    import asyncio
    from faqih.services.neo4j_client import Neo4jClient
    from faqih.cache.graph_cache import GraphCache
    from faqih.services.redis_client import RedisService

    async def _prefetch():
        s = get_settings()

        neo4j = Neo4jClient(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
        await neo4j.connect()

        redis = RedisService(s.redis_url)
        await redis.connect()

        cache = GraphCache(redis)

        try:
            # Clamp depth to safe range (Cypher doesn't support parameterized depth)
            depth = max(1, min(depth, 5))

            # Check if already cached
            cached = await cache.get(entity_ids, depth)
            if cached:
                logger.info("Prefetch: already cached for %d entities", len(entity_ids))
                return

            # Fetch from Neo4j
            query = """
            UNWIND $entity_ids AS eid
            MATCH (e:Entity {canonical_id: eid})
            MATCH path = (e)-[*1..%d]-(related:Entity)
            WITH related, length(path) AS distance
            MATCH (c:Chunk)-[:HAS_CONTENT]->(related)
            RETURN DISTINCT c.chunk_id AS chunk_id,
                   related.canonical_id AS entity_id,
                   distance
            ORDER BY distance
            LIMIT 50
            """ % depth

            results = await neo4j.run_query(query, {"entity_ids": entity_ids})

            # Cache results
            await cache.set(entity_ids, depth, results)
            logger.info("Prefetched %d results for %d entities", len(results), len(entity_ids))

        finally:
            await neo4j.close()
            await redis.close()

    asyncio.run(_prefetch())
