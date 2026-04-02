"""Redis client wrapper for caching and conversation memory."""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisService:
    """Async Redis client for caching and session storage."""

    def __init__(self, url: str):
        self._url = url
        self._client: aioredis.Redis | None = None

    async def connect(self):
        """Initialize Redis connection pool."""
        self._client = aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        await self._client.ping()
        logger.info("Connected to Redis at %s", self._url)

    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            logger.info("Redis connection closed")

    @property
    def client(self) -> aioredis.Redis:
        if not self._client:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client

    # ── Key-Value Operations ────────────────────────────────

    async def get(self, key: str) -> str | None:
        """Get a value by key."""
        return await self.client.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None):
        """Set a value with optional TTL in seconds."""
        if ttl:
            await self.client.setex(key, ttl, value)
        else:
            await self.client.set(key, value)

    async def delete(self, key: str):
        """Delete a key."""
        await self.client.delete(key)

    # ── JSON Operations ─────────────────────────────────────

    async def get_json(self, key: str) -> Any | None:
        """Get and deserialize JSON value."""
        data = await self.get(key)
        return json.loads(data) if data else None

    async def set_json(self, key: str, value: Any, ttl: int | None = None):
        """Serialize and store JSON value."""
        await self.set(key, json.dumps(value, ensure_ascii=False), ttl)

    # ── List Operations (for conversation history) ──────────

    async def lpush(self, key: str, *values: str):
        """Push values to the left of a list."""
        await self.client.lpush(key, *values)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        """Get a range from a list."""
        return await self.client.lrange(key, start, end)

    async def ltrim(self, key: str, start: int, end: int):
        """Trim a list to the specified range."""
        await self.client.ltrim(key, start, end)

    # ── Hash Operations (for entity tracking) ───────────────

    async def hset(self, key: str, field: str, value: str):
        """Set a hash field."""
        await self.client.hset(key, field, value)

    async def hget(self, key: str, field: str) -> str | None:
        """Get a hash field."""
        return await self.client.hget(key, field)

    async def hgetall(self, key: str) -> dict[str, str]:
        """Get all fields in a hash."""
        return await self.client.hgetall(key)

    # ── TTL Management ──────────────────────────────────────

    async def expire(self, key: str, ttl: int):
        """Set TTL on a key."""
        await self.client.expire(key, ttl)

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return bool(await self.client.exists(key))
