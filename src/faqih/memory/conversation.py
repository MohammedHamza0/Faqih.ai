"""Redis-backed conversation memory with entity tracking."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from faqih.models.schemas import ConversationTurn, Session
from faqih.services.redis_client import RedisService

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation state in Redis.

    Features:
    - Sliding window of last N turns
    - Entity tracking across session
    - Reference resolution ("هذا الحكم", "الرأي الثاني")
    """

    # Keys
    SESSION_KEY = "session:{session_id}"
    TURNS_KEY = "session:{session_id}:turns"
    ENTITIES_KEY = "session:{session_id}:entities"
    REFS_KEY = "session:{session_id}:refs"

    def __init__(self, redis: RedisService, window_size: int = 10, ttl_hours: int = 24):
        self._redis = redis
        self._window_size = window_size
        self._ttl = ttl_hours * 3600

    async def create_session(self) -> Session:
        """Create a new conversation session."""
        session = Session()
        await self._redis.set_json(
            self.SESSION_KEY.format(session_id=session.session_id),
            session.model_dump(mode="json"),
            ttl=self._ttl,
        )
        logger.info("Created session: %s", session.session_id)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        data = await self._redis.get_json(
            self.SESSION_KEY.format(session_id=session_id)
        )
        if data:
            return Session(**data)
        return None

    async def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        entities: list[str] | None = None,
        chunk_ids: list[str] | None = None,
    ):
        """Add a conversation turn and update entity tracking."""
        turn = ConversationTurn(
            role=role,
            content=content,
            entities_mentioned=entities or [],
            chunk_ids_cited=chunk_ids or [],
        )

        turns_key = self.TURNS_KEY.format(session_id=session_id)

        # Push turn to list (most recent first)
        await self._redis.lpush(turns_key, turn.model_dump_json())

        # Trim to window size
        await self._redis.ltrim(turns_key, 0, self._window_size - 1)
        await self._redis.expire(turns_key, self._ttl)

        # Update entity history
        if entities:
            entities_key = self.ENTITIES_KEY.format(session_id=session_id)
            for entity in entities:
                await self._redis.hset(entities_key, entity, datetime.utcnow().isoformat())
            await self._redis.expire(entities_key, self._ttl)

    async def get_recent_turns(self, session_id: str) -> list[ConversationTurn]:
        """Get recent turns (most recent first)."""
        turns_key = self.TURNS_KEY.format(session_id=session_id)
        raw_turns = await self._redis.lrange(turns_key, 0, self._window_size - 1)

        turns = []
        for raw in raw_turns:
            try:
                turns.append(ConversationTurn(**json.loads(raw)))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to parse turn: %s", e)

        # Return in chronological order
        return list(reversed(turns))

    async def get_entity_history(self, session_id: str) -> list[str]:
        """Get all entities mentioned in this session."""
        entities_key = self.ENTITIES_KEY.format(session_id=session_id)
        entity_map = await self._redis.hgetall(entities_key)
        return list(entity_map.keys())

    async def resolve_reference(
        self, session_id: str, reference: str
    ) -> str | None:
        """
        Resolve an anaphoric reference like "هذا الحكم" or "الرأي الثاني".

        Looks up the reference in the session's resolved_references map.
        """
        refs_key = self.REFS_KEY.format(session_id=session_id)
        return await self._redis.hget(refs_key, reference)

    async def store_reference(
        self, session_id: str, reference: str, resolved_to: str
    ):
        """Store a resolved reference mapping."""
        refs_key = self.REFS_KEY.format(session_id=session_id)
        await self._redis.hset(refs_key, reference, resolved_to)
        await self._redis.expire(refs_key, self._ttl)

    async def build_context_from_history(self, session_id: str) -> list[dict[str, str]]:
        """
        Build LLM-compatible message history from recent turns.

        Returns list of {"role": "user"|"assistant", "content": "..."}
        """
        turns = await self.get_recent_turns(session_id)
        return [{"role": t.role, "content": t.content} for t in turns]
