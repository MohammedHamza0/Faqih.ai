"""LLM-based answer generation with streaming."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from faqih.generation.context_assembler import ContextAssembler
from faqih.generation.prompts import format_context, get_system_prompt
from faqih.models.schemas import QueryIntent
from faqih.services.llm import LLMClient

logger = logging.getLogger(__name__)


class FiqhGenerator:
    """
    Generates Fiqh answers using LLM with citation-backed context.

    Supports streaming for SSE delivery to frontend.
    """

    def __init__(self, llm: LLMClient, assembler: ContextAssembler | None = None):
        self._llm = llm
        self._assembler = assembler or ContextAssembler()

    async def generate(
        self,
        query: str,
        chunks_data: list[dict],
        intent: QueryIntent | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate a complete Fiqh answer (non-streaming).

        Args:
            query: User question
            chunks_data: Retrieved chunk payloads
            intent: Detected query intent
            conversation_history: Previous conversation turns

        Returns:
            Complete answer text with citations
        """
        messages = self._build_messages(query, chunks_data, intent, conversation_history)
        return await self._llm.complete(messages)

    async def generate_stream(
        self,
        query: str,
        chunks_data: list[dict],
        intent: QueryIntent | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a Fiqh answer token by token.

        Yields individual tokens/chunks for SSE delivery.
        """
        messages = self._build_messages(query, chunks_data, intent, conversation_history)
        async for token in self._llm.stream(messages):
            yield token

    def _build_messages(
        self,
        query: str,
        chunks_data: list[dict],
        intent: QueryIntent | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build the full message list for the LLM."""
        # Determine prompt variant
        question_type = intent.question_type if intent else "fatwa"
        detail_level = intent.detail_level if intent else "standard"
        is_comparative = question_type == "muqarana"

        # Assemble context
        ordered_chunks = self._assembler.assemble(chunks_data, is_comparative=is_comparative)

        # Build system prompt
        system_prompt = get_system_prompt(question_type, detail_level)

        # Format context with chunks
        context_message = format_context(ordered_chunks, query)

        # Build message list
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if available
        if conversation_history:
            messages.extend(conversation_history)

        # Add current query with context
        messages.append({"role": "user", "content": context_message})

        return messages
