"""Context assembler — orders chunks by Fiqh relevance type."""

from __future__ import annotations

import logging

from faqih.models.enums import ChunkType

logger = logging.getLogger(__name__)

# Priority order for chunk types in the assembled context
CHUNK_TYPE_ORDER = {
    ChunkType.HUKM: 0,       # Ruling first
    ChunkType.DALIL: 1,      # Evidence second
    ChunkType.KHILAF: 2,     # Disagreement third
    ChunkType.TALIL: 3,      # Reasoning fourth
    ChunkType.SHART: 4,      # Conditions fifth
    ChunkType.ISTITHNA: 5,   # Exceptions sixth
    ChunkType.GENERAL: 6,    # General last
}


class ContextAssembler:
    """
    Assembles and orders retrieved chunks for optimal LLM consumption.

    Ordering logic:
    1. Hukm (ruling) — so the model sees the core judgment first
    2. Dalil (evidence) — supporting proofs
    3. Khilaf (disagreement) — for comparative context
    4. Talil (reasoning) — scholarly justification
    5. Shart (conditions) — prerequisites and conditions
    6. Istithna (exceptions) — edge cases
    """

    def __init__(self, max_chunks: int = 8):
        self._max_chunks = max_chunks

    def assemble(
        self,
        chunks_data: list[dict],
        is_comparative: bool = False,
    ) -> list[dict]:
        """
        Assemble and order chunks for context injection.

        Args:
            chunks_data: List of chunk payload dicts from retrieval
            is_comparative: If True, prioritize khilaf chunks

        Returns:
            Ordered list of chunk dicts (max self._max_chunks)
        """
        if not chunks_data:
            return []

        # Adjust priority for comparative questions
        order = dict(CHUNK_TYPE_ORDER)
        if is_comparative:
            order[ChunkType.KHILAF] = 0  # Promote disagreement to top
            order[ChunkType.HUKM] = 1

        # Sort by chunk type priority, then by original retrieval score
        def sort_key(chunk: dict) -> tuple:
            chunk_type = chunk.get("chunk_type", ChunkType.GENERAL)
            try:
                type_priority = order.get(ChunkType(chunk_type), 6)
            except ValueError:
                type_priority = 6
            return (type_priority,)

        sorted_chunks = sorted(chunks_data, key=sort_key)

        # Deduplicate by content similarity (simple exact text check)
        seen_texts = set()
        unique_chunks = []
        for chunk in sorted_chunks:
            text_key = chunk.get("text", "")[:200]  # Compare first 200 chars
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_chunks.append(chunk)

        result = unique_chunks[: self._max_chunks]
        logger.info(
            "Assembled %d chunks from %d retrieved (comparative=%s)",
            len(result), len(chunks_data), is_comparative,
        )
        return result
