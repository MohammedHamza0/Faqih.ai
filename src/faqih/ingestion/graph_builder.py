"""Knowledge graph builder — LLM entity extraction + Neo4j writer."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from faqih.ingestion.cleaner import ArabicTextCleaner
from faqih.models.enums import EntityType, RelationType
from faqih.models.schemas import Chunk, Entity, Relationship
from faqih.services.llm import LLMClient
from faqih.services.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# ─── Entity Extraction Prompt ───────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """أنت خبير في الفقه الإسلامي. مهمتك استخراج الكيانات والعلاقات من نص فقهي.

أعد النتيجة بصيغة JSON فقط بالشكل التالي:
{
  "entities": [
    {"type": "masala|scholar|book|ayah|hadith|ijma|qiyas|term", "text": "النص", "canonical_id": "نص_مطبّع_فريد"}
  ],
  "relationships": [
    {"source": "canonical_id_المصدر", "target": "canonical_id_الهدف", "type": "AGREES_WITH|DISAGREES_WITH|QUALIFIES|EVIDENCED_BY|EXCEPTION_OF|REFERENCES|DERIVED_FROM", "madhab": "hanafi|maliki|shafii|hanbali|null"}
  ]
}

قواعد:
- canonical_id هو النص بعد إزالة التشكيل والتطبيع (حروف صغيرة، بدون همزات)
- استخرج كل المسائل الفقهية والعلماء والأدلة المذكورة
- حدد العلاقات بين الكيانات بدقة
- لا تضف كيانات غير موجودة في النص
"""

EXTRACTION_USER_PROMPT = """استخرج الكيانات والعلاقات من النص الفقهي التالي:

كتاب: {book_title}
المذهب: {madhab}
الباب: {chapter_path}

النص:
{text}
"""


class GraphBuilder:
    """
    Builds the Fiqh knowledge graph from text chunks.

    Pipeline: Chunk → LLM extraction → Entity resolution → Neo4j write
    """

    def __init__(self, llm: LLMClient, neo4j: Neo4jClient):
        self._llm = llm
        self._neo4j = neo4j
        self._cleaner = ArabicTextCleaner()

    async def setup(self):
        """Initialize Neo4j indexes."""
        await self._neo4j.setup_indexes()

    async def process_chunk(self, chunk: Chunk) -> tuple[list[Entity], list[Relationship]]:
        """
        Extract entities and relationships from a chunk using LLM,
        then write to Neo4j.
        """
        # Step 1: LLM extraction
        entities, relationships = await self._extract_from_llm(chunk)
        logger.info(
            "Extracted %d entities, %d relationships from chunk %s",
            len(entities), len(relationships), chunk.chunk_id[:8],
        )

        # Step 2: Write chunk node to Neo4j
        chunk_node_id = await self._write_chunk_node(chunk)
        chunk.graph_node_id = chunk_node_id

        # Step 3: Write entities and link to chunk
        entity_node_map: dict[str, str] = {}  # canonical_id → neo4j node id
        for entity in entities:
            node_id = await self._write_entity_node(entity)
            entity_node_map[entity.canonical_id or entity.text] = node_id

            # Link entity to chunk
            await self._neo4j.create_relationship(
                source_id=chunk_node_id,
                target_id=node_id,
                rel_type="HAS_CONTENT",
                properties={"chunk_type": chunk.chunk_type},
            )

        # Step 4: Write relationships between entities
        for rel in relationships:
            source_node = entity_node_map.get(rel.source_id)
            target_node = entity_node_map.get(rel.target_id)
            if source_node and target_node:
                await self._neo4j.create_relationship(
                    source_id=source_node,
                    target_id=target_node,
                    rel_type=rel.type,
                    properties={
                        "madhab": rel.madhab or "",
                        "weight": rel.weight,
                    },
                )

        return entities, relationships

    async def _extract_from_llm(
        self, chunk: Chunk
    ) -> tuple[list[Entity], list[Relationship]]:
        """Use LLM to extract entities and relationships from chunk text."""
        chapter_str = " > ".join(chunk.chapter_path) if chunk.chapter_path else "غير محدد"

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": EXTRACTION_USER_PROMPT.format(
                    book_title=chunk.book_title,
                    madhab=chunk.madhab or "غير محدد",
                    chapter_path=chapter_str,
                    text=chunk.text[:3000],  # Limit text length for LLM
                ),
            },
        ]

        try:
            result = await self._llm.complete_json(messages, temperature=0.1)
        except Exception as e:
            logger.error("LLM extraction failed for chunk %s: %s", chunk.chunk_id[:8], e)
            return [], []

        entities = []
        for ent_data in result.get("entities", []):
            try:
                entity = Entity(
                    type=EntityType(ent_data.get("type", "term")),
                    text=ent_data.get("text", ""),
                    canonical_id=ent_data.get("canonical_id", ""),
                    source_chunk_id=chunk.chunk_id,
                )
                entities.append(entity)
            except (ValueError, KeyError) as e:
                logger.warning("Skipping invalid entity: %s", e)

        relationships = []
        for rel_data in result.get("relationships", []):
            try:
                rel = Relationship(
                    source_id=rel_data.get("source", ""),
                    target_id=rel_data.get("target", ""),
                    type=RelationType(rel_data.get("type", "REFERENCES")),
                    madhab=rel_data.get("madhab"),
                )
                relationships.append(rel)
            except (ValueError, KeyError) as e:
                logger.warning("Skipping invalid relationship: %s", e)

        return entities, relationships

    async def _write_chunk_node(self, chunk: Chunk) -> str:
        """Write a Chunk node to Neo4j."""
        properties = {
            "chunk_id": chunk.chunk_id,
            "book_id": chunk.book_id,
            "book_title": chunk.book_title,
            "author": chunk.author,
            "madhab": chunk.madhab or "",
            "chapter_path": chunk.chapter_path,
            "chunk_type": chunk.chunk_type,
            "page_start": chunk.page_start or 0,
            "page_end": chunk.page_end or 0,
            "token_count": chunk.token_count,
        }
        return await self._neo4j.create_node(["Chunk"], properties)

    async def _write_entity_node(self, entity: Entity) -> str:
        """
        Write an Entity node to Neo4j with entity resolution.
        Uses MERGE to deduplicate entities with the same canonical_id.
        """
        canonical = entity.canonical_id or self._normalize_canonical(entity.text)
        labels = ["Entity", entity.type.value.capitalize()]

        node_id = await self._neo4j.merge_node(
            labels=labels,
            match_props={"canonical_id": canonical},
            set_props={
                "text": entity.text,
                "type": entity.type,
                "canonical_id": canonical,
            },
        )
        return node_id

    def _normalize_canonical(self, text: str) -> str:
        """Normalize text for canonical entity matching."""
        cleaned, _ = self._cleaner.clean(text)
        # Further normalization: lowercase, strip extra spaces
        return " ".join(cleaned.lower().split())

    async def process_chunks_batch(
        self, chunks: list[Chunk], batch_size: int = 5
    ) -> tuple[list[Entity], list[Relationship]]:
        """Process multiple chunks, with concurrency control."""
        all_entities = []
        all_relationships = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            results = await asyncio.gather(
                *(self.process_chunk(chunk) for chunk in batch)
            )
            for entities, rels in results:
                all_entities.extend(entities)
                all_relationships.extend(rels)
            logger.info("Processed graph batch %d/%d", i + batch_size, len(chunks))

        return all_entities, all_relationships
