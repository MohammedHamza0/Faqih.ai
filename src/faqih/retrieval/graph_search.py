"""Knowledge graph traversal via Neo4j Cypher queries."""

from __future__ import annotations

import logging

from faqih.models.schemas import RetrievalResult
from faqih.services.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphSearcher:
    """
    Performs graph traversal in Neo4j to find related chunks.

    Strategy:
    1. Find Masala nodes matching query entities
    2. Traverse edges (QUALIFIES, EVIDENCED_BY, DISAGREES_WITH) to depth 2-3
    3. Collect all related chunk_ids
    4. Score by connection strength (PageRank-like)
    """

    def __init__(self, neo4j: Neo4jClient):
        self._neo4j = neo4j

    async def search(
        self,
        query_entities: list[str],
        max_depth: int = 2,
        madhab_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Find chunks related to the given entities via graph traversal.

        Args:
            query_entities: List of entity canonical IDs or text to match
            max_depth: Maximum traversal depth
            madhab_filter: Optional madhab filter

        Returns:
            List of RetrievalResult with graph-based scores
        """
        if not query_entities:
            return []

        # Clamp depth to safe range (Cypher doesn't support parameterized depth)
        max_depth = max(1, min(max_depth, 5))

        # Step 1: Find matching entity nodes
        entity_match_query = """
        UNWIND $entities AS entity_text
        MATCH (e:Entity)
        WHERE e.canonical_id CONTAINS entity_text
           OR e.text CONTAINS entity_text
        RETURN DISTINCT elementId(e) AS entity_id, e.text AS text
        LIMIT 20
        """
        entity_nodes = await self._neo4j.run_query(
            entity_match_query, {"entities": query_entities}
        )

        if not entity_nodes:
            logger.info("No matching entities found in graph")
            return []

        entity_ids = [n["entity_id"] for n in entity_nodes]

        # Step 2: Traverse relationships to find related chunks
        madhab_clause = ""
        params: dict = {"entity_ids": entity_ids, "max_depth": max_depth}

        if madhab_filter and madhab_filter != "all":
            madhab_clause = "AND c.madhab = $madhab"
            params["madhab"] = madhab_filter

        traversal_query = f"""
        UNWIND $entity_ids AS eid
        MATCH (e) WHERE elementId(e) = eid
        MATCH path = (e)-[*1..{max_depth}]-(related:Entity)
        WITH related, length(path) AS distance
        MATCH (c:Chunk)-[:HAS_CONTENT]->(related)
        WHERE c.chunk_id IS NOT NULL {madhab_clause}
        WITH c.chunk_id AS chunk_id,
             min(distance) AS min_distance,
             count(DISTINCT related) AS connection_count
        RETURN chunk_id,
               connection_count,
               min_distance,
               toFloat(connection_count) / (min_distance + 1) AS relevance_score
        ORDER BY relevance_score DESC
        LIMIT 20
        """

        results = await self._neo4j.run_query(traversal_query, params)

        retrieval_results = []
        for r in results:
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=r["chunk_id"],
                    score=r["relevance_score"],
                    source="graph",
                )
            )

        logger.info("Graph search returned %d results", len(retrieval_results))
        return retrieval_results

    async def find_entity_chunks(
        self, entity_text: str
    ) -> list[str]:
        """Find chunk IDs directly connected to a specific entity."""
        query = """
        MATCH (e:Entity)-[:HAS_CONTENT]-(c:Chunk)
        WHERE e.text CONTAINS $text OR e.canonical_id CONTAINS $text
        RETURN DISTINCT c.chunk_id AS chunk_id
        LIMIT 10
        """
        results = await self._neo4j.run_query(query, {"text": entity_text})
        return [r["chunk_id"] for r in results]
