"""Vector search via Qdrant."""

from __future__ import annotations

import logging

from qdrant_client import models

from faqih.models.schemas import RetrievalResult
from faqih.services.qdrant_client import QdrantService

logger = logging.getLogger(__name__)


class VectorSearcher:
    """Performs dense vector similarity search in Qdrant."""

    def __init__(self, qdrant: QdrantService, collection_name: str = "fiqh_chunks"):
        self._qdrant = qdrant
        self._collection = collection_name

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        madhab_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Search for similar chunks using dense vector.

        Args:
            query_vector: Query embedding (typically HyDE embedding)
            top_k: Number of results
            madhab_filter: Optional madhab filter

        Returns:
            List of RetrievalResult with scores
        """
        # Build filter
        query_filter = None
        if madhab_filter and madhab_filter != "all":
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="madhab",
                        match=models.MatchValue(value=madhab_filter),
                    )
                ]
            )

        scored_points = await self._qdrant.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

        results = []
        for point in scored_points:
            results.append(
                RetrievalResult(
                    chunk_id=point.payload.get("chunk_id", str(point.id)),
                    score=point.score,
                    source="vector",
                )
            )

        logger.info("Vector search returned %d results", len(results))
        return results
