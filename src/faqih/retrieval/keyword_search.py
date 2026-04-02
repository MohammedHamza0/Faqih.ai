"""BM25 keyword search via Elasticsearch."""

from __future__ import annotations

import logging

from faqih.models.schemas import RetrievalResult
from faqih.services.elasticsearch_client import ElasticsearchService

logger = logging.getLogger(__name__)


class KeywordSearcher:
    """BM25-based keyword search using Elasticsearch with Fiqh Arabic analyzer."""

    def __init__(self, es: ElasticsearchService, index_name: str = "fiqh_chunks"):
        self._es = es
        self._index_name = index_name

    async def search(
        self,
        query_text: str,
        top_k: int = 20,
        madhab_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Perform BM25 keyword search.

        Args:
            query_text: The search query (will use fiqh_arabic analyzer)
            top_k: Number of results
            madhab_filter: Optional madhab filter

        Returns:
            List of RetrievalResult with BM25 scores
        """
        filters = {}
        if madhab_filter and madhab_filter != "all":
            filters["madhab"] = madhab_filter

        hits = await self._es.search(
            index_name=self._index_name,
            query_text=query_text,
            size=top_k,
            filters=filters if filters else None,
        )

        results = []
        for hit in hits:
            results.append(
                RetrievalResult(
                    chunk_id=hit["chunk_id"],
                    score=hit["score"],
                    source="bm25",
                )
            )

        logger.info("BM25 search returned %d results", len(results))
        return results
