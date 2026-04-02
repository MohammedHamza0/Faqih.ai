"""Hybrid retrieval engine — orchestrates parallel search + fusion."""

from __future__ import annotations

import asyncio
import logging

from faqih.models.schemas import FusedResult, QueryIntent, RetrievalResult
from faqih.retrieval.fusion import Reranker, reciprocal_rank_fusion
from faqih.retrieval.graph_search import GraphSearcher
from faqih.retrieval.keyword_search import KeywordSearcher
from faqih.retrieval.vector_search import VectorSearcher

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Orchestrates hybrid retrieval across three sources in parallel:
    1. Vector search (Qdrant, HyDE embedding)
    2. Graph traversal (Neo4j, entity-based)
    3. BM25 keyword search (Elasticsearch)

    Results are fused with RRF and optionally reranked.
    """

    def __init__(
        self,
        vector_searcher: VectorSearcher,
        graph_searcher: GraphSearcher,
        keyword_searcher: KeywordSearcher,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        retrieval_top_k: int = 20,
        reranker_top_k: int = 8,
    ):
        self._vector = vector_searcher
        self._graph = graph_searcher
        self._keyword = keyword_searcher
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._retrieval_top_k = retrieval_top_k
        self._reranker_top_k = reranker_top_k

    async def retrieve(
        self,
        query_text: str,
        hyde_embedding: list[float],
        query_entities: list[str],
        intent: QueryIntent | None = None,
        expanded_queries: list[str] | None = None,
    ) -> list[FusedResult]:
        """
        Execute the full hybrid retrieval pipeline.

        Args:
            query_text: Original user query
            hyde_embedding: HyDE query vector
            query_entities: Entities extracted from query + conversation history
            intent: Detected query intent with madhab preference
            expanded_queries: Additional query formulations

        Returns:
            Top-K fused and optionally reranked results
        """
        madhab = None
        if intent and intent.madhab_preference != "all":
            madhab = str(intent.madhab_preference)

        # ── Run all three searches in parallel ──────────────
        vector_task = self._vector.search(
            query_vector=hyde_embedding,
            top_k=self._retrieval_top_k,
            madhab_filter=madhab,
        )

        graph_task = self._graph.search(
            query_entities=query_entities,
            max_depth=2,
            madhab_filter=madhab,
        )

        keyword_task = self._keyword.search(
            query_text=query_text,
            top_k=self._retrieval_top_k,
            madhab_filter=madhab,
        )

        vector_results, graph_results, keyword_results = await asyncio.gather(
            vector_task, graph_task, keyword_task
        )

        logger.info(
            "Retrieval: vector=%d, graph=%d, bm25=%d",
            len(vector_results), len(graph_results), len(keyword_results),
        )

        # ── Also search with expanded queries if available ──
        if expanded_queries:
            expansion_results = []
            for eq in expanded_queries[1:]:  # Skip original (already searched)
                eq_results = await self._keyword.search(eq, top_k=10, madhab_filter=madhab)
                expansion_results.extend(eq_results)
            keyword_results.extend(expansion_results)

        # ── RRF Fusion ──────────────────────────────────────
        all_result_lists = [vector_results, graph_results, keyword_results]
        fused = reciprocal_rank_fusion(all_result_lists, k=self._rrf_k)

        # ── Optional Reranking ──────────────────────────────
        if self._reranker:
            # Get top-30 for reranking
            top_30 = fused[:30]
            # TODO: Fetch chunk texts from Qdrant/ES for reranking
            # For now, return RRF results
            return top_30[: self._reranker_top_k]

        return fused[: self._reranker_top_k]
