"""Reciprocal Rank Fusion and cross-encoder reranking."""

from __future__ import annotations

import logging
from collections import defaultdict

from faqih.models.schemas import FusedResult, RetrievalResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[FusedResult]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF_score(d) = Σ 1 / (k + rank_i(d))

    Args:
        result_lists: List of ranked result lists from different sources
        k: RRF constant (default 60)

    Returns:
        Fused and sorted list of FusedResult
    """
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            scores[result.chunk_id] += 1.0 / (k + rank)
            sources[result.chunk_id].add(result.source)

    # Build fused results sorted by RRF score
    fused = []
    for chunk_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        fused.append(
            FusedResult(
                chunk_id=chunk_id,
                rrf_score=rrf_score,
                sources=sorted(sources[chunk_id]),
            )
        )

    logger.info(
        "RRF fusion: %d unique chunks from %d sources",
        len(fused), len(result_lists),
    )
    return fused


class Reranker:
    """
    Cross-encoder reranker for Arabic Fiqh text.

    Reranks the top-N results from RRF fusion using a cross-encoder model
    that scores (query, passage) pairs directly.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def load(self):
        """Load the cross-encoder model."""
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model: %s", self._model_name)
        self._model = CrossEncoder(self._model_name)
        logger.info("Reranker model loaded")

    def rerank(
        self,
        query: str,
        results: list[FusedResult],
        texts: dict[str, str],
        top_k: int = 8,
    ) -> list[FusedResult]:
        """
        Rerank fused results using cross-encoder.

        Args:
            query: Original user query
            results: Fused results to rerank
            texts: Mapping of chunk_id → chunk text
            top_k: Number of results to keep after reranking

        Returns:
            Top-K reranked FusedResults
        """
        if not self._model:
            logger.warning("Reranker not loaded, returning RRF results as-is")
            return results[:top_k]

        # Build query-passage pairs
        pairs = []
        valid_results = []
        for r in results:
            text = texts.get(r.chunk_id)
            if text:
                pairs.append([query, text])
                valid_results.append(r)

        if not pairs:
            return results[:top_k]

        # Score with cross-encoder
        scores = self._model.predict(pairs)

        # Assign reranker scores
        for i, score in enumerate(scores):
            valid_results[i].reranker_score = float(score)

        # Sort by reranker score
        valid_results.sort(key=lambda r: r.reranker_score or 0, reverse=True)

        logger.info("Reranked %d results, returning top %d", len(valid_results), top_k)
        return valid_results[:top_k]
