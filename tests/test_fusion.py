"""Tests for RRF fusion algorithm."""

from faqih.models.schemas import RetrievalResult
from faqih.retrieval.fusion import reciprocal_rank_fusion


class TestRRFFusion:
    """Test Reciprocal Rank Fusion."""

    def test_single_source(self):
        """RRF with a single source should preserve ranking."""
        results = [
            RetrievalResult(chunk_id="a", score=0.9, source="vector"),
            RetrievalResult(chunk_id="b", score=0.8, source="vector"),
            RetrievalResult(chunk_id="c", score=0.7, source="vector"),
        ]
        fused = reciprocal_rank_fusion([results])
        assert fused[0].chunk_id == "a"
        assert fused[1].chunk_id == "b"
        assert fused[2].chunk_id == "c"

    def test_two_sources_same_docs(self):
        """Documents appearing in multiple sources should rank higher."""
        vector_results = [
            RetrievalResult(chunk_id="a", score=0.9, source="vector"),
            RetrievalResult(chunk_id="b", score=0.8, source="vector"),
        ]
        bm25_results = [
            RetrievalResult(chunk_id="b", score=10.0, source="bm25"),
            RetrievalResult(chunk_id="c", score=9.0, source="bm25"),
        ]

        fused = reciprocal_rank_fusion([vector_results, bm25_results])
        # "b" appears in both → should have highest RRF score
        assert fused[0].chunk_id == "b"
        assert "vector" in fused[0].sources
        assert "bm25" in fused[0].sources

    def test_empty_sources(self):
        """Empty result lists should return empty fusion."""
        fused = reciprocal_rank_fusion([[], []])
        assert len(fused) == 0

    def test_rrf_scores_decrease(self):
        """RRF scores should be monotonically decreasing."""
        results = [RetrievalResult(chunk_id=f"c{i}", score=1.0, source="vector") for i in range(10)]
        fused = reciprocal_rank_fusion([results])
        for i in range(len(fused) - 1):
            assert fused[i].rrf_score >= fused[i + 1].rrf_score

    def test_k_parameter(self):
        """Different k values should produce different scores."""
        results = [
            RetrievalResult(chunk_id="a", score=0.9, source="vector"),
        ]
        fused_60 = reciprocal_rank_fusion([results], k=60)
        fused_10 = reciprocal_rank_fusion([results], k=10)
        # Smaller k → higher scores
        assert fused_10[0].rrf_score > fused_60[0].rrf_score
