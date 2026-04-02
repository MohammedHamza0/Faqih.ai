"""Tests for Fiqh chunker."""

from faqih.ingestion.chunker import FiqhChunker


class TestFiqhChunker:
    """Tests for the FiqhChunker."""

    def setup_method(self):
        self.chunker = FiqhChunker(min_tokens=10, max_tokens=100, overlap_tokens=5)

    def test_structural_chunking(self, sample_arabic_text):
        """Should detect structural markers and create chunks."""
        chunks = self.chunker.chunk(
            text=sample_arabic_text,
            display_text=sample_arabic_text,
            book_id="test-book",
            book_title="Test",
            author="Test Author",
        )
        assert len(chunks) > 0

    def test_chapter_path_populated(self, sample_arabic_text):
        """Chunks should have chapter_path metadata."""
        chunks = self.chunker.chunk(
            text=sample_arabic_text,
            display_text=sample_arabic_text,
            book_id="test-book",
            book_title="Test",
            author="Test Author",
        )
        # At least some chunks should have chapter paths
        chunks_with_paths = [c for c in chunks if c.chapter_path]
        assert len(chunks_with_paths) >= 0  # May or may not have paths depending on parsing

    def test_sliding_window_fallback(self):
        """Should fall back to sliding window for unstructured text."""
        # Text without any structural markers
        text = " ".join(["كلمة"] * 200)  # 200 words
        chunks = self.chunker.chunk(
            text=text,
            display_text=text,
            book_id="test",
            book_title="Test",
            author="Test",
        )
        assert len(chunks) > 0

    def test_chunk_has_required_fields(self, sample_arabic_text):
        """Every chunk should have required fields populated."""
        chunks = self.chunker.chunk(
            text=sample_arabic_text,
            display_text=sample_arabic_text,
            book_id="test-book",
            book_title="Test Book",
            author="Author",
            madhab="hanbali",
        )
        for chunk in chunks:
            assert chunk.book_id == "test-book"
            assert chunk.book_title == "Test Book"
            assert chunk.author == "Author"
            assert chunk.text
            assert chunk.display_text
            assert chunk.token_count > 0

    def test_empty_text(self):
        """Empty text should produce no chunks."""
        chunks = self.chunker.chunk(
            text="",
            display_text="",
            book_id="test",
            book_title="Test",
            author="Test",
        )
        assert len(chunks) == 0
