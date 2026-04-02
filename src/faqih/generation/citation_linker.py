"""Citation linker — parses and maps citations in generated text."""

from __future__ import annotations

import logging
import re

from faqih.models.schemas import Citation

logger = logging.getLogger(__name__)

# Pattern to match citations in format [الكتاب، الباب] or [الكتاب - الباب]
CITATION_PATTERN = re.compile(r"\[([^\]]+?)[،,\-]\s*([^\]]+?)\]")


class CitationLinker:
    """
    Parses citation markers in LLM-generated text and links
    them to source chunk objects for frontend citation cards.
    """

    def link_citations(
        self,
        generated_text: str,
        chunks_data: list[dict],
    ) -> list[Citation]:
        """
        Find all citation markers in the generated text and link
        them to the source chunks.

        Args:
            generated_text: The LLM-generated answer text
            chunks_data: The chunks that were used as context

        Returns:
            List of Citation objects with chunk references
        """
        citations = []
        seen_markers = set()

        for match in CITATION_PATTERN.finditer(generated_text):
            marker = match.group(0)  # Full match [الكتاب، الباب]
            book_part = match.group(1).strip()
            chapter_part = match.group(2).strip()

            if marker in seen_markers:
                continue
            seen_markers.add(marker)

            # Find the best matching chunk
            best_chunk = self._find_matching_chunk(book_part, chapter_part, chunks_data)

            if best_chunk:
                citation = Citation(
                    marker=marker,
                    chunk_id=best_chunk.get("chunk_id", ""),
                    book_title=best_chunk.get("book_title", book_part),
                    chapter_path=best_chunk.get("chapter_path", [chapter_part]),
                    page_start=best_chunk.get("page_start"),
                    page_end=best_chunk.get("page_end"),
                )
                citations.append(citation)
            else:
                # Create a citation without a chunk link
                citation = Citation(
                    marker=marker,
                    chunk_id="",
                    book_title=book_part,
                    chapter_path=[chapter_part],
                )
                citations.append(citation)

        logger.info("Linked %d citations in generated text", len(citations))
        return citations

    def _find_matching_chunk(
        self,
        book_part: str,
        chapter_part: str,
        chunks_data: list[dict],
    ) -> dict | None:
        """
        Find the chunk that best matches the citation reference.

        Uses fuzzy matching on book title and chapter path.
        """
        best_match = None
        best_score = 0

        for chunk in chunks_data:
            score = 0
            book_title = chunk.get("book_title", "")
            chapter_path = chunk.get("chapter_path", [])

            # Book title match
            if book_part in book_title or book_title in book_part:
                score += 2

            # Chapter path match
            chapter_str = " ".join(chapter_path)
            if chapter_part in chapter_str or chapter_str in chapter_part:
                score += 1

            # Partial word overlap
            book_words = set(book_part.split())
            title_words = set(book_title.split())
            overlap = len(book_words & title_words)
            score += overlap * 0.5

            if score > best_score:
                best_score = score
                best_match = chunk

        return best_match if best_score > 0 else None
