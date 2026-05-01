"""Intelligent Fiqh-aware text chunker."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from faqih.models.schemas import Chunk

logger = logging.getLogger(__name__)


# ─── Structural Markers ────────────────────────────────────

# Ordered by hierarchy depth (most general → most specific)
STRUCTURAL_MARKERS = [
    ("كتاب", 0),  # Book / Major division
    ("باب", 1),  # Chapter
    ("فصل", 2),  # Section
    ("مسألة", 3),  # Legal issue
    ("فرع", 4),  # Sub-issue / Branch
]

# Compiled patterns for structural detection
STRUCTURAL_PATTERNS = [
    (re.compile(rf"^({marker})\s*[:：]?\s*(.+)", re.MULTILINE), marker, level)
    for marker, level in STRUCTURAL_MARKERS
]

# Sentence boundary pattern for Arabic text
SENTENCE_BOUNDARY = re.compile(r"(?<=[.。؟!\n])\s+")

# Page marker pattern from extractor
PAGE_MARKER_PATTERN = re.compile(r"<<PAGE:(\d+)>>")


@dataclass
class StructuralNode:
    """A node in the document structure tree."""

    marker: str
    title: str
    level: int
    content_lines: list[str] = field(default_factory=list)
    children: list[StructuralNode] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None

    def get_chapter_path(self) -> list[str]:
        """Get the full path from root to this node."""
        return [f"{self.marker}: {self.title}"]


class FiqhChunker:
    """
    Intelligent chunker that understands Fiqh book structure.

    Strategy:
    1. Parse structural markers (كتاب → باب → فصل → مسألة → فرع)
    2. Each مسألة or فرع becomes an independent chunk
    3. Fallback to sliding window for unstructured text
    """

    def __init__(
        self,
        min_tokens: int = 400,
        max_tokens: int = 600,
        overlap_tokens: int = 100,
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(
        self,
        text: str,
        display_text: str,
        book_id: str,
        book_title: str,
        author: str,
        madhab: str | None = None,
    ) -> list[Chunk]:
        """
        Chunk text using structural parsing with sliding window fallback.

        Returns list of Chunk objects with full metadata.
        """
        page_map = self._build_page_map(text)

        # Remove page markers from text for processing
        clean_text = PAGE_MARKER_PATTERN.sub("", text)
        clean_display = PAGE_MARKER_PATTERN.sub("", display_text)

        # Try structural chunking first
        structural_chunks = self._structural_chunk(
            clean_text,
            clean_display,
            page_map,
            book_id,
            book_title,
            author,
            madhab,
        )

        if structural_chunks:
            logger.info(
                "Structural chunking produced %d chunks for '%s'",
                len(structural_chunks),
                book_title,
            )
            return structural_chunks

        # Fallback to sliding window
        logger.info("No structural markers found, using sliding window for '%s'", book_title)
        return self._sliding_window_chunk(
            clean_text,
            clean_display,
            page_map,
            book_id,
            book_title,
            author,
            madhab,
        )

    def _build_page_map(self, text: str) -> dict[int, int]:
        """
        Build a map of character position → page number from page markers.
        """
        page_map = {}
        for match in PAGE_MARKER_PATTERN.finditer(text):
            page_map[match.start()] = int(match.group(1))
        return page_map

    def _get_page_at_position(self, position: int, page_map: dict[int, int]) -> int:
        """Get the page number for a given character position."""
        current_page = 1
        for marker_pos, page_num in sorted(page_map.items()):
            if marker_pos <= position:
                current_page = page_num
            else:
                break
        return current_page

    def _structural_chunk(
        self,
        text: str,
        display_text: str,
        page_map: dict[int, int],
        book_id: str,
        book_title: str,
        author: str,
        madhab: str | None,
    ) -> list[Chunk]:
        """Attempt to chunk by structural markers."""
        chunks = []
        lines = text.split("\n")
        display_lines = display_text.split("\n")

        # Build hierarchy path as we scan
        current_path: list[tuple[str, str]] = []  # [(marker, title), ...]
        current_content: list[str] = []
        current_display: list[str] = []
        current_level = -1
        char_offset = 0

        for i, line in enumerate(lines):
            display_line = display_lines[i] if i < len(display_lines) else line

            # Check if this line is a structural marker
            found_marker = False
            for pattern, marker, level in STRUCTURAL_PATTERNS:
                match = pattern.match(line.strip())
                if match:
                    # Save accumulated content as a chunk
                    if current_content and current_level >= 3:  # مسألة or فرع level
                        chunk = self._create_chunk(
                            "\n".join(current_content),
                            "\n".join(current_display),
                            [f"{m}: {t}" for m, t in current_path],
                            page_map,
                            char_offset,
                            book_id,
                            book_title,
                            author,
                            madhab,
                        )
                        if chunk:
                            chunks.append(chunk)

                    # Update hierarchy path
                    title = match.group(2).strip()
                    # Trim path to current level
                    current_path = [
                        (m, t)
                        for m, t in current_path
                        if STRUCTURAL_MARKERS[[x[0] for x in STRUCTURAL_MARKERS].index(m)][1]
                        < level
                    ]
                    current_path.append((marker, title))
                    current_level = level
                    current_content = []
                    current_display = []
                    found_marker = True
                    break

            if not found_marker:
                current_content.append(line)
                current_display.append(display_line)

            char_offset += len(line) + 1  # +1 for newline

        # Don't forget the last accumulated chunk
        if current_content and current_level >= 3:
            chunk = self._create_chunk(
                "\n".join(current_content),
                "\n".join(current_display),
                [f"{m}: {t}" for m, t in current_path],
                page_map,
                char_offset,
                book_id,
                book_title,
                author,
                madhab,
            )
            if chunk:
                chunks.append(chunk)

        # If we found structural markers but chunks are too few,
        # apply sliding window to the large sections
        if not chunks and current_path:
            return []  # Let caller fall back to sliding window

        return chunks

    def _sliding_window_chunk(
        self,
        text: str,
        display_text: str,
        page_map: dict[int, int],
        book_id: str,
        book_title: str,
        author: str,
        madhab: str | None,
    ) -> list[Chunk]:
        """Sliding window chunking with sentence-boundary awareness."""
        chunks = []
        sentences = SENTENCE_BOUNDARY.split(text)
        display_sentences = SENTENCE_BOUNDARY.split(display_text)

        current_tokens: list[str] = []
        current_display: list[str] = []
        current_token_count = 0
        char_offset = 0

        for i, sentence in enumerate(sentences):
            display_sent = display_sentences[i] if i < len(display_sentences) else sentence
            sent_tokens = len(sentence.split())

            # If adding this sentence would exceed max, emit chunk
            if current_token_count + sent_tokens > self.max_tokens and current_tokens:
                chunk_text = " ".join(current_tokens)
                chunk_display = " ".join(current_display)

                chunk = self._create_chunk(
                    chunk_text,
                    chunk_display,
                    [],
                    page_map,
                    char_offset - len(chunk_text),
                    book_id,
                    book_title,
                    author,
                    madhab,
                )
                if chunk:
                    chunks.append(chunk)

                # Keep overlap
                overlap_tokens = 0
                overlap_start = len(current_tokens)
                for j in range(len(current_tokens) - 1, -1, -1):
                    overlap_tokens += len(current_tokens[j].split())
                    if overlap_tokens >= self.overlap_tokens:
                        overlap_start = j
                        break

                current_tokens = current_tokens[overlap_start:]
                current_display = current_display[overlap_start:]
                current_token_count = sum(len(t.split()) for t in current_tokens)

            current_tokens.append(sentence)
            current_display.append(display_sent)
            current_token_count += sent_tokens
            char_offset += len(sentence) + 1

        # Emit final chunk
        if current_tokens:
            chunk_text = " ".join(current_tokens)
            chunk_display = " ".join(current_display)
            chunk = self._create_chunk(
                chunk_text,
                chunk_display,
                [],
                page_map,
                char_offset - len(chunk_text),
                book_id,
                book_title,
                author,
                madhab,
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        text: str,
        display_text: str,
        chapter_path: list[str],
        page_map: dict[int, int],
        char_offset: int,
        book_id: str,
        book_title: str,
        author: str,
        madhab: str | None,
    ) -> Chunk | None:
        """Create a Chunk object if text is non-trivial."""
        text = text.strip()
        display_text = display_text.strip()
        if not text or len(text.split()) < 10:
            return None

        page_start = self._get_page_at_position(max(0, char_offset), page_map)
        page_end = self._get_page_at_position(char_offset + len(text), page_map)

        return Chunk(
            book_id=book_id,
            book_title=book_title,
            author=author,
            madhab=madhab,
            chapter_path=chapter_path,
            text=text,
            display_text=display_text,
            page_start=page_start,
            page_end=page_end,
            token_count=len(text.split()),
        )
