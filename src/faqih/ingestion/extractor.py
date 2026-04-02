"""Book text extraction from PDF and DOCX files."""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document

logger = logging.getLogger(__name__)


class BookExtractor:
    """Extracts raw text from PDF and DOCX files with page markers."""

    # Page boundary marker used downstream for page tracking
    PAGE_MARKER = "\n<<PAGE:{page_num}>>\n"

    def extract(self, file_path: str | Path) -> tuple[str, int]:
        """
        Extract text from a file.

        Returns:
            Tuple of (raw_text_with_page_markers, total_pages)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix in (".docx", ".doc"):
            return self._extract_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use PDF or DOCX.")

    def _extract_pdf(self, file_path: Path) -> tuple[str, int]:
        """Extract text from PDF using PyMuPDF."""
        logger.info("Extracting PDF: %s", file_path.name)
        doc = fitz.open(str(file_path))
        pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                marker = self.PAGE_MARKER.format(page_num=page_num + 1)
                pages.append(f"{marker}{text}")

        total_pages = len(doc)
        doc.close()

        full_text = "\n".join(pages)
        logger.info("Extracted %d pages from PDF", total_pages)
        return full_text, total_pages

    def _extract_docx(self, file_path: Path) -> tuple[str, int]:
        """Extract text from DOCX using python-docx."""
        logger.info("Extracting DOCX: %s", file_path.name)
        doc = Document(str(file_path))
        paragraphs = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        full_text = "\n".join(paragraphs)
        # DOCX doesn't have explicit pages; estimate based on content
        estimated_pages = max(1, len(full_text) // 2000)
        logger.info("Extracted %d paragraphs from DOCX", len(paragraphs))
        return full_text, estimated_pages
