"""Arabic text cleaning and normalization for Fiqh texts."""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# ─── Arabic Unicode Ranges ──────────────────────────────────

# Diacritics (tashkeel) pattern
_DIACRITICS_PATTERN = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed]"
)

# Common page header/footer patterns in Fiqh books
_HEADER_FOOTER_PATTERNS = [
    re.compile(r"^[\d\s\-–—]+$", re.MULTILINE),  # Standalone page numbers
    re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE),  # -123- style
    re.compile(r"^ص\s*\d+", re.MULTILINE),  # ص 123
    re.compile(r"^الجزء\s*(الأول|الثاني|الثالث|الرابع)", re.MULTILINE),  # Part headers
]

# Hamza normalization mappings
_HAMZA_MAP = {
    "\u0623": "\u0627",  # أ → ا
    "\u0625": "\u0627",  # إ → ا
    "\u0622": "\u0627",  # آ → ا
    "\u0624": "\u0648",  # ؤ → و
    "\u0626": "\u064a",  # ئ → ي
}

# Alef variations normalization
_ALEF_MAP = {
    "\u0671": "\u0627",  # ٱ → ا
    "\u0672": "\u0627",  # ٲ → ا
    "\u0673": "\u0627",  # ٳ → ا
    "\u0675": "\u0627",  # ٵ → ا
}

# Tatweel (kashida) removal
_TATWEEL = "\u0640"


class ArabicTextCleaner:
    """
    Comprehensive Arabic text cleaner for Fiqh texts.

    Produces two versions:
    - cleaned text (for embedding/search): normalized, no diacritics
    - display text (for UI): keeps diacritics
    """

    def __init__(self, normalize_hamza: bool = True, strip_diacritics: bool = True):
        self._normalize_hamza = normalize_hamza
        self._strip_diacritics = strip_diacritics

    def clean(self, text: str) -> tuple[str, str]:
        """
        Clean Arabic text.

        Returns:
            (cleaned_text, display_text)
            - cleaned_text: fully normalized, no diacritics — for embeddings
            - display_text: lightly cleaned, keeps diacritics — for display
        """
        # Step 1: Unicode NFKC normalization (both versions)
        text = self._normalize_unicode(text)

        # Step 2: Remove headers, footers, page numbers
        text = self._remove_headers_footers(text)

        # Step 3: Clean whitespace
        text = self._clean_whitespace(text)

        # Display version: lightly cleaned (keeps diacritics)
        display_text = text

        # Search version: further normalization
        cleaned_text = text

        if self._normalize_hamza:
            cleaned_text = self._apply_hamza_normalization(cleaned_text)

        if self._strip_diacritics:
            cleaned_text = self._remove_diacritics(cleaned_text)

        # Remove tatweel (kashida)
        cleaned_text = cleaned_text.replace(_TATWEEL, "")

        # Normalize alef variations
        cleaned_text = self._normalize_alef(cleaned_text)

        # Final whitespace cleanup
        cleaned_text = self._clean_whitespace(cleaned_text)
        display_text = self._clean_whitespace(display_text)

        return cleaned_text, display_text

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Apply NFKC Unicode normalization."""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _remove_headers_footers(text: str) -> str:
        """Remove common page headers, footers, and standalone page numbers."""
        for pattern in _HEADER_FOOTER_PATTERNS:
            text = pattern.sub("", text)
        return text

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Normalize whitespace: collapse multiple spaces/newlines."""
        # Collapse multiple spaces to single
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _remove_diacritics(text: str) -> str:
        """Strip Arabic diacritics (tashkeel) from text."""
        return _DIACRITICS_PATTERN.sub("", text)

    @staticmethod
    def _apply_hamza_normalization(text: str) -> str:
        """Normalize hamza variations to base forms."""
        for src, dst in _HAMZA_MAP.items():
            text = text.replace(src, dst)
        return text

    @staticmethod
    def _normalize_alef(text: str) -> str:
        """Normalize alef variations."""
        for src, dst in _ALEF_MAP.items():
            text = text.replace(src, dst)
        return text


# ─── Convenience Function ───────────────────────────────────


def clean_arabic(text: str) -> tuple[str, str]:
    """Quick-access function for Arabic text cleaning."""
    cleaner = ArabicTextCleaner()
    return cleaner.clean(text)
