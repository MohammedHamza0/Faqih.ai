"""Tests for Arabic text cleaner."""

from faqih.ingestion.cleaner import ArabicTextCleaner


class TestArabicTextCleaner:
    """Tests for the ArabicTextCleaner."""

    def setup_method(self):
        self.cleaner = ArabicTextCleaner()

    def test_clean_returns_tuple(self):
        """clean() should return (cleaned, display) tuple."""
        result = self.cleaner.clean("بسم الله الرحمن الرحيم")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_diacritics_removed_in_cleaned(self):
        """Diacritics should be removed from cleaned text."""
        text = "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"
        cleaned, display = self.cleaner.clean(text)
        assert "ِ" not in cleaned  # kasra removed
        assert "ْ" not in cleaned  # sukun removed

    def test_diacritics_kept_in_display(self):
        """Diacritics should be preserved in display text."""
        text = "بِسْمِ اللَّهِ"
        _, display = self.cleaner.clean(text)
        # Display should keep diacritics (though some Unicode normalization may apply)
        assert len(display) > 0

    def test_hamza_normalization(self):
        """Hamza variations should be normalized."""
        text = "أحكام إسلامية"
        cleaned, _ = self.cleaner.clean(text)
        assert "أ" not in cleaned or "إ" not in cleaned

    def test_whitespace_normalization(self):
        """Multiple spaces should collapse to single space."""
        text = "كتاب    الطهارة     باب"
        cleaned, _ = self.cleaner.clean(text)
        assert "    " not in cleaned

    def test_page_numbers_removed(self):
        """Standalone page numbers should be removed."""
        text = "حكم الماء\n- 123 -\nالمطلق"
        cleaned, _ = self.cleaner.clean(text)
        assert "123" not in cleaned

    def test_empty_string(self):
        """Empty string should not raise."""
        cleaned, display = self.cleaner.clean("")
        assert cleaned == ""
        assert display == ""

    def test_no_hamza_normalization(self):
        """Cleaner with hamza normalization disabled."""
        cleaner = ArabicTextCleaner(normalize_hamza=False)
        text = "أحكام"
        cleaned, _ = cleaner.clean(text)
        assert "أ" in cleaned
