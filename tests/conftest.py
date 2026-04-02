"""Test fixtures and shared configuration."""

import pytest


@pytest.fixture
def sample_arabic_text():
    """Sample Arabic Fiqh text for testing."""
    return """
كتاب الطهارة
باب أحكام المياه
مسألة: حكم الماء المطلق
الماء المطلق طاهر في نفسه مطهر لغيره بالإجماع. قال ابن قدامة في المغني: لا نعلم في ذلك خلافاً.
والدليل قوله تعالى: {وَأَنزَلْنَا مِنَ السَّمَاءِ مَاءً طَهُوراً}
فرع: حكم الماء المستعمل
اختلف الفقهاء في حكم الماء المستعمل في رفع الحدث:
فذهب الحنفية إلى أنه طاهر غير مطهر.
وذهب الشافعية في الجديد إلى أنه طاهر مطهر.
وذهب الحنابلة إلى أنه طاهر غير مطهر.
"""


@pytest.fixture
def sample_chunk_data():
    """Sample chunk payload data."""
    return {
        "chunk_id": "test-chunk-001",
        "book_id": "test-book-001",
        "book_title": "المغني",
        "author": "ابن قدامة",
        "madhab": "hanbali",
        "chapter_path": ["كتاب: الطهارة", "باب: أحكام المياه", "مسألة: حكم الماء المطلق"],
        "chunk_type": "hukm",
        "text": "الماء المطلق طاهر في نفسه مطهر لغيره بالإجماع",
        "display_text": "الماءُ المُطلَقُ طاهِرٌ في نَفسِهِ مُطَهِّرٌ لِغَيرِهِ بالإجماعِ",
        "page_start": 15,
        "page_end": 16,
        "token_count": 12,
    }
