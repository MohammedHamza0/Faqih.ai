"""Tests for Pydantic models and enums."""

from faqih.models.enums import ChunkType, Madhab, QuestionType, RelationType
from faqih.models.schemas import Chunk, QueryIntent, Session


class TestEnums:
    """Test enum values and serialization."""

    def test_chunk_types(self):
        assert ChunkType.HUKM == "hukm"
        assert ChunkType.DALIL == "dalil"
        assert ChunkType.KHILAF == "khilaf"

    def test_madhab_values(self):
        assert Madhab.HANAFI == "hanafi"
        assert Madhab.SHAFII == "shafii"

    def test_relation_types(self):
        assert RelationType.DISAGREES_WITH == "DISAGREES_WITH"
        assert RelationType.EVIDENCED_BY == "EVIDENCED_BY"


class TestChunkModel:
    """Test Chunk Pydantic model."""

    def test_create_chunk(self):
        chunk = Chunk(
            book_id="book1",
            book_title="المغني",
            author="ابن قدامة",
            text="test text",
            display_text="test display",
        )
        assert chunk.chunk_id  # Auto-generated UUID
        assert chunk.book_title == "المغني"
        assert chunk.chunk_type == ChunkType.GENERAL

    def test_chunk_serialization(self):
        chunk = Chunk(
            book_id="b1",
            book_title="Test",
            author="Author",
            text="text",
            display_text="display",
            madhab=Madhab.HANAFI,
            chapter_path=["كتاب: الطهارة"],
        )
        data = chunk.model_dump()
        assert data["madhab"] == "hanafi"
        assert data["chapter_path"] == ["كتاب: الطهارة"]


class TestSessionModel:
    """Test Session model."""

    def test_create_session(self):
        session = Session()
        assert session.session_id
        assert session.turns == []
        assert session.entity_history == []

    def test_query_intent_defaults(self):
        intent = QueryIntent()
        assert intent.madhab_preference == "all"
        assert intent.question_type == QuestionType.FATWA
        assert intent.language == "arabic"
