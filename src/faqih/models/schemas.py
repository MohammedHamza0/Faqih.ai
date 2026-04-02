"""Pydantic schemas for the Fiqh RAG system."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from faqih.models.enums import (
    ChunkType,
    DetailLevel,
    EntityType,
    Madhab,
    QuestionType,
    RelationType,
)


# ─── Ingestion Schemas ──────────────────────────────────────


class Chunk(BaseModel):
    """A semantically meaningful segment of a Fiqh text."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    book_id: str
    book_title: str
    author: str
    madhab: Optional[Madhab] = None
    chapter_path: list[str] = Field(
        default_factory=list,
        description="Hierarchical path: [كتاب, باب, فصل, مسألة]",
    )
    chunk_type: ChunkType = ChunkType.GENERAL
    text: str = Field(description="Cleaned text for embedding and search")
    display_text: str = Field(description="Original text with diacritics for display")
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    embedding: Optional[list[float]] = Field(default=None, exclude=True)
    graph_node_id: Optional[str] = None
    token_count: int = 0


class Entity(BaseModel):
    """An entity extracted from a Fiqh text chunk."""

    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: EntityType
    text: str
    canonical_id: Optional[str] = None
    source_chunk_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class Relationship(BaseModel):
    """A relationship between two entities in the knowledge graph."""

    source_id: str
    target_id: str
    type: RelationType
    madhab: Optional[Madhab] = None
    weight: float = 1.0
    metadata: dict = Field(default_factory=dict)


class BookMetadata(BaseModel):
    """Metadata for an ingested book."""

    book_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    author: str
    madhab: Optional[Madhab] = None
    file_path: str
    total_pages: int = 0
    total_chunks: int = 0
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Query & Conversation Schemas ───────────────────────────


class QueryIntent(BaseModel):
    """Detected intent from a user query."""

    madhab_preference: Optional[Madhab | str] = Field(
        default="all",
        description="Specific madhab or 'all' for comparative",
    )
    question_type: QuestionType = QuestionType.FATWA
    detail_level: DetailLevel = DetailLevel.STANDARD
    language: str = "arabic"


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    role: str = Field(description="'user' or 'assistant'")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    entities_mentioned: list[str] = Field(default_factory=list)
    chunk_ids_cited: list[str] = Field(default_factory=list)


class Session(BaseModel):
    """A conversation session with memory."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turns: list[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    entity_history: list[str] = Field(
        default_factory=list,
        description="Entities mentioned across the session",
    )
    resolved_references: dict[str, str] = Field(
        default_factory=dict,
        description="Map of anaphoric references to resolved entities",
    )
    active_madhab: Optional[Madhab] = None


# ─── Retrieval Schemas ──────────────────────────────────────


class RetrievalResult(BaseModel):
    """A single result from any retrieval source."""

    chunk_id: str
    score: float
    source: str = Field(description="'vector' | 'graph' | 'bm25'")
    chunk: Optional[Chunk] = None


class FusedResult(BaseModel):
    """A result after RRF fusion and optional reranking."""

    chunk_id: str
    rrf_score: float
    reranker_score: Optional[float] = None
    chunk: Optional[Chunk] = None
    sources: list[str] = Field(default_factory=list)


# ─── Generation Schemas ─────────────────────────────────────


class Citation(BaseModel):
    """A citation linking generated text to a source chunk."""

    marker: str = Field(description="Citation text, e.g. [الكتاب، الباب]")
    chunk_id: str
    book_title: str
    chapter_path: list[str]
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class GenerationResponse(BaseModel):
    """Complete response from the generation pipeline."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    intent: Optional[QueryIntent] = None
    chunks_used: list[str] = Field(default_factory=list)
    session_id: str
    cached: bool = False
