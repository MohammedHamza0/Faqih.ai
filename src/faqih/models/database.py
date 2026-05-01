"""SQLAlchemy ORM models for PostgreSQL persistence."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from faqih.models.enums import ChunkType


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all ORM models."""

    pass


# ─── Book Model ─────────────────────────────────────────────


class BookRecord(Base):
    """Persisted book metadata."""

    __tablename__ = "books"

    book_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    author: Mapped[str] = mapped_column(String(300), nullable=False)
    madhab: Mapped[str | None] = mapped_column(String(20), nullable=True)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list[ChunkRecord]] = relationship(back_populates="book")


# ─── Chunk Model ────────────────────────────────────────────


class ChunkRecord(Base):
    """Persisted chunk metadata (text stored in Qdrant/ES, reference here)."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    book_id: Mapped[str] = mapped_column(String(36), ForeignKey("books.book_id"), nullable=False)
    chapter_path: Mapped[dict] = mapped_column(JSON, default=list)
    chunk_type: Mapped[str] = mapped_column(String(20), default=ChunkType.GENERAL)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    graph_node_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    book: Mapped[BookRecord] = relationship(back_populates="chunks")


# ─── Session Model ──────────────────────────────────────────


class SessionRecord(Base):
    """Persisted conversation session."""

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    active_madhab: Mapped[str | None] = mapped_column(String(20), nullable=True)
    entity_history: Mapped[dict] = mapped_column(JSON, default=list)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


class TurnRecord(Base):
    """Persisted conversation turn."""

    __tablename__ = "turns"

    turn_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.session_id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    entities_mentioned: Mapped[dict] = mapped_column(JSON, default=list)
    chunk_ids_cited: Mapped[dict] = mapped_column(JSON, default=list)


# ─── Database Engine Factory ────────────────────────────────


def create_db_engine(database_url: str):
    """Create async SQLAlchemy engine."""
    return create_async_engine(database_url, echo=False, pool_size=10, max_overflow=20)


def create_session_factory(engine):
    """Create async session factory."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine):
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
