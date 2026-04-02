"""Chunk detail endpoint for citation cards."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from faqih.api.dependencies import get_qdrant
from faqih.services.qdrant_client import QdrantService

router = APIRouter()


@router.get("/{chunk_id}")
async def get_chunk(
    chunk_id: str,
    request: Request,
    qdrant: QdrantService = Depends(get_qdrant),
):
    """
    Get full chunk content for a citation card.

    Returns the chunk text, display text, metadata, and source info.
    """
    collection = request.app.state.settings.qdrant_collection
    point = await qdrant.get_point(collection, chunk_id)

    if not point or not point.payload:
        raise HTTPException(status_code=404, detail="Chunk not found")

    payload = point.payload
    return {
        "chunk_id": payload.get("chunk_id", chunk_id),
        "book_title": payload.get("book_title", ""),
        "author": payload.get("author", ""),
        "madhab": payload.get("madhab", ""),
        "chapter_path": payload.get("chapter_path", []),
        "chunk_type": payload.get("chunk_type", ""),
        "text": payload.get("text", ""),
        "display_text": payload.get("display_text", ""),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
    }
