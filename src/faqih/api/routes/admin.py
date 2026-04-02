"""Admin endpoints for ingestion and management."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from faqih.api.dependencies import verify_admin_key
from faqih.ingestion.pipeline import IngestionPipeline

router = APIRouter()


class IngestRequest(BaseModel):
    """Request body for book ingestion."""

    file_path: str
    book_title: str
    author: str
    madhab: str | None = None
    recreate_indexes: bool = False


@router.post("/ingest", dependencies=[Depends(verify_admin_key)])
async def ingest_book(body: IngestRequest, request: Request):
    """
    Trigger the ingestion pipeline for a new book.

    Requires admin API key in X-API-Key header.
    """
    file_path = Path(body.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {body.file_path}")

    pipeline = IngestionPipeline(settings=request.app.state.settings)
    try:
        await pipeline.setup(recreate_indexes=body.recreate_indexes)
        metadata = await pipeline.ingest_book(
            file_path=file_path,
            book_title=body.book_title,
            author=body.author,
            madhab=body.madhab,
        )
        return {
            "status": "success",
            "book_id": metadata.book_id,
            "total_pages": metadata.total_pages,
            "total_chunks": metadata.total_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await pipeline.teardown()
