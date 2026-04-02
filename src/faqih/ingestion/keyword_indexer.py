"""Keyword indexer — stores chunks in Elasticsearch for BM25 search."""

from __future__ import annotations

import logging

from faqih.models.schemas import Chunk
from faqih.services.elasticsearch_client import ElasticsearchService

logger = logging.getLogger(__name__)


class KeywordIndexer:
    """Indexes chunks into Elasticsearch with the custom Fiqh Arabic analyzer."""

    def __init__(
        self,
        elasticsearch: ElasticsearchService,
        index_name: str = "fiqh_chunks",
    ):
        self._es = elasticsearch
        self._index_name = index_name

    async def setup(self, recreate: bool = False):
        """Create the Elasticsearch index if needed."""
        await self._es.create_index(
            index_name=self._index_name,
            recreate=recreate,
        )

    async def index_chunks(self, chunks: list[Chunk]):
        """Index a list of chunks into Elasticsearch."""
        if not chunks:
            return

        logger.info("Indexing %d chunks into Elasticsearch", len(chunks))

        documents = []
        for chunk in chunks:
            doc = {
                "chunk_id": chunk.chunk_id,
                "book_id": chunk.book_id,
                "book_title": chunk.book_title,
                "author": chunk.author,
                "madhab": chunk.madhab or "",
                "chapter_path": chunk.chapter_path,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_count": chunk.token_count,
            }
            documents.append(doc)

        await self._es.bulk_index(
            index_name=self._index_name,
            documents=documents,
        )

        logger.info("Successfully indexed %d chunks in Elasticsearch", len(chunks))
