"""Vector indexer — embeds chunks and stores in Qdrant."""

from __future__ import annotations

import logging

from qdrant_client import models

from faqih.models.schemas import Chunk
from faqih.services.embedding import EmbeddingService
from faqih.services.qdrant_client import QdrantService

logger = logging.getLogger(__name__)


class VectorIndexer:
    """
    Indexes chunks into Qdrant with dense embeddings.

    Each point stores:
    - dense vector (768-dim from AraBERT)
    - full metadata payload
    """

    def __init__(
        self,
        qdrant: QdrantService,
        embedding: EmbeddingService,
        collection_name: str = "fiqh_chunks",
    ):
        self._qdrant = qdrant
        self._embedding = embedding
        self._collection_name = collection_name

    async def setup(self, recreate: bool = False):
        """Create the Qdrant collection if needed."""
        await self._qdrant.create_collection(
            collection_name=self._collection_name,
            dense_dim=self._embedding.dim,
            recreate=recreate,
        )

    async def index_chunks(self, chunks: list[Chunk], batch_size: int = 32):
        """
        Embed and index a list of chunks into Qdrant.

        Args:
            chunks: List of Chunk objects to index
            batch_size: Batch size for embedding computation
        """
        if not chunks:
            return

        logger.info("Indexing %d chunks into Qdrant", len(chunks))

        # Batch embed all chunk texts
        texts = [chunk.text for chunk in chunks]
        all_embeddings = self._embedding.encode(texts, batch_size=batch_size, show_progress=True)

        # Build Qdrant points
        points = []
        for i, chunk in enumerate(chunks):
            payload = {
                "chunk_id": chunk.chunk_id,
                "book_id": chunk.book_id,
                "book_title": chunk.book_title,
                "author": chunk.author,
                "madhab": chunk.madhab or "",
                "chapter_path": chunk.chapter_path,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,
                "display_text": chunk.display_text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_count": chunk.token_count,
                "graph_node_id": chunk.graph_node_id or "",
            }

            point = models.PointStruct(
                id=chunk.chunk_id,
                vector={"dense": all_embeddings[i].tolist()},
                payload=payload,
            )
            points.append(point)

        # Upsert in batches
        await self._qdrant.upsert_points(
            collection_name=self._collection_name,
            points=points,
            batch_size=100,
        )

        logger.info("Successfully indexed %d chunks in Qdrant", len(chunks))
