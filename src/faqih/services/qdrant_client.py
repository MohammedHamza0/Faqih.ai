"""Qdrant vector database client wrapper."""

from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger(__name__)


class QdrantService:
    """Async wrapper around the Qdrant client."""

    def __init__(self, host: str, port: int, grpc_port: int = 6334):
        self._host = host
        self._port = port
        self._grpc_port = grpc_port
        self._client: AsyncQdrantClient | None = None

    async def connect(self):
        """Initialize the Qdrant client."""
        self._client = AsyncQdrantClient(
            host=self._host,
            port=self._port,
            grpc_port=self._grpc_port,
            prefer_grpc=True,
        )
        logger.info("Connected to Qdrant at %s:%d", self._host, self._port)

    async def close(self):
        """Close the Qdrant client."""
        if self._client:
            await self._client.close()
            logger.info("Qdrant connection closed")

    @property
    def client(self) -> AsyncQdrantClient:
        if not self._client:
            raise RuntimeError("Qdrant client not connected. Call connect() first.")
        return self._client

    async def create_collection(
        self,
        collection_name: str,
        dense_dim: int = 768,
        recreate: bool = False,
    ):
        """Create a collection with dense and sparse named vectors."""
        collections = await self.client.get_collections()
        exists = any(c.name == collection_name for c in collections.collections)

        if exists and recreate:
            await self.client.delete_collection(collection_name)
            logger.info("Deleted existing collection: %s", collection_name)
            exists = False

        if not exists:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dense_dim,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                },
            )
            logger.info("Created collection: %s (dense_dim=%d)", collection_name, dense_dim)
        else:
            logger.info("Collection already exists: %s", collection_name)

    async def upsert_points(
        self,
        collection_name: str,
        points: list[models.PointStruct],
        batch_size: int = 100,
    ):
        """Batch upsert points into a collection."""
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )
        logger.info("Upserted %d points into %s", len(points), collection_name)

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 20,
        query_filter: models.Filter | None = None,
        with_payload: bool = True,
    ) -> list[models.ScoredPoint]:
        """Search for similar vectors."""
        results = await self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
            limit=limit,
            query_filter=query_filter,
            with_payload=with_payload,
        )
        return results.points

    async def get_point(self, collection_name: str, point_id: str) -> models.Record | None:
        """Retrieve a single point by ID."""
        results = await self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
        )
        return results[0] if results else None
