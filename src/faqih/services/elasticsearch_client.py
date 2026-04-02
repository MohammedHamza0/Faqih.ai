"""Elasticsearch client wrapper with Arabic analyzer support."""

from __future__ import annotations

import logging
from typing import Any

from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)

# ─── Fiqh Synonym Mapping ───────────────────────────────────

FIQH_SYNONYMS = [
    "واجب, فرض",
    "حرام, محرم",
    "مكروه, مكروه تحريما, مكروه تنزيها",
    "مباح, جائز, حلال",
    "مندوب, مستحب, سنة",
    "صلاة, صلوة",
    "زكاة, زكوة",
    "الوضوء, الوظوء",
    "طهارة, تطهير",
    "نجاسة, نجس",
    "حيض, محيض",
    "صيام, صوم",
    "حج, حجة",
    "نكاح, زواج",
    "طلاق, تطليق",
    "بيع, مبايعة",
    "ربا, ربو",
    "إجارة, استئجار",
    "كفارة, كفارات",
]


class ElasticsearchService:
    """Async Elasticsearch client with custom Arabic analysis."""

    def __init__(self, url: str):
        self._url = url
        self._client: AsyncElasticsearch | None = None

    async def connect(self):
        """Initialize Elasticsearch client."""
        self._client = AsyncElasticsearch(
            self._url,
            verify_certs=False,
            request_timeout=30,
        )
        info = await self._client.info()
        logger.info("Connected to Elasticsearch %s", info["version"]["number"])

    async def close(self):
        """Close the Elasticsearch client."""
        if self._client:
            await self._client.close()
            logger.info("Elasticsearch connection closed")

    @property
    def client(self) -> AsyncElasticsearch:
        if not self._client:
            raise RuntimeError("Elasticsearch not connected. Call connect() first.")
        return self._client

    async def create_index(self, index_name: str, recreate: bool = False):
        """Create index with custom Arabic analyzer + Fiqh synonyms."""
        exists = await self.client.indices.exists(index=index_name)

        if exists and recreate:
            await self.client.indices.delete(index=index_name)
            logger.info("Deleted existing index: %s", index_name)
            exists = False

        if not exists:
            settings = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "filter": {
                            "fiqh_synonyms": {
                                "type": "synonym",
                                "synonyms": FIQH_SYNONYMS,
                            },
                            "arabic_stop": {
                                "type": "stop",
                                "stopwords": "_arabic_",
                            },
                            "arabic_stemmer": {
                                "type": "stemmer",
                                "language": "arabic",
                            },
                        },
                        "analyzer": {
                            "fiqh_arabic": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "arabic_normalization",
                                    "fiqh_synonyms",
                                    "arabic_stop",
                                    "arabic_stemmer",
                                ],
                            },
                        },
                    },
                },
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "book_id": {"type": "keyword"},
                        "book_title": {"type": "text", "analyzer": "fiqh_arabic"},
                        "author": {"type": "keyword"},
                        "madhab": {"type": "keyword"},
                        "chapter_path": {"type": "keyword"},
                        "chunk_type": {"type": "keyword"},
                        "text": {
                            "type": "text",
                            "analyzer": "fiqh_arabic",
                            "search_analyzer": "fiqh_arabic",
                        },
                        "page_start": {"type": "integer"},
                        "page_end": {"type": "integer"},
                        "token_count": {"type": "integer"},
                    }
                },
            }
            await self.client.indices.create(index=index_name, body=settings)
            logger.info("Created index: %s with fiqh_arabic analyzer", index_name)
        else:
            logger.info("Index already exists: %s", index_name)

    async def bulk_index(
        self,
        index_name: str,
        documents: list[dict[str, Any]],
        batch_size: int = 500,
    ):
        """Bulk index documents."""
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            actions = []
            for doc in batch:
                actions.append({"index": {"_index": index_name, "_id": doc["chunk_id"]}})
                actions.append(doc)

            await self.client.bulk(body=actions, refresh=False)

        await self.client.indices.refresh(index=index_name)
        logger.info("Indexed %d documents in %s", len(documents), index_name)

    async def search(
        self,
        index_name: str,
        query_text: str,
        size: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """BM25 search with optional filters."""
        must_clauses: list[dict] = [
            {"match": {"text": {"query": query_text, "analyzer": "fiqh_arabic"}}}
        ]

        if filters:
            for field, value in filters.items():
                must_clauses.append({"term": {field: value}})

        body = {
            "query": {"bool": {"must": must_clauses}},
            "size": size,
        }

        result = await self.client.search(index=index_name, body=body)
        hits = []
        for hit in result["hits"]["hits"]:
            hits.append({
                "chunk_id": hit["_id"],
                "score": hit["_score"],
                "source": hit["_source"],
            })
        return hits
