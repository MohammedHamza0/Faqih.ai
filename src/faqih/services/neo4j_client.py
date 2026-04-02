"""Neo4j async client wrapper."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Async wrapper around the Neo4j Python driver."""

    def __init__(self, uri: str, user: str, password: str):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: AsyncDriver | None = None

    async def connect(self):
        """Establish connection to Neo4j."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        await self._driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", self._uri)

    async def close(self):
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j connection closed")

    @asynccontextmanager
    async def session(self, database: str = "neo4j"):
        """Get an async Neo4j session."""
        if not self._driver:
            raise RuntimeError("Neo4j client not connected. Call connect() first.")
        async with self._driver.session(database=database) as session:
            yield session

    async def run_query(
        self, query: str, parameters: dict[str, Any] | None = None, database: str = "neo4j"
    ) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        async with self.session(database=database) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def create_node(
        self, labels: list[str], properties: dict[str, Any], database: str = "neo4j"
    ) -> str:
        """Create a node and return its element ID."""
        label_str = ":".join(labels)
        query = f"CREATE (n:{label_str} $props) RETURN elementId(n) AS node_id"
        results = await self.run_query(query, {"props": properties}, database)
        return results[0]["node_id"] if results else ""

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        database: str = "neo4j",
    ):
        """Create a relationship between two nodes by element ID."""
        query = (
            "MATCH (a) WHERE elementId(a) = $source_id "
            "MATCH (b) WHERE elementId(b) = $target_id "
            f"CREATE (a)-[r:{rel_type} $props]->(b) "
            "RETURN type(r)"
        )
        await self.run_query(
            query,
            {"source_id": source_id, "target_id": target_id, "props": properties or {}},
            database,
        )

    async def merge_node(
        self,
        labels: list[str],
        match_props: dict[str, Any],
        set_props: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> str:
        """Merge (upsert) a node — create if not exists, update if exists."""
        label_str = ":".join(labels)
        set_clause = ""
        params: dict[str, Any] = {"match_props": match_props}

        if set_props:
            set_clause = "SET n += $set_props"
            params["set_props"] = set_props

        query = f"MERGE (n:{label_str} {{canonical_id: $match_props.canonical_id}}) {set_clause} RETURN elementId(n) AS node_id"
        results = await self.run_query(query, params, database)
        return results[0]["node_id"] if results else ""

    async def setup_indexes(self, database: str = "neo4j"):
        """Create indexes for efficient lookups."""
        indexes = [
            "CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)",
            "CREATE INDEX masala_canonical_idx IF NOT EXISTS FOR (m:Masala) ON (m.canonical_id)",
            "CREATE INDEX entity_canonical_idx IF NOT EXISTS FOR (e:Entity) ON (e.canonical_id)",
            "CREATE INDEX book_id_idx IF NOT EXISTS FOR (b:Book) ON (b.book_id)",
        ]
        for idx_query in indexes:
            await self.run_query(idx_query, database=database)
        logger.info("Neo4j indexes created")
