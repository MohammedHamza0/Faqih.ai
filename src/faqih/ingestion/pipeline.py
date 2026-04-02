"""Ingestion pipeline orchestrator — end-to-end book processing."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from faqih.config import get_settings
from faqih.ingestion.chunker import FiqhChunker
from faqih.ingestion.cleaner import ArabicTextCleaner
from faqih.ingestion.extractor import BookExtractor
from faqih.ingestion.graph_builder import GraphBuilder
from faqih.ingestion.keyword_indexer import KeywordIndexer
from faqih.ingestion.vector_indexer import VectorIndexer
from faqih.models.enums import Madhab
from faqih.models.schemas import BookMetadata
from faqih.services.elasticsearch_client import ElasticsearchService
from faqih.services.embedding import EmbeddingService
from faqih.services.llm import LLMClient
from faqih.services.neo4j_client import Neo4jClient
from faqih.services.qdrant_client import QdrantService

logger = logging.getLogger(__name__)
console = Console()


class IngestionPipeline:
    """
    Full ingestion pipeline for Fiqh books.

    Steps:
    1. Extract text from PDF/DOCX
    2. Clean and normalize Arabic text
    3. Chunk by Fiqh structure (with sliding window fallback)
    4. Build knowledge graph (LLM entity extraction → Neo4j)
    5. Index vectors (Qdrant) and keywords (Elasticsearch) in parallel
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._extractor = BookExtractor()
        self._cleaner = ArabicTextCleaner()
        self._chunker = FiqhChunker(
            min_tokens=self._settings.chunk_min_tokens,
            max_tokens=self._settings.chunk_max_tokens,
            overlap_tokens=self._settings.chunk_overlap_tokens,
        )

        # Services (initialized during setup)
        self._neo4j: Neo4jClient | None = None
        self._qdrant: QdrantService | None = None
        self._es: ElasticsearchService | None = None
        self._embedding: EmbeddingService | None = None
        self._llm: LLMClient | None = None
        self._graph_builder: GraphBuilder | None = None
        self._vector_indexer: VectorIndexer | None = None
        self._keyword_indexer: KeywordIndexer | None = None

    async def setup(self, recreate_indexes: bool = False):
        """Initialize all services and create indexes."""
        s = self._settings
        console.print("[bold cyan]Initializing services...[/bold cyan]")

        # Initialize service clients
        self._neo4j = Neo4jClient(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
        await self._neo4j.connect()

        self._qdrant = QdrantService(s.qdrant_host, s.qdrant_port, s.qdrant_grpc_port)
        await self._qdrant.connect()

        self._es = ElasticsearchService(s.elasticsearch_url)
        await self._es.connect()

        # Load embedding model
        self._embedding = EmbeddingService(s.embedding_model)
        self._embedding.load()

        # Initialize LLM client
        api_key = {
            "google": s.google_api_key,
            "openai": s.openai_api_key,
            "anthropic": s.anthropic_api_key,
        }.get(s.llm_provider, s.google_api_key)
        self._llm = LLMClient(
            provider=s.llm_provider,
            model=s.llm_model,
            api_key=api_key,
            temperature=s.llm_temperature,
        )

        # Initialize pipeline components
        self._graph_builder = GraphBuilder(self._llm, self._neo4j)
        await self._graph_builder.setup()

        self._vector_indexer = VectorIndexer(
            self._qdrant, self._embedding, s.qdrant_collection
        )
        await self._vector_indexer.setup(recreate=recreate_indexes)

        self._keyword_indexer = KeywordIndexer(self._es, s.elasticsearch_index)
        await self._keyword_indexer.setup(recreate=recreate_indexes)

        console.print("[bold green]All services initialized ✓[/bold green]")

    async def teardown(self):
        """Close all service connections."""
        if self._neo4j:
            await self._neo4j.close()
        if self._qdrant:
            await self._qdrant.close()
        if self._es:
            await self._es.close()

    async def ingest_book(
        self,
        file_path: str | Path,
        book_title: str,
        author: str,
        madhab: str | None = None,
    ) -> BookMetadata:
        """
        Run the full ingestion pipeline on a single book.

        Returns:
            BookMetadata with ingestion results
        """
        start_time = time.time()
        file_path = Path(file_path)
        console.print(f"\n[bold]Ingesting: {book_title}[/bold] by {author}")
        console.print(f"File: {file_path}")

        metadata = BookMetadata(
            title=book_title,
            author=author,
            madhab=madhab,
            file_path=str(file_path),
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # ── Step 1: Extract ──────────────────────────────
            task = progress.add_task("Extracting text...", total=None)
            raw_text, total_pages = self._extractor.extract(file_path)
            metadata.total_pages = total_pages
            progress.update(task, description=f"✓ Extracted {total_pages} pages")

            # ── Step 2: Clean ────────────────────────────────
            task = progress.add_task("Cleaning Arabic text...", total=None)
            cleaned_text, display_text = self._cleaner.clean(raw_text)
            progress.update(task, description="✓ Text cleaned and normalized")

            # ── Step 3: Chunk ────────────────────────────────
            task = progress.add_task("Chunking by Fiqh structure...", total=None)
            chunks = self._chunker.chunk(
                text=cleaned_text,
                display_text=display_text,
                book_id=metadata.book_id,
                book_title=book_title,
                author=author,
                madhab=madhab,
            )
            metadata.total_chunks = len(chunks)
            progress.update(task, description=f"✓ Created {len(chunks)} chunks")

            # ── Step 4: Knowledge Graph ──────────────────────
            task = progress.add_task("Building knowledge graph...", total=None)
            entities, relationships = await self._graph_builder.process_chunks_batch(chunks)
            progress.update(
                task,
                description=f"✓ Graph: {len(entities)} entities, {len(relationships)} relationships",
            )

            # ── Step 5: Indexing (parallel) ──────────────────
            task = progress.add_task("Indexing vectors + keywords...", total=None)
            await asyncio.gather(
                self._vector_indexer.index_chunks(chunks),
                self._keyword_indexer.index_chunks(chunks),
            )
            progress.update(task, description="✓ Indexed in Qdrant + Elasticsearch")

        elapsed = time.time() - start_time
        console.print(
            f"\n[bold green]Ingestion complete![/bold green] "
            f"{len(chunks)} chunks in {elapsed:.1f}s"
        )

        return metadata


# ─── CLI Entry Point ────────────────────────────────────────


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--title", "-t", required=True, help="Book title")
@click.option("--author", "-a", required=True, help="Book author")
@click.option("--madhab", "-m", type=click.Choice(["hanafi", "maliki", "shafii", "hanbali"]))
@click.option("--recreate", is_flag=True, help="Recreate all indexes")
def cli(file_path: str, title: str, author: str, madhab: str | None, recreate: bool):
    """Ingest a Fiqh book into the system."""

    async def run():
        pipeline = IngestionPipeline()
        try:
            await pipeline.setup(recreate_indexes=recreate)
            metadata = await pipeline.ingest_book(file_path, title, author, madhab)
            console.print(f"\nBook ID: {metadata.book_id}")
        finally:
            await pipeline.teardown()

    asyncio.run(run())


if __name__ == "__main__":
    cli()
