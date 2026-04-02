"""CLI script for ingesting a Fiqh book."""

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from faqih.ingestion.pipeline import IngestionPipeline

console = Console()


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--title", "-t", required=True, help="Book title (Arabic preferred)")
@click.option("--author", "-a", required=True, help="Book author")
@click.option(
    "--madhab", "-m",
    type=click.Choice(["hanafi", "maliki", "shafii", "hanbali"]),
    help="Book's madhab (optional)",
)
@click.option("--recreate", is_flag=True, help="Recreate all indexes from scratch")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def main(file_path: str, title: str, author: str, madhab: str | None, recreate: bool, verbose: bool):
    """
    Ingest a Fiqh book (PDF/DOCX) into the Al-Ijtihad system.

    Example:
        python scripts/ingest_book.py data/books/al-mughni.pdf \\
            --title "المغني" --author "ابن قدامة" --madhab hanbali
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    console.print("\n[bold cyan]═══ Al-Ijtihad Book Ingestion ═══[/bold cyan]\n")

    async def run():
        pipeline = IngestionPipeline()
        try:
            await pipeline.setup(recreate_indexes=recreate)
            metadata = await pipeline.ingest_book(
                file_path=file_path,
                book_title=title,
                author=author,
                madhab=madhab,
            )
            console.print(f"\n[bold green]Book ID: {metadata.book_id}[/bold green]")
            console.print(f"Pages: {metadata.total_pages}")
            console.print(f"Chunks: {metadata.total_chunks}")
        except Exception as e:
            console.print(f"\n[bold red]Error: {e}[/bold red]")
            sys.exit(1)
        finally:
            await pipeline.teardown()

    asyncio.run(run())


if __name__ == "__main__":
    main()
