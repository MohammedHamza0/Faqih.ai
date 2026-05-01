"""Health check script for all backing services."""

from __future__ import annotations

import sys

import httpx
from neo4j import GraphDatabase
from redis import Redis
from rich.console import Console
from rich.table import Table

from faqih.config import get_settings


def check_neo4j(settings) -> tuple[bool, str]:
    """Check Neo4j connectivity."""
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        driver.verify_connectivity()
        driver.close()
        return True, "Connected"
    except Exception as e:
        return False, str(e)[:80]


def check_qdrant(settings) -> tuple[bool, str]:
    """Check Qdrant connectivity."""
    try:
        r = httpx.get(f"http://{settings.qdrant_host}:{settings.qdrant_port}/healthz", timeout=5)
        if r.status_code == 200:
            return True, "Healthy"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:80]


def check_elasticsearch(settings) -> tuple[bool, str]:
    """Check Elasticsearch connectivity."""
    try:
        r = httpx.get(f"{settings.elasticsearch_url}/_cluster/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return True, f"Status: {data.get('status', 'unknown')}"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:80]


def check_redis(settings) -> tuple[bool, str]:
    """Check Redis connectivity."""
    try:
        client = Redis.from_url(settings.redis_url, socket_timeout=5)
        if client.ping():
            client.close()
            return True, "PONG"
        client.close()
        return False, "No PONG"
    except Exception as e:
        return False, str(e)[:80]


def check_postgres(settings) -> tuple[bool, str]:
    """Check PostgreSQL connectivity."""
    try:
        # Use sync connection for health check
        sync_url = settings.database_url.replace("postgresql+asyncpg", "postgresql")
        import psycopg2  # noqa: F811

        conn = psycopg2.connect(sync_url, connect_timeout=5)
        conn.close()
        return True, "Connected"
    except ImportError:
        # Fallback: try via httpx to pg bouncer or just report
        try:
            import asyncio
            from urllib.parse import urlparse

            import asyncpg

            parsed = urlparse(settings.database_url.replace("postgresql+asyncpg", "postgresql"))

            async def _check():
                conn = await asyncpg.connect(
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip("/"),
                    host=parsed.hostname or "localhost",
                    port=parsed.port or 5432,
                    timeout=5,
                )
                await conn.close()

            asyncio.run(_check())
            return True, "Connected"
        except Exception as e:
            return False, str(e)[:80]
    except Exception as e:
        return False, str(e)[:80]


def main():
    """Run health checks for all services and display results."""
    settings = get_settings()
    console = Console()

    table = Table(title="Faqih.ai Service Health Check", show_lines=True)
    table.add_column("Service", style="bold cyan", width=18)
    table.add_column("Status", width=10)
    table.add_column("Details", width=50)

    checks = [
        ("Neo4j", check_neo4j),
        ("Qdrant", check_qdrant),
        ("Elasticsearch", check_elasticsearch),
        ("Redis", check_redis),
        ("PostgreSQL", check_postgres),
    ]

    all_healthy = True
    for name, check_fn in checks:
        ok, detail = check_fn(settings)
        status = "[green]✓ UP[/green]" if ok else "[red]✗ DOWN[/red]"
        if not ok:
            all_healthy = False
        table.add_row(name, status, detail)

    console.print()
    console.print(table)
    console.print()

    if all_healthy:
        console.print("[bold green]All services are healthy![/bold green]")
    else:
        console.print("[bold red]Some services are down. Run: docker-compose up -d[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
