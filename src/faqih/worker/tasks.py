"""Celery worker task definitions."""

from __future__ import annotations

from faqih.cache.prefetch import celery_app, prefetch_related_masail

# Re-export for celery worker discovery
__all__ = ["celery_app", "prefetch_related_masail"]
