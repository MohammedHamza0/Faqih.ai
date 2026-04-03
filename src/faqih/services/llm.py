"""LLM client abstraction with Ollama backend.

Uses local Ollama models (qwen3:8b, llama3.2, etc.) via the
native Ollama REST API at http://localhost:11434.
Supports automatic failover between local models.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

# ─── Error log file ───────────────────────────────────────

LOG_DIR = Path("logs")
LLM_ERROR_LOG = LOG_DIR / "llm_errors.log"


class LLMExhaustedError(Exception):
    """Raised when all LLM providers have failed."""

    def __init__(self, query: str, errors: list[tuple[str, Exception]]):
        self.query = query
        self.errors = errors
        provider_summary = ", ".join(f"{name}: {type(e).__name__}" for name, e in errors)
        super().__init__(f"All LLM providers failed — {provider_summary}")


def _log_exhaustion_to_file(query: str, errors: list[tuple[str, Exception]]):
    """Log all-providers-exhausted failures to a persistent log file."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(LLM_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"QUERY: {query[:500]}\n")
            f.write(f"PROVIDERS TRIED: {len(errors)}\n")
            for name, error in errors:
                f.write(f"  - {name}: [{type(error).__name__}] {str(error)[:300]}\n")
            f.write(f"{'='*80}\n")
    except Exception as log_err:
        logger.warning("Failed to write to LLM error log file: %s", log_err)


# ─── Abstract Provider ─────────────────────────────────────


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, model: str, temperature: float, max_tokens: int):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Get a complete response from the LLM."""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens from the LLM."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


# ─── Ollama Provider ───────────────────────────────────────


class OllamaProvider(LLMProvider):
    """Local Ollama provider via native REST API (/api/chat)."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(f"ollama/{model}", model, temperature, max_tokens)
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=None)

    async def complete(
        self, messages, temperature=None, max_tokens=None, json_mode=False
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    async def stream(
        self, messages, temperature=None, max_tokens=None
    ) -> AsyncIterator[str]:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temp,
                "num_predict": tokens,
            },
        }

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue


# ─── Provider Factory ──────────────────────────────────────


def create_provider(
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    base_url: str = "http://localhost:11434",
) -> LLMProvider:
    """Create an Ollama provider instance for the given model."""
    return OllamaProvider(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )


# ─── LLM Client with Failover ─────────────────────────────


class LLMClient:
    """
    Unified async LLM client with automatic failover.

    Tries the primary provider first, then falls back to each provider
    in the fallback chain on ANY error. If all providers fail, logs the
    failure to logs/llm_errors.log and raises LLMExhaustedError.
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: list[LLMProvider] | None = None,
    ):
        self._primary = primary
        self._fallbacks = fallbacks or []
        self._providers = [self._primary] + self._fallbacks
        logger.info(
            "LLMClient initialized: primary=%s, fallbacks=%s",
            primary.name,
            [p.name for p in self._fallbacks],
        )

    @property
    def provider_names(self) -> list[str]:
        """List of all provider names in order."""
        return [p.name for p in self._providers]

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Get a complete response, with automatic failover on ANY error."""
        errors: list[tuple[str, Exception]] = []
        query_text = next((m["content"] for m in messages if m["role"] == "user"), "")

        for provider in self._providers:
            try:
                result = await provider.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
                return result
            except Exception as e:
                errors.append((provider.name, e))
                logger.warning(
                    "Provider '%s' failed: [%s] %s. Trying next...",
                    provider.name, type(e).__name__, str(e)[:200],
                )

        # All providers exhausted — log to file and raise
        _log_exhaustion_to_file(query_text, errors)
        logger.error("All %d LLM providers exhausted.", len(errors))
        raise LLMExhaustedError(query_text, errors)

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict:
        """Get a JSON response from the LLM with failover."""
        errors: list[tuple[str, Exception]] = []
        query_text = next((m["content"] for m in messages if m["role"] == "user"), "")

        for provider in self._providers:
            try:
                response = await provider.complete(
                    messages,
                    temperature=temperature,
                    json_mode=True,
                )
                text = response.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    text = "\n".join(lines)
                
                # Check for empty response to avoid obscure json errors
                if not text:
                    raise ValueError("Received empty response from provider")
                    
                parsed_json = json.loads(text)
                return parsed_json
            except Exception as e:
                errors.append((provider.name, e))
                logger.warning(
                    "Provider '%s' failed in JSON complete: [%s] %s. Trying next...",
                    provider.name, type(e).__name__, str(e)[:200],
                )

        # All providers exhausted
        _log_exhaustion_to_file(query_text, errors)
        logger.error("All %d LLM providers exhausted for complete_json.", len(errors))
        return {}

    async def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens with automatic failover on ANY error."""
        errors: list[tuple[str, Exception]] = []
        query_text = next((m["content"] for m in messages if m["role"] == "user"), "")

        for provider in self._providers:
            try:
                async for token in provider.stream(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield token
                return  # Stream completed successfully
            except Exception as e:
                errors.append((provider.name, e))
                logger.warning(
                    "Provider '%s' stream failed: [%s] %s. Trying next...",
                    provider.name, type(e).__name__, str(e)[:200],
                )

        _log_exhaustion_to_file(query_text, errors)
        logger.error("All %d LLM providers exhausted for streaming.", len(errors))
        raise LLMExhaustedError(query_text, errors)
