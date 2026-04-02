"""LLM client abstraction with multi-provider failover.

Supports: Google Gemini, OpenAI, Groq, OpenRouter, Cohere.
Automatically fails over to the next provider on ANY error
(quota exhaustion, timeouts, connection errors, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import google.generativeai as genai

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


# ─── Google Gemini Provider ────────────────────────────────


class GeminiProvider(LLMProvider):
    """Google Gemini via google-generativeai SDK."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__("google", model, temperature, max_tokens)
        genai.configure(api_key=api_key)
        self._model_instance = genai.GenerativeModel(model)

    @staticmethod
    def _to_gemini_format(messages: list[dict[str, str]]) -> tuple[str, list[dict]]:
        """Convert OpenAI-style messages to Gemini format."""
        system_instruction = ""
        contents = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_instruction = text
            elif role == "assistant":
                contents.append({"role": "model", "parts": [text]})
            else:
                contents.append({"role": "user", "parts": [text]})
        return system_instruction, contents

    async def complete(self, messages, temperature=None, max_tokens=None, json_mode=False) -> str:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens
        system_instruction, contents = self._to_gemini_format(messages)

        generation_config = genai.GenerationConfig(temperature=temp, max_output_tokens=tokens)
        if json_mode:
            generation_config.response_mime_type = "application/json"

        model = self._model_instance
        if system_instruction:
            model = genai.GenerativeModel(self.model, system_instruction=system_instruction)

        response = await model.generate_content_async(contents, generation_config=generation_config)
        return response.text or ""

    async def stream(self, messages, temperature=None, max_tokens=None) -> AsyncIterator[str]:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens
        system_instruction, contents = self._to_gemini_format(messages)

        generation_config = genai.GenerationConfig(temperature=temp, max_output_tokens=tokens)

        model = self._model_instance
        if system_instruction:
            model = genai.GenerativeModel(self.model, system_instruction=system_instruction)

        response = await model.generate_content_async(
            contents, generation_config=generation_config, stream=True
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text


# ─── OpenAI-Compatible Provider (OpenAI, Groq, OpenRouter) ─


class OpenAICompatibleProvider(LLMProvider):
    """Provider for any OpenAI-compatible API (OpenAI, Groq, OpenRouter)."""

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        super().__init__(name, model, temperature, max_tokens)
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if default_headers:
            kwargs["default_headers"] = default_headers

        self._client = AsyncOpenAI(**kwargs)

    async def complete(self, messages, temperature=None, max_tokens=None, json_mode=False) -> str:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def stream(self, messages, temperature=None, max_tokens=None) -> AsyncIterator[str]:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


# ─── Cohere Provider ───────────────────────────────────────


class CohereProvider(LLMProvider):
    """Cohere via cohere SDK (AsyncClientV2)."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__("cohere", model, temperature, max_tokens)
        from cohere import AsyncClientV2

        self._client = AsyncClientV2(api_key=api_key)

    async def complete(self, messages, temperature=None, max_tokens=None, json_mode=False) -> str:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                chat_messages.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "assistant":
                chat_messages.append({"role": "assistant", "content": msg["content"]})
            else:
                chat_messages.append({"role": "user", "content": msg["content"]})

        response = await self._client.chat(
            model=self.model,
            messages=chat_messages,
            temperature=temp,
            max_tokens=tokens,
        )
        return response.message.content[0].text

    async def stream(self, messages, temperature=None, max_tokens=None) -> AsyncIterator[str]:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        chat_messages = []
        for msg in messages:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})

        stream = self._client.chat_stream(
            model=self.model,
            messages=chat_messages,
            temperature=temp,
            max_tokens=tokens,
        )
        async for event in stream:
            if event.type == "content-delta":
                yield event.delta.message.content.text


# ─── Provider Factory ──────────────────────────────────────


def create_provider(
    name: str,
    api_key: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> LLMProvider:
    """Create a provider instance by name."""
    if name == "google":
        return GeminiProvider(api_key, model, temperature, max_tokens)
    elif name == "openai":
        return OpenAICompatibleProvider(name, api_key, model, temperature, max_tokens)
    elif name == "groq":
        return OpenAICompatibleProvider(
            name, api_key, model, temperature, max_tokens,
            base_url="https://api.groq.com/openai/v1",
        )
    elif name == "openrouter":
        return OpenAICompatibleProvider(
            name, api_key, model, temperature, max_tokens,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://faqih.ai", "X-Title": "Faqih.ai"},
        )
    elif name == "cohere":
        return CohereProvider(api_key, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {name}")


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
        response = await self.complete(messages, temperature=temperature, json_mode=True)
        try:
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response: %s", response[:200])
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
