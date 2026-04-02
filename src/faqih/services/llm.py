"""LLM client abstraction for Google Gemini, OpenAI, and Anthropic."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import google.generativeai as genai

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified async LLM client supporting Google Gemini, OpenAI, and Anthropic.

    Default provider is Google Gemini (via Google AI Studio).
    """

    def __init__(
        self,
        provider: str = "google",
        model: str = "gemini-2.0-flash",
        api_key: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        if provider == "google":
            genai.configure(api_key=api_key)
            self._gemini = genai.GenerativeModel(model)
            self._openai = None
            self._anthropic = None
        elif provider == "openai":
            from openai import AsyncOpenAI
            self._openai = AsyncOpenAI(api_key=api_key)
            self._gemini = None
            self._anthropic = None
        elif provider == "anthropic":
            from anthropic import AsyncAnthropic
            self._anthropic = AsyncAnthropic(api_key=api_key)
            self._gemini = None
            self._openai = None
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Get a complete response from the LLM."""
        temp = temperature if temperature is not None else self._temperature
        tokens = max_tokens or self._max_tokens

        if self._provider == "google":
            # Convert messages to Gemini format
            system_instruction, contents = self._to_gemini_format(messages)

            generation_config = genai.GenerationConfig(
                temperature=temp,
                max_output_tokens=tokens,
            )
            if json_mode:
                generation_config.response_mime_type = "application/json"

            # Use system instruction if present
            model = self._gemini
            if system_instruction:
                model = genai.GenerativeModel(
                    self._model,
                    system_instruction=system_instruction,
                )

            response = await model.generate_content_async(
                contents,
                generation_config=generation_config,
            )
            return response.text or ""

        elif self._provider == "openai":
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = await self._openai.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""

        elif self._provider == "anthropic":
            system_msg = ""
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    chat_messages.append(msg)

            response = await self._anthropic.messages.create(
                model=self._model,
                max_tokens=tokens,
                temperature=temp,
                system=system_msg,
                messages=chat_messages,
            )
            return response.content[0].text

        return ""

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict:
        """Get a JSON response from the LLM."""
        response = await self.complete(messages, temperature=temperature, json_mode=True)
        try:
            # Handle markdown-wrapped JSON (some models wrap in ```json ... ```)
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
        """Stream response tokens from the LLM."""
        temp = temperature if temperature is not None else self._temperature
        tokens = max_tokens or self._max_tokens

        if self._provider == "google":
            system_instruction, contents = self._to_gemini_format(messages)

            generation_config = genai.GenerationConfig(
                temperature=temp,
                max_output_tokens=tokens,
            )

            model = self._gemini
            if system_instruction:
                model = genai.GenerativeModel(
                    self._model,
                    system_instruction=system_instruction,
                )

            response = await model.generate_content_async(
                contents,
                generation_config=generation_config,
                stream=True,
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        elif self._provider == "openai":
            stream = await self._openai.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

        elif self._provider == "anthropic":
            system_msg = ""
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    chat_messages.append(msg)

            async with self._anthropic.messages.stream(
                model=self._model,
                max_tokens=tokens,
                temperature=temp,
                system=system_msg,
                messages=chat_messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

    @staticmethod
    def _to_gemini_format(
        messages: list[dict[str, str]],
    ) -> tuple[str, list[dict]]:
        """
        Convert OpenAI-style messages to Gemini format.

        Returns (system_instruction, contents) where contents is
        a list of {"role": "user"|"model", "parts": [text]}.
        """
        system_instruction = ""
        contents = []

        for msg in messages:
            role = msg["role"]
            text = msg["content"]

            if role == "system":
                system_instruction = text
            elif role == "assistant":
                contents.append({"role": "model", "parts": [text]})
            else:  # "user"
                contents.append({"role": "user", "parts": [text]})

        return system_instruction, contents
