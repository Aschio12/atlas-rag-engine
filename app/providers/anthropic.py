"""Anthropic Messages API adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.providers.base import (
    LLM,
    ChatMessage,
    LLMResponse,
    StreamChunk,
    Usage,
    post_json,
    stream_lines,
)
from app.providers.errors import ProviderError

API_VERSION = "2023-06-01"


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
    ):
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=120)

    def _body(self, messages: list[ChatMessage], temperature: float, max_tokens: int) -> dict:
        # The Messages API takes the system prompt as a separate field.
        system_messages = [m for m in messages if m.role == "system"]
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": turns,
        }
        if any(m.cache for m in system_messages):
            # Block form lets us attach cache_control to the shared prefix (prompt caching).
            body["system"] = [
                {"type": "text", "text": m.content}
                | ({"cache_control": {"type": "ephemeral"}} if m.cache else {})
                for m in system_messages
            ]
        elif system_messages:
            body["system"] = "\n\n".join(m.content for m in system_messages)
        return body

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> LLMResponse:
        body = self._body(messages, temperature, max_tokens)
        resp = await post_json(
            self._client,
            self.name,
            "POST",
            f"{self.base_url}/v1/messages",
            json=body,
            headers={"x-api-key": self._api_key, "anthropic-version": API_VERSION},
        )
        data = resp.json()
        try:
            text = "".join(b["text"] for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise ProviderError(self.name, "unexpected response shape") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            text,
            self.model,
            Usage(
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cache_read_input_tokens", 0),
            ),
        )

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> AsyncIterator[StreamChunk]:
        body = {**self._body(messages, temperature, max_tokens), "stream": True}
        prompt = cached = completion = 0
        async for line in stream_lines(
            self._client,
            self.name,
            f"{self.base_url}/v1/messages",
            json=body,
            headers={"x-api-key": self._api_key, "anthropic-version": API_VERSION},
        ):
            if not line.startswith("data:"):
                continue  # `event:` names repeat the type inside the data
            try:
                event = json.loads(line[5:])
            except ValueError as exc:
                raise ProviderError(self.name, "unexpected stream data") from exc
            kind = event.get("type")
            if kind == "message_start":
                usage = (event.get("message") or {}).get("usage") or {}
                prompt = usage.get("input_tokens", 0)
                cached = usage.get("cache_read_input_tokens", 0)
                completion = usage.get("output_tokens", 0)
            elif kind == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield StreamChunk(delta["text"])
            elif kind == "message_delta":
                completion = (event.get("usage") or {}).get("output_tokens", completion)
            elif kind == "error":
                raise ProviderError(self.name, f"stream error: {str(event.get('error'))[:200]}")
            elif kind == "message_stop":
                break
        yield StreamChunk(usage=Usage(prompt, completion, cached))
