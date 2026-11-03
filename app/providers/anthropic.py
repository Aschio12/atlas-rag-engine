"""Anthropic Messages API adapter."""

from __future__ import annotations

import httpx

from app.providers.base import LLM, ChatMessage, LLMResponse, Usage, post_json
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

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> LLMResponse:
        # The Messages API takes the system prompt as a separate field.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": turns,
        }
        if system:
            body["system"] = system
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
            text, self.model, Usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        )
