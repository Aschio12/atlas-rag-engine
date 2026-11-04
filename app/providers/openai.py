"""OpenAI-compatible adapters. Works with OpenAI and any server exposing the same REST API
(vLLM, LM Studio, OpenRouter, ...) by changing base_url."""

from __future__ import annotations

import httpx

from app.providers.base import LLM, ChatMessage, Embedder, LLMResponse, Usage, post_json
from app.providers.errors import ProviderError


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


class OpenAILLM(LLM):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
    ):
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=120)

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = await post_json(
            self._client,
            self.name,
            "POST",
            f"{self.base_url}/chat/completions",
            json=body,
            headers=_headers(self._api_key),
        )
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(self.name, "unexpected response shape") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            text,
            self.model,
            Usage(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            ),
        )


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
    ):
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimension: int | None = None
        self._client = client or httpx.AsyncClient(timeout=120)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await post_json(
            self._client,
            self.name,
            "POST",
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            headers=_headers(self._api_key),
        )
        try:
            rows = sorted(resp.json()["data"], key=lambda r: r["index"])
            vectors = [r["embedding"] for r in rows]
        except (KeyError, TypeError) as exc:
            raise ProviderError(self.name, "unexpected response shape") from exc
        if len(vectors) != len(texts):
            raise ProviderError(self.name, "embedding count does not match input count")
        self.dimension = len(vectors[0])
        return vectors
