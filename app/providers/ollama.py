"""Ollama adapters (local models, no API key)."""

from __future__ import annotations

import httpx

from app.providers.base import LLM, ChatMessage, Embedder, LLMResponse, Usage, post_json
from app.providers.errors import ProviderError


class OllamaLLM(LLM):
    name = "ollama"

    def __init__(self, base_url: str, model: str, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client or httpx.AsyncClient(timeout=120)

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> LLMResponse:
        body = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = (
            await post_json(self._client, self.name, "POST", f"{self.base_url}/api/chat", json=body)
        ).json()
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError(self.name, "unexpected response shape") from exc
        return LLMResponse(
            text, self.model, Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        )


class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(self, base_url: str, model: str, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimension: int | None = None
        self._client = client or httpx.AsyncClient(timeout=120)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        body = {"model": self.model, "input": texts}
        data = (
            await post_json(
                self._client, self.name, "POST", f"{self.base_url}/api/embed", json=body
            )
        ).json()
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ProviderError(self.name, "embedding count does not match input count")
        self.dimension = len(vectors[0])
        return vectors
