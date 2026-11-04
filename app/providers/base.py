"""Provider interfaces and shared value types (Phase 1.5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from app.providers.errors import ProviderError


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str
    # Mark a large, reusable prefix (e.g. a whole document that many requests share) so providers
    # that support prompt caching can bill it once. Others ignore the flag.
    cache: bool = False


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens served from the provider's prompt cache


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class Point:
    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredPoint:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Reranked:
    index: int  # position in the input list
    score: float


class LLM(ABC):
    name: str
    model: str

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


class Embedder(ABC):
    name: str
    model: str
    dimension: int | None  # None until known (some providers reveal it on first call)

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in order."""


class Reranker(ABC):
    name: str

    @abstractmethod
    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[Reranked]:
        """Return up to top_n documents ordered by descending relevance."""


class VectorStore(ABC):
    name: str

    @abstractmethod
    async def ensure_collection(self, collection: str, dimension: int) -> None: ...

    @abstractmethod
    async def upsert(self, collection: str, points: list[Point]) -> None: ...

    @abstractmethod
    async def drop_collection(self, collection: str) -> None: ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredPoint]:
        """Cosine-similarity search. `filters` are exact-match payload conditions (AND)."""

    @abstractmethod
    async def delete(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> None: ...


async def post_json(
    client: httpx.AsyncClient,
    provider: str,
    method: str,
    url: str,
    *,
    json: dict | None = None,
    headers: dict[str, str] | None = None,
    ok_statuses: tuple[int, ...] = (),
) -> httpx.Response:
    """Send a request, converting transport and HTTP failures into ProviderError."""
    try:
        resp = await client.request(method, url, json=json, headers=headers)
    except httpx.HTTPError as exc:
        raise ProviderError(provider, f"request failed: {type(exc).__name__}") from exc
    if resp.status_code >= 400 and resp.status_code not in ok_statuses:
        raise ProviderError(
            provider, f"HTTP {resp.status_code}: {resp.text[:200]}", resp.status_code
        )
    return resp
