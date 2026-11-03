"""Deterministic, dependency-free providers for tests, CI and zero-setup demos."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from app.providers.base import (
    LLM,
    ChatMessage,
    Embedder,
    LLMResponse,
    Point,
    Reranked,
    Reranker,
    ScoredPoint,
    Usage,
    VectorStore,
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class FakeLLM(LLM):
    """Echoes the last user message. Not an intelligence; a stand-in for wiring checks."""

    name = "fake"

    def __init__(self, model: str = "fake-llm", reply: str | None = None):
        self.model = model
        self._reply = reply
        self.calls: list[list[ChatMessage]] = []

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> LLMResponse:
        self.calls.append(messages)
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        text = self._reply if self._reply is not None else f"[fake-llm] You said: {last_user}"
        prompt = sum(len(tokenize(m.content)) for m in messages)
        return LLMResponse(text, self.model, Usage(prompt, len(tokenize(text))))


class FakeEmbedder(Embedder):
    """Hashed bag-of-words embedding: texts sharing words get similar vectors."""

    name = "fake"

    def __init__(self, dimension: int = 64, model: str = "fake-embed"):
        self.dimension = dimension
        self.model = model

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "big")
            vec[h % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class FakeReranker(Reranker):
    """Scores by query-term overlap (Jaccard)."""

    name = "fake"

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[Reranked]:
        q = set(tokenize(query))
        scored = []
        for i, doc in enumerate(documents):
            d = set(tokenize(doc))
            union = q | d
            scored.append(Reranked(i, len(q & d) / len(union) if union else 0.0))
        scored.sort(key=lambda r: (-r.score, r.index))
        return scored[:top_n]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _matches(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    return all(payload.get(k) == v for k, v in (filters or {}).items())


class InMemoryVectorStore(VectorStore):
    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Point]] = {}
        self._dims: dict[str, int] = {}

    async def ensure_collection(self, collection: str, dimension: int) -> None:
        existing = self._dims.get(collection)
        if existing is not None and existing != dimension:
            raise ValueError(f"collection {collection!r} has dimension {existing}, not {dimension}")
        self._dims[collection] = dimension
        self._data.setdefault(collection, {})

    async def upsert(self, collection: str, points: list[Point]) -> None:
        if collection not in self._data:
            raise KeyError(f"unknown collection {collection!r}")
        for p in points:
            if len(p.vector) != self._dims[collection]:
                raise ValueError("vector dimension mismatch")
            self._data[collection][p.id] = p

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredPoint]:
        hits = [
            ScoredPoint(p.id, _cosine(vector, p.vector), p.payload)
            for p in self._data[collection].values()
            if _matches(p.payload, filters)
        ]
        hits.sort(key=lambda h: (-h.score, h.id))
        return hits[:limit]

    async def delete(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> None:
        if ids is None and not filters:
            raise ValueError("delete requires ids or filters")
        store = self._data[collection]
        for pid in list(store):
            if (ids is not None and pid in ids) or (
                filters and _matches(store[pid].payload, filters)
            ):
                del store[pid]
