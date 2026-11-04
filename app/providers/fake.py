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
    SparseVector,
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
    """Dense cosine search plus IDF-weighted sparse search, all in process memory."""

    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Point]] = {}
        self._dims: dict[str, int] = {}
        self._sparse: dict[str, bool] = {}

    async def ensure_collection(
        self, collection: str, dimension: int, *, sparse: bool = False
    ) -> None:
        existing = self._dims.get(collection)
        if existing is not None and existing != dimension:
            raise ValueError(f"collection {collection!r} has dimension {existing}, not {dimension}")
        self._dims[collection] = dimension
        self._sparse[collection] = self._sparse.get(collection, False) or sparse
        self._data.setdefault(collection, {})

    async def drop_collection(self, collection: str) -> None:
        self._data.pop(collection, None)
        self._dims.pop(collection, None)
        self._sparse.pop(collection, None)

    async def upsert(self, collection: str, points: list[Point]) -> None:
        if collection not in self._data:
            raise KeyError(f"unknown collection {collection!r}")
        for p in points:
            if len(p.vector) != self._dims[collection]:
                raise ValueError("vector dimension mismatch")
            if p.sparse is not None and not self._sparse[collection]:
                raise ValueError("collection was created without sparse vectors")
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

    async def sparse_search(
        self,
        collection: str,
        vector: SparseVector,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredPoint]:
        points = [p for p in self._data[collection].values() if p.sparse is not None]
        total = len(points)
        df: dict[int, int] = {}
        for p in points:
            for term in set(p.sparse.indices):
                df[term] = df.get(term, 0) + 1
        weights = dict(zip(vector.indices, vector.values, strict=True))
        hits = []
        for p in points:
            if not _matches(p.payload, filters):
                continue
            score = 0.0
            for term, value in zip(p.sparse.indices, p.sparse.values, strict=True):
                if term in weights:
                    n = df[term]  # the same IDF form Qdrant applies with its idf modifier
                    score += weights[term] * value * math.log(1 + (total - n + 0.5) / (n + 0.5))
            if score > 0:
                hits.append(ScoredPoint(p.id, score, p.payload))
        hits.sort(key=lambda h: (-h.score, h.id))
        return hits[:limit]

    async def count(self, collection: str, filters: dict[str, Any] | None = None) -> int:
        points = self._data.get(collection, {})
        return sum(1 for p in points.values() if _matches(p.payload, filters))

    async def delete(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
    ) -> None:
        if ids is None and not filters:
            raise ValueError("delete requires ids or filters")
        if ids is not None and (filters or exclude):
            raise ValueError("ids cannot be combined with filters or exclude")
        store = self._data.get(collection)
        if store is None:
            return  # no such collection: nothing to delete
        for pid in list(store):
            payload = store[pid].payload
            selected = pid in ids if ids is not None else _matches(payload, filters)
            kept = bool(exclude) and any(payload.get(k) == v for k, v in exclude.items())
            if selected and not kept:
                del store[pid]
