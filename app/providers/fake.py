"""Deterministic, dependency-free providers for tests, CI and zero-setup demos."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import AsyncIterator
from typing import Any

from app.providers.base import (
    LLM,
    ChatMessage,
    Embedder,
    Filters,
    LLMResponse,
    Point,
    Range,
    Reranked,
    Reranker,
    ScoredPoint,
    SparseVector,
    StreamChunk,
    Usage,
    VectorStore,
    check_filters,
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


_STOP = frozenset(
    "a an the of to in on for and or is are was were be do does did what which who how why when "
    "where i my me you your it its this that with from by at as can could should would "
    "about".split()
)
_SOURCE = re.compile(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
_QUESTION_TAG = re.compile(r"<question>(.*?)</question>", re.DOTALL)


def _extractive_answer(prompt: str) -> str | None:
    """What the fake LLM says to a grounded-answer prompt: the sentences of the sources that share
    the most words with the question, each cited, or NOT_FOUND when none does. It is a stub that
    makes the whole answering path (citations, abstention, streaming) work with no model."""
    if "<sources>" not in prompt or "Question:" not in prompt:
        return None
    question = prompt.rsplit("Question:", 1)[1]
    wanted = {t for t in tokenize(question) if t not in _STOP and len(t) > 2}
    picks: list[tuple[int, str, str]] = []
    for label, text in _SOURCE.findall(prompt):
        best: tuple[int, str] | None = None
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = sentence.strip().lstrip("#-*0123456789. ").strip()
            if len(sentence.split()) < 3:
                continue
            score = len(wanted & set(tokenize(sentence)))
            if score and (best is None or score > best[0]):
                best = (score, sentence)
        if best:
            picks.append((best[0], label, best[1]))
    if not picks:
        return "NOT_FOUND The provided sources do not mention this."
    picks.sort(key=lambda p: -p[0])
    return " ".join(f"{sentence.rstrip('.')}. [{label}]" for _, label, sentence in picks[:2])


def _scripted_reply(prompt: str) -> str | None:
    """Stub replies for the pipeline's other prompts, so a zero-setup run is quiet."""
    if "Rewrite the user's latest question" in prompt and (m := _QUESTION_TAG.search(prompt)):
        return json.dumps({"query": m.group(1).strip()})  # already standalone: unchanged
    if "Reply with the updated summary only." in prompt:
        recent = prompt.split("New messages:", 1)[-1].replace(
            "Reply with the updated summary only.", ""
        )
        return " ".join(recent.split())[:400]
    return _extractive_answer(prompt)


class FakeLLM(LLM):
    """Echoes the last user message, except that it answers grounded-answer prompts extractively
    and plays along with the rewrite and summary prompts. Not an intelligence; a stand-in for
    wiring checks and zero-setup demos."""

    name = "fake"

    def __init__(self, model: str = "fake-llm", reply: str | None = None):
        self.model = model
        self._reply = reply
        self.calls: list[list[ChatMessage]] = []

    def _text(self, messages: list[ChatMessage]) -> str:
        if self._reply is not None:
            return self._reply
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        scripted = _scripted_reply(last_user)
        return scripted if scripted is not None else f"[fake-llm] You said: {last_user}"

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> LLMResponse:
        self.calls.append(messages)
        text = self._text(messages)
        prompt = sum(len(tokenize(m.content)) for m in messages)
        return LLMResponse(text, self.model, Usage(prompt, len(tokenize(text))))

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> AsyncIterator[StreamChunk]:
        response = await self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        for piece in re.findall(r"\S+\s*|\s+", response.text):
            yield StreamChunk(piece)
        yield StreamChunk(usage=response.usage)


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
    model = "fake-rerank"

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


def _one(value: Any, cond: Any) -> bool:
    """Does one payload value satisfy one condition? (Same rules as the Qdrant adapter.)"""
    if isinstance(cond, Range):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        return (
            (cond.gt is None or value > cond.gt)
            and (cond.gte is None or value >= cond.gte)
            and (cond.lt is None or value < cond.lt)
            and (cond.lte is None or value <= cond.lte)
        )
    if isinstance(cond, list):
        return value in cond
    return value == cond


def _lookup(payload: dict[str, Any], key: str) -> Any:
    """`attributes.year` reads payload["attributes"]["year"] (a missing step gives None)."""
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _satisfies(payload: dict[str, Any], key: str, cond: Any) -> bool:
    value = _lookup(payload, key)
    if isinstance(value, list):  # array payloads (heading_path) match on any element
        return any(_one(v, cond) for v in value)
    return _one(value, cond)


def _matches(payload: dict[str, Any], filters: Filters | None, exclude: Filters | None = None):
    if filters:
        check_filters(filters)
    if exclude:
        check_filters(exclude)
    return all(_satisfies(payload, k, v) for k, v in (filters or {}).items()) and not any(
        _satisfies(payload, k, v) for k, v in (exclude or {}).items()
    )


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
        filters: Filters | None = None,
        exclude: Filters | None = None,
    ) -> list[ScoredPoint]:
        hits = [
            ScoredPoint(p.id, _cosine(vector, p.vector), p.payload)
            for p in self._data.get(collection, {}).values()  # a missing collection has no hits
            if _matches(p.payload, filters, exclude)
        ]
        hits.sort(key=lambda h: (-h.score, h.id))
        return hits[:limit]

    async def sparse_search(
        self,
        collection: str,
        vector: SparseVector,
        limit: int = 10,
        filters: Filters | None = None,
        exclude: Filters | None = None,
    ) -> list[ScoredPoint]:
        points = [p for p in self._data.get(collection, {}).values() if p.sparse is not None]
        total = len(points)
        df: dict[int, int] = {}
        for p in points:
            for term in set(p.sparse.indices):
                df[term] = df.get(term, 0) + 1
        weights = dict(zip(vector.indices, vector.values, strict=True))
        hits = []
        for p in points:
            if not _matches(p.payload, filters, exclude):
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

    async def set_payload(
        self, collection: str, payload: dict[str, Any], *, filters: Filters
    ) -> None:
        if not filters:
            raise ValueError("set_payload requires filters")
        check_filters(filters)
        for point in self._data.get(collection, {}).values():
            if _matches(point.payload, filters):
                point.payload.update(payload)

    async def get_vectors(self, collection: str, ids: list[str]) -> dict[str, list[float]]:
        points = self._data.get(collection, {})
        return {i: list(points[i].vector) for i in ids if i in points}

    async def count(self, collection: str, filters: Filters | None = None) -> int:
        points = self._data.get(collection, {})
        return sum(1 for p in points.values() if _matches(p.payload, filters))

    async def delete(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        filters: Filters | None = None,
        exclude: Filters | None = None,
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
            selected = pid in ids if ids is not None else _matches(payload, filters, exclude)
            if selected:
                del store[pid]
