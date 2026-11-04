"""Candidate search: asking the vector store for the points nearest a query (Phases 3.1, 3.2).

A *channel* is one way of searching with one piece of query text: dense (embedding similarity)
or sparse (BM25-style term overlap). Every channel searches each collection in its own vector
collection, with the workspace and collection scope applied after the caller's filters so
nothing a caller sends can widen it.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from app.db.models import Collection
from app.ingest.sparse import encode_query
from app.ingest.vectors import vector_collection
from app.providers.base import Filters, ScoredPoint, VectorStore
from app.retrieve.filters import CompiledFilter

OVERFETCH = 2  # points per wanted candidate: question points and dropped points share chunks
MAX_FETCH = 200


@dataclass(frozen=True)
class Candidate:
    point: ScoredPoint
    collection: Collection


def fetch_size(pool: int) -> int:
    return min(pool * OVERFETCH, MAX_FETCH)


def scope_filters(
    metadata: CompiledFilter, workspace_id: uuid.UUID, collection_id: uuid.UUID
) -> Filters:
    # The scope is applied last so nothing in the caller's filters can widen it.
    return {
        **metadata.filters,
        "workspace_id": str(workspace_id),
        "collection_id": str(collection_id),
    }


def _merge(found: list[list[Candidate]]) -> list[Candidate]:
    merged = [c for part in found for c in part]
    merged.sort(key=lambda c: (-c.point.score, c.point.id))
    return merged


async def search_dense(
    store: VectorStore,
    collections: list[Collection],
    vectors: dict[str, list[float]],  # embedding model -> query vector
    workspace_id: uuid.UUID,
    metadata: CompiledFilter,
    limit: int,
) -> list[Candidate]:
    async def one(col: Collection) -> list[Candidate]:
        found = await store.search(
            vector_collection(col.id),
            vectors[col.embed_model],
            limit,
            filters=scope_filters(metadata, workspace_id, col.id),
            exclude=metadata.exclude or None,
        )
        return [Candidate(p, col) for p in found]

    return _merge(await asyncio.gather(*(one(c) for c in collections)))


async def search_sparse(
    store: VectorStore,
    collections: list[Collection],
    text: str,
    workspace_id: uuid.UUID,
    metadata: CompiledFilter,
    limit: int,
) -> list[Candidate]:
    query = encode_query(text)
    if not query.indices:  # nothing but stopword-like noise: there is nothing to match
        return []

    async def one(col: Collection) -> list[Candidate]:
        found = await store.sparse_search(
            vector_collection(col.id),
            query,
            limit,
            filters=scope_filters(metadata, workspace_id, col.id),
            exclude=metadata.exclude or None,
        )
        return [Candidate(p, col) for p in found]

    return _merge(await asyncio.gather(*(one(c) for c in collections)))
