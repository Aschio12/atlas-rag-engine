"""Qdrant vector store over its REST API.

Every collection has a named dense vector (`dense`, cosine) and, optionally, a named sparse
vector (`sparse`) with the IDF modifier, which is how Qdrant does BM25-style lexical scoring:
we store term frequencies, the server supplies inverse document frequency at query time.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from app.providers.base import (
    Filters,
    Point,
    Range,
    ScoredPoint,
    SparseVector,
    VectorStore,
    check_filters,
    post_json,
)
from app.providers.errors import ProviderError

DENSE = "dense"
SPARSE = "sparse"
# Payload fields with an index, and its type. Every query and delete filters on the first group;
# the rest serve the metadata filters of retrieval (Phase 3.1).
INDEXED_FIELDS = {
    "workspace_id": "keyword",
    "collection_id": "keyword",
    "document_id": "keyword",
    "kind": "keyword",
    "chunk_id": "keyword",
    "source_id": "keyword",
    "source_type": "keyword",
    "file_type": "keyword",
    "language": "keyword",
    "heading_path": "keyword",  # an array: matches when any heading in the path does
    "page": "integer",
    "created_at": "integer",
}


def _condition(key: str, cond: Any) -> dict[str, Any]:
    if isinstance(cond, Range):
        return {"key": key, "range": cond.as_dict()}
    if isinstance(cond, list):
        return {"key": key, "match": {"any": cond}}
    return {"key": key, "match": {"value": cond}}


def _filter(filters: Filters | None, exclude: Filters | None = None) -> dict | None:
    check_filters(filters)
    check_filters(exclude)
    out: dict[str, Any] = {}
    if filters:
        out["must"] = [_condition(k, v) for k, v in filters.items()]
    if exclude:
        out["must_not"] = [_condition(k, v) for k, v in exclude.items()]
    return out or None


def _point_id(raw: str) -> str:
    """Qdrant ids must be UUIDs or unsigned ints; ours are UUID strings (chunk ids)."""
    try:
        return str(uuid.UUID(raw))
    except ValueError as exc:
        raise ProviderError("qdrant", f"point id must be a UUID, got {raw!r}") from exc


class QdrantStore(VectorStore):
    name = "qdrant"

    def __init__(
        self,
        url: str,
        api_key: str = "",
        client: httpx.AsyncClient | None = None,
    ):
        self.url = url.rstrip("/")
        self._headers = {"api-key": api_key} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=30)

    async def _call(self, method: str, path: str, **kw):
        return await post_json(
            self._client, self.name, method, f"{self.url}{path}", headers=self._headers, **kw
        )

    async def ensure_collection(
        self, collection: str, dimension: int, *, sparse: bool = False
    ) -> None:
        resp = await self._call("GET", f"/collections/{collection}", ok_statuses=(404,))
        if resp.status_code == 200:
            try:
                params = resp.json()["result"]["config"]["params"]
                vectors = params["vectors"]
                size = vectors[DENSE]["size"]
            except (KeyError, TypeError) as exc:
                raise ProviderError(
                    self.name,
                    f"collection {collection!r} has an unexpected layout (created by an older "
                    "version?); re-index it",
                ) from exc
            if size != dimension:
                raise ProviderError(
                    self.name, f"collection {collection!r} has dimension {size}, not {dimension}"
                )
            if sparse and SPARSE not in (params.get("sparse_vectors") or {}):
                raise ProviderError(
                    self.name, f"collection {collection!r} was created without sparse vectors"
                )
            # Collections made before a field was indexed get the new index too.
            have = resp.json()["result"].get("payload_schema") or {}
            await self._ensure_indexes(collection, skip=set(have))
            return
        body: dict[str, Any] = {"vectors": {DENSE: {"size": dimension, "distance": "Cosine"}}}
        if sparse:
            body["sparse_vectors"] = {SPARSE: {"modifier": "idf"}}
        await self._call("PUT", f"/collections/{collection}", json=body)
        await self._ensure_indexes(collection)

    async def _ensure_indexes(self, collection: str, skip: set[str] = frozenset()) -> None:
        for field, schema in INDEXED_FIELDS.items():
            if field not in skip:
                await self._call(
                    "PUT",
                    f"/collections/{collection}/index?wait=true",
                    json={"field_name": field, "field_schema": schema},
                )

    async def drop_collection(self, collection: str) -> None:
        await self._call("DELETE", f"/collections/{collection}", ok_statuses=(404,))

    async def upsert(self, collection: str, points: list[Point]) -> None:
        if not points:
            return

        def vector(p: Point) -> dict[str, Any]:
            named: dict[str, Any] = {DENSE: p.vector}
            if p.sparse is not None:
                named[SPARSE] = {"indices": p.sparse.indices, "values": p.sparse.values}
            return named

        body = {
            "points": [
                {"id": _point_id(p.id), "vector": vector(p), "payload": p.payload} for p in points
            ]
        }
        await self._call("PUT", f"/collections/{collection}/points?wait=true", json=body)

    async def _query(
        self, collection: str, name: str, vector: Any, limit: int, filters, exclude
    ) -> list[ScoredPoint]:
        body: dict[str, Any] = {
            "vector": {"name": name, "vector": vector},
            "limit": limit,
            "with_payload": True,
        }
        if f := _filter(filters, exclude):
            body["filter"] = f
        resp = await self._call(
            "POST", f"/collections/{collection}/points/search", json=body, ok_statuses=(404,)
        )
        if resp.status_code == 404:
            return []  # nothing has been indexed into this collection yet
        try:
            return [
                ScoredPoint(str(r["id"]), float(r["score"]), r.get("payload") or {})
                for r in resp.json()["result"]
            ]
        except (KeyError, TypeError) as exc:
            raise ProviderError(self.name, "unexpected response shape") from exc

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: Filters | None = None,
        exclude: Filters | None = None,
    ) -> list[ScoredPoint]:
        return await self._query(collection, DENSE, vector, limit, filters, exclude)

    async def sparse_search(
        self,
        collection: str,
        vector: SparseVector,
        limit: int = 10,
        filters: Filters | None = None,
        exclude: Filters | None = None,
    ) -> list[ScoredPoint]:
        sparse = {"indices": vector.indices, "values": vector.values}
        return await self._query(collection, SPARSE, sparse, limit, filters, exclude)

    async def set_payload(
        self, collection: str, payload: dict[str, Any], *, filters: Filters
    ) -> None:
        if not filters:
            raise ValueError("set_payload requires filters")
        await self._call(
            "POST",
            f"/collections/{collection}/points/payload?wait=true",
            json={"payload": payload, "filter": _filter(filters)},
            ok_statuses=(404,),  # nothing indexed yet: nothing to update
        )

    async def get_vectors(self, collection: str, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        body = {"ids": [_point_id(i) for i in ids], "with_vector": [DENSE], "with_payload": False}
        resp = await self._call(
            "POST", f"/collections/{collection}/points", json=body, ok_statuses=(404,)
        )
        if resp.status_code == 404:
            return {}
        try:
            return {
                str(r["id"]): [float(x) for x in r["vector"][DENSE]] for r in resp.json()["result"]
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(self.name, "unexpected response shape") from exc

    async def count(self, collection: str, filters: Filters | None = None) -> int:
        body: dict[str, Any] = {"exact": True}
        if f := _filter(filters):
            body["filter"] = f
        resp = await self._call(
            "POST", f"/collections/{collection}/points/count", json=body, ok_statuses=(404,)
        )
        if resp.status_code == 404:
            return 0  # no such collection: nothing is stored
        try:
            return int(resp.json()["result"]["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(self.name, "unexpected response shape") from exc

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
        selector: dict[str, Any] = (
            {"points": [_point_id(i) for i in ids]}
            if ids is not None
            else {"filter": _filter(filters, exclude)}
        )
        await self._call(
            "POST",
            f"/collections/{collection}/points/delete?wait=true",
            json=selector,
            ok_statuses=(404,),  # deleting from a collection that does not exist is a no-op
        )
