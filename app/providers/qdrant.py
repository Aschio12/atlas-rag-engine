"""Qdrant vector store over its REST API.

Every collection has a named dense vector (`dense`, cosine) and, optionally, a named sparse
vector (`sparse`) with the IDF modifier, which is how Qdrant does BM25-style lexical scoring:
we store term frequencies, the server supplies inverse document frequency at query time.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from app.providers.base import Point, ScoredPoint, SparseVector, VectorStore, post_json
from app.providers.errors import ProviderError

DENSE = "dense"
SPARSE = "sparse"
INDEXED_FIELDS = ("workspace_id", "collection_id", "document_id", "kind", "chunk_id")


def _filter(filters: dict[str, Any] | None, exclude: dict[str, Any] | None = None) -> dict | None:
    out: dict[str, Any] = {}
    if filters:
        out["must"] = [{"key": k, "match": {"value": v}} for k, v in filters.items()]
    if exclude:
        out["must_not"] = [{"key": k, "match": {"value": v}} for k, v in exclude.items()]
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
            return
        body: dict[str, Any] = {"vectors": {DENSE: {"size": dimension, "distance": "Cosine"}}}
        if sparse:
            body["sparse_vectors"] = {SPARSE: {"modifier": "idf"}}
        await self._call("PUT", f"/collections/{collection}", json=body)
        for field in INDEXED_FIELDS:  # filters on these are used by every query and delete
            await self._call(
                "PUT",
                f"/collections/{collection}/index?wait=true",
                json={"field_name": field, "field_schema": "keyword"},
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

    async def _query(self, collection: str, name: str, vector: Any, limit: int, filters):
        body: dict[str, Any] = {
            "vector": {"name": name, "vector": vector},
            "limit": limit,
            "with_payload": True,
        }
        if f := _filter(filters):
            body["filter"] = f
        resp = await self._call("POST", f"/collections/{collection}/points/search", json=body)
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
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredPoint]:
        return await self._query(collection, DENSE, vector, limit, filters)

    async def sparse_search(
        self,
        collection: str,
        vector: SparseVector,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredPoint]:
        sparse = {"indices": vector.indices, "values": vector.values}
        return await self._query(collection, SPARSE, sparse, limit, filters)

    async def count(self, collection: str, filters: dict[str, Any] | None = None) -> int:
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
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
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
