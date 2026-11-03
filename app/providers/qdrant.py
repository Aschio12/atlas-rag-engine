"""Qdrant vector store over its REST API (dense cosine search; sparse/hybrid arrives in Phase 3)."""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from app.providers.base import Point, ScoredPoint, VectorStore, post_json
from app.providers.errors import ProviderError


def _filter(filters: dict[str, Any] | None) -> dict | None:
    if not filters:
        return None
    return {"must": [{"key": k, "match": {"value": v}} for k, v in filters.items()]}


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

    async def ensure_collection(self, collection: str, dimension: int) -> None:
        resp = await self._call("GET", f"/collections/{collection}", ok_statuses=(404,))
        if resp.status_code == 200:
            size = resp.json()["result"]["config"]["params"]["vectors"]["size"]
            if size != dimension:
                raise ProviderError(
                    self.name, f"collection {collection!r} has dimension {size}, not {dimension}"
                )
            return
        await self._call(
            "PUT",
            f"/collections/{collection}",
            json={"vectors": {"size": dimension, "distance": "Cosine"}},
        )

    async def upsert(self, collection: str, points: list[Point]) -> None:
        if not points:
            return
        body = {
            "points": [
                {"id": _point_id(p.id), "vector": p.vector, "payload": p.payload} for p in points
            ]
        }
        await self._call("PUT", f"/collections/{collection}/points?wait=true", json=body)

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredPoint]:
        body: dict[str, Any] = {"vector": vector, "limit": limit, "with_payload": True}
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

    async def delete(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> None:
        if ids is None and not filters:
            raise ValueError("delete requires ids or filters")
        selector: dict[str, Any] = (
            {"points": [_point_id(i) for i in ids]}
            if ids is not None
            else {"filter": _filter(filters)}
        )
        await self._call(
            "POST", f"/collections/{collection}/points/delete?wait=true", json=selector
        )
