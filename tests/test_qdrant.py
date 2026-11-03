import json
import uuid

import httpx
import pytest

from app.providers.base import Point
from app.providers.errors import ProviderError
from app.providers.qdrant import QdrantStore

ID = str(uuid.uuid4())


def store(handler, key="") -> QdrantStore:
    return QdrantStore(
        "http://qdrant:6333/", key, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def info(size):
    return {"result": {"config": {"params": {"vectors": {"size": size}}}}}


async def test_ensure_collection_creates_when_missing():
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path, json.loads(req.content) if req.content else None))
        return httpx.Response(404 if req.method == "GET" else 200, json={"result": True})

    await store(handler).ensure_collection("chunks", 384)
    assert calls[1] == (
        "PUT",
        "/collections/chunks",
        {"vectors": {"size": 384, "distance": "Cosine"}},
    )


async def test_ensure_collection_noop_when_dimension_matches_and_errors_when_not():
    ok = store(lambda r: httpx.Response(200, json=info(384)))
    await ok.ensure_collection("chunks", 384)
    with pytest.raises(ProviderError, match="dimension"):
        await ok.ensure_collection("chunks", 768)


async def test_upsert_sends_points_and_empty_is_noop():
    seen = []

    def handler(req):
        seen.append((req.method, str(req.url), json.loads(req.content)))
        return httpx.Response(200, json={})

    s = store(handler)
    await s.upsert("c", [])
    assert seen == []
    await s.upsert("c", [Point(ID, [0.1, 0.2], {"ws": "a"})])
    method, url, body = seen[0]
    assert method == "PUT" and url.endswith("/collections/c/points?wait=true")
    assert body["points"][0] == {"id": ID, "vector": [0.1, 0.2], "payload": {"ws": "a"}}


async def test_upsert_rejects_non_uuid_ids():
    with pytest.raises(ProviderError):
        await store(lambda r: httpx.Response(200, json={})).upsert("c", [Point("chunk-1", [1.0])])


async def test_search_builds_filter_and_parses_hits():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"result": [{"id": ID, "score": 0.9, "payload": {"k": 1}}]})

    hits = await store(handler).search("c", [1.0, 0.0], limit=3, filters={"workspace_id": "w1"})
    assert seen["body"]["filter"] == {"must": [{"key": "workspace_id", "match": {"value": "w1"}}]}
    assert seen["body"]["limit"] == 3
    assert (hits[0].id, hits[0].score, hits[0].payload) == (ID, 0.9, {"k": 1})


async def test_delete_by_ids_filters_and_guard():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={})

    s = store(handler)
    await s.delete("c", ids=[ID])
    await s.delete("c", filters={"document_id": "d1"})
    assert seen[0] == {"points": [ID]}
    assert seen[1] == {"filter": {"must": [{"key": "document_id", "match": {"value": "d1"}}]}}
    with pytest.raises(ValueError):
        await s.delete("c")


async def test_api_key_header_only_when_configured():
    seen = {}

    def handler(req):
        seen["h"] = dict(req.headers)
        return httpx.Response(200, json={"result": []})

    await store(handler, key="qk").search("c", [1.0])
    assert seen["h"]["api-key"] == "qk"
    await store(handler).search("c", [1.0])
    assert "api-key" not in seen["h"]


async def test_drop_collection_tolerates_missing():
    seen = {}

    def handler(req):
        seen["m"], seen["p"] = req.method, req.url.path
        return httpx.Response(404, json={})

    await store(handler).drop_collection("c")
    assert (seen["m"], seen["p"]) == ("DELETE", "/collections/c")
