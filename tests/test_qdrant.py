import json
import uuid

import httpx
import pytest

from app.providers.base import Point, Range, SparseVector
from app.providers.errors import ProviderError
from app.providers.qdrant import INDEXED_FIELDS, QdrantStore

ID = str(uuid.uuid4())


def store(handler, key="") -> QdrantStore:
    return QdrantStore(
        "http://qdrant:6333/", key, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def info(size, sparse=False):
    params = {"vectors": {"dense": {"size": size}}}
    if sparse:
        params["sparse_vectors"] = {"sparse": {"modifier": "idf"}}
    return {"result": {"config": {"params": params}}}


async def test_ensure_collection_creates_when_missing():
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path, json.loads(req.content) if req.content else None))
        return httpx.Response(404 if req.method == "GET" else 200, json={"result": True})

    await store(handler).ensure_collection("chunks", 384)
    assert calls[1] == (
        "PUT",
        "/collections/chunks",
        {"vectors": {"dense": {"size": 384, "distance": "Cosine"}}},
    )
    indexed = {c[2]["field_name"] for c in calls[2:] if c[1].endswith("/index")}
    assert indexed == set(INDEXED_FIELDS)
    schemas = {c[2]["field_name"]: c[2]["field_schema"] for c in calls[2:]}
    assert schemas["document_id"] == "keyword" and schemas["language"] == "keyword"
    assert schemas["page"] == "integer" and schemas["created_at"] == "integer"


async def test_ensure_collection_with_sparse_vectors_uses_the_idf_modifier():
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path, json.loads(req.content) if req.content else None))
        return httpx.Response(404 if req.method == "GET" else 200, json={"result": True})

    await store(handler).ensure_collection("chunks", 8, sparse=True)
    assert calls[1][2]["sparse_vectors"] == {"sparse": {"modifier": "idf"}}


async def test_ensure_collection_requires_matching_sparse_support_and_layout():
    with_sparse = store(lambda r: httpx.Response(200, json=info(8, sparse=True)))
    await with_sparse.ensure_collection("c", 8, sparse=True)
    without = store(lambda r: httpx.Response(200, json=info(8)))
    await without.ensure_collection("c", 8)
    with pytest.raises(ProviderError, match="without sparse"):
        await without.ensure_collection("c", 8, sparse=True)
    legacy = store(
        lambda r: httpx.Response(
            200, json={"result": {"config": {"params": {"vectors": {"size": 8}}}}}
        )
    )
    with pytest.raises(ProviderError, match="older version"):
        await legacy.ensure_collection("c", 8)


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
    assert body["points"][0] == {"id": ID, "vector": {"dense": [0.1, 0.2]}, "payload": {"ws": "a"}}


async def test_upsert_sends_the_sparse_vector_when_present():
    seen = []
    s = store(lambda req: seen.append(json.loads(req.content)) or httpx.Response(200, json={}))
    await s.upsert("c", [Point(ID, [0.1], {}, SparseVector([5, 9], [1.5, 0.5]))])
    assert seen[0]["points"][0]["vector"] == {
        "dense": [0.1],
        "sparse": {"indices": [5, 9], "values": [1.5, 0.5]},
    }


async def test_upsert_rejects_non_uuid_ids():
    with pytest.raises(ProviderError):
        await store(lambda r: httpx.Response(200, json={})).upsert("c", [Point("chunk-1", [1.0])])


async def test_search_builds_filter_and_parses_hits():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"result": [{"id": ID, "score": 0.9, "payload": {"k": 1}}]})

    hits = await store(handler).search("c", [1.0, 0.0], limit=3, filters={"workspace_id": "w1"})
    assert seen["body"]["vector"] == {"name": "dense", "vector": [1.0, 0.0]}
    assert seen["body"]["filter"] == {"must": [{"key": "workspace_id", "match": {"value": "w1"}}]}
    assert seen["body"]["limit"] == 3
    assert (hits[0].id, hits[0].score, hits[0].payload) == (ID, 0.9, {"k": 1})


async def test_sparse_search_queries_the_named_sparse_vector():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"result": [{"id": ID, "score": 2.5, "payload": {}}]})

    hits = await store(handler).sparse_search("c", SparseVector([3, 4], [1.0, 2.0]), limit=5)
    assert seen["body"]["vector"] == {
        "name": "sparse",
        "vector": {"indices": [3, 4], "values": [1.0, 2.0]},
    }
    assert hits[0].score == 2.5


async def test_count_is_exact_and_filtered():
    seen = {}

    def handler(req):
        seen["path"], seen["body"] = req.url.path, json.loads(req.content)
        return httpx.Response(200, json={"result": {"count": 7}})

    assert await store(handler).count("c", {"document_id": "d1"}) == 7
    assert seen["path"] == "/collections/c/points/count"
    assert seen["body"]["exact"] is True
    assert seen["body"]["filter"]["must"][0]["key"] == "document_id"
    assert await store(lambda r: httpx.Response(200, json={"result": {"count": 0}})).count("c") == 0
    with pytest.raises(ProviderError):
        await store(lambda r: httpx.Response(200, json={"nope": 1})).count("c")


async def test_delete_can_spare_points_with_must_not():
    seen = []
    s = store(lambda req: seen.append(json.loads(req.content)) or httpx.Response(200, json={}))
    await s.delete("c", filters={"document_id": "d1"}, exclude={"index_version": "v2"})
    assert seen[0]["filter"] == {
        "must": [{"key": "document_id", "match": {"value": "d1"}}],
        "must_not": [{"key": "index_version", "match": {"value": "v2"}}],
    }
    with pytest.raises(ValueError):
        await s.delete("c", ids=[ID], filters={"a": 1})


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


def body_of(req):
    return json.loads(req.content)


async def test_search_translates_any_of_range_and_exclusion_filters():
    seen = {}

    def handler(req):
        seen.update(body_of(req))
        return httpx.Response(200, json={"result": []})

    await store(handler).search(
        "c",
        [1.0, 0.0],
        5,
        filters={
            "workspace_id": "w",
            "language": ["en", "de"],
            "page": Range(gte=2, lt=9),
            "heading_path": "Intro",
        },
        exclude={"document_id": "d1", "source_type": ["web", "github"]},
    )
    assert seen["filter"] == {
        "must": [
            {"key": "workspace_id", "match": {"value": "w"}},
            {"key": "language", "match": {"any": ["en", "de"]}},
            {"key": "page", "range": {"gte": 2, "lt": 9}},
            {"key": "heading_path", "match": {"value": "Intro"}},
        ],
        "must_not": [
            {"key": "document_id", "match": {"value": "d1"}},
            {"key": "source_type", "match": {"any": ["web", "github"]}},
        ],
    }


async def test_sparse_search_accepts_the_same_filters():
    seen = {}

    def handler(req):
        seen.update(body_of(req))
        return httpx.Response(200, json={"result": []})

    await store(handler).sparse_search(
        "c",
        SparseVector([1], [1.0]),
        5,
        filters={"page": Range(lte=3)},
        exclude={"kind": "question"},
    )
    assert seen["filter"]["must"] == [{"key": "page", "range": {"lte": 3}}]
    assert seen["filter"]["must_not"] == [{"key": "kind", "match": {"value": "question"}}]


async def test_search_on_a_collection_that_does_not_exist_finds_nothing():
    s = store(lambda req: httpx.Response(404, json={"status": {"error": "Not found"}}))
    assert await s.search("c", [1.0], 5) == []
    assert await s.sparse_search("c", SparseVector([1], [1.0]), 5) == []


@pytest.mark.parametrize(
    "bad",
    [{"language": []}, {"language": None}, {"language": [1.5]}, {"language": {"a": 1}}],
)
async def test_unusable_filter_conditions_are_rejected_before_any_request(bad):
    def handler(req):
        raise AssertionError("no request expected")

    with pytest.raises(ValueError):
        await store(handler).search("c", [1.0], 5, filters=bad)


async def test_existing_collections_gain_the_indexes_they_lack():
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path))
        if req.method == "GET":
            payload = {"workspace_id": {}, "collection_id": {}, "document_id": {}}
            body = {**info(8, sparse=True)["result"], "payload_schema": payload}
            return httpx.Response(200, json={"result": body})
        return httpx.Response(200, json={"result": True})

    await store(handler).ensure_collection("c", 8, sparse=True)
    created = [path for method, path in calls if method == "PUT"]
    assert len(created) == len(INDEXED_FIELDS) - 3  # only the ones missing


async def test_get_vectors_asks_for_the_dense_vector_only_and_maps_ids():
    seen = {}

    def handler(req):
        seen["path"], seen["body"] = req.url.path, body_of(req)
        return httpx.Response(
            200, json={"result": [{"id": ID, "vector": {"dense": [0.5, 0.25], "sparse": {}}}]}
        )

    got = await store(handler).get_vectors("c", [ID])
    assert got == {ID: [0.5, 0.25]}
    assert seen["path"] == "/collections/c/points"
    assert seen["body"] == {"ids": [ID], "with_vector": ["dense"], "with_payload": False}


async def test_get_vectors_handles_missing_collections_empty_input_and_bad_shapes():
    assert await store(lambda r: httpx.Response(404, json={})).get_vectors("c", [ID]) == {}

    def never(req):
        raise AssertionError("no request expected")

    assert await store(never).get_vectors("c", []) == {}
    with pytest.raises(ProviderError, match="response shape"):
        await store(lambda r: httpx.Response(200, json={"result": [{"id": ID}]})).get_vectors(
            "c", [ID]
        )


async def test_set_payload_targets_points_by_filter():
    seen = {}

    def handler(req):
        seen["path"], seen["body"] = req.url.path + "?" + req.url.query.decode(), body_of(req)
        return httpx.Response(200, json={"result": {"status": "completed"}})

    await store(handler).set_payload(
        "c", {"title": "New", "attributes": {"year": 1}}, filters={"document_id": "d1"}
    )
    assert seen["path"] == "/collections/c/points/payload?wait=true"
    assert seen["body"] == {
        "payload": {"title": "New", "attributes": {"year": 1}},
        "filter": {"must": [{"key": "document_id", "match": {"value": "d1"}}]},
    }
    assert (
        await store(lambda r: httpx.Response(404, json={})).set_payload(
            "c", {"x": 1}, filters={"document_id": "d1"}
        )
        is None
    )
    with pytest.raises(ValueError):
        await store(handler).set_payload("c", {"x": 1}, filters={})
