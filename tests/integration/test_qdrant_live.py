"""Runs the Qdrant adapter against a real Qdrant server. Skipped unless one is configured:

    docker run -p 6333:6333 qdrant/qdrant:v1.13.4
    ATLAS_TEST_QDRANT_URL=http://localhost:6333 pytest tests/integration

CI runs this against a Qdrant service container, which is what proves the REST shapes the unit
tests only mock (named dense + sparse vectors, IDF modifier, payload indexes, count, must_not).
"""

import os
import uuid

import pytest

from app.ingest.sparse import encode_document, encode_query
from app.providers.base import Point, Range
from app.providers.fake import FakeEmbedder
from app.providers.qdrant import INDEXED_FIELDS, QdrantStore

URL = os.environ.get("ATLAS_TEST_QDRANT_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set ATLAS_TEST_QDRANT_URL to run")

DOCS = {
    "refunds": "Customers can request a refund within 14 days under the refund policy.",
    "shipping": "Shipping takes 7 days. Parcels are tracked by number.",
    "support": "Our support team answers questions every weekday morning.",
}


@pytest.fixture
async def store():
    s = QdrantStore(URL)
    name = f"atlas_test_{uuid.uuid4().hex}"
    yield s, name
    await s.drop_collection(name)
    await s._client.aclose()


async def load(s, name, version="v1", doc="d1"):
    emb = FakeEmbedder(dimension=32)
    vectors = await emb.embed(list(DOCS.values()))
    ids = {k: str(uuid.uuid4()) for k in DOCS}
    await s.ensure_collection(name, 32, sparse=True)
    await s.upsert(
        name,
        [
            Point(
                ids[k], vec,
                {"document_id": doc, "index_version": version, "key": k, "kind": "chunk"},
                encode_document(text),
            )
            for (k, text), vec in zip(DOCS.items(), vectors, strict=True)
        ],
    )  # fmt: skip
    return ids, emb


async def test_dense_and_sparse_search_roundtrip(store):
    s, name = store
    ids, emb = await load(s, name)
    (q,) = await emb.embed(["refund within 14 days"])
    dense = await s.search(name, q, limit=3)
    assert dense[0].payload["key"] == "refunds"
    sparse = await s.sparse_search(name, encode_query("how do refunds work"), limit=3)
    assert sparse[0].payload["key"] == "refunds"
    only_support = await s.sparse_search(name, encode_query("weekday support"), limit=3)
    assert [h.payload["key"] for h in only_support][:1] == ["support"]


async def test_filters_counts_and_ensure_collection_is_idempotent(store):
    s, name = store
    await load(s, name)
    await s.ensure_collection(name, 32, sparse=True)  # second call is a no-op
    assert await s.count(name) == 3
    assert await s.count(name, {"document_id": "d1"}) == 3
    assert await s.count(name, {"document_id": "other"}) == 0
    (q,) = await FakeEmbedder(dimension=32).embed(["anything"])
    assert await s.search(name, q, filters={"document_id": "other"}) == []


async def test_delete_with_exclude_removes_only_the_old_generation(store):
    s, name = store
    await load(s, name, version="old")
    await load(s, name, version="new")  # new random ids: both generations coexist
    assert await s.count(name) == 6
    await s.delete(name, filters={"document_id": "d1"}, exclude={"index_version": "new"})
    assert await s.count(name) == 3
    assert await s.count(name, {"index_version": "new"}) == 3


async def test_missing_collections_count_as_empty_and_clean(store):
    s, _ = store
    ghost = f"atlas_test_missing_{uuid.uuid4().hex}"
    assert await s.count(ghost) == 0
    await s.delete(ghost, filters={"document_id": "x"})


async def test_dimension_and_sparse_mismatches_are_reported(store):
    from app.providers.errors import ProviderError

    s, name = store
    await s.ensure_collection(name, 32, sparse=False)
    with pytest.raises(ProviderError, match="dimension"):
        await s.ensure_collection(name, 64)
    with pytest.raises(ProviderError, match="without sparse"):
        await s.ensure_collection(name, 32, sparse=True)


ATTRS = {
    "refunds": {
        "language": "en", "page": 1, "source_type": "file", "heading_path": ["Terms", "Refunds"]
    },
    "shipping": {"language": "en", "page": 4, "source_type": "web", "heading_path": ["Shipping"]},
    "support": {"language": "de", "page": 9, "source_type": "file", "heading_path": []},
}  # fmt: skip


async def load_attributed(s, name):
    emb = FakeEmbedder(dimension=32)
    vectors = await emb.embed(list(DOCS.values()))
    await s.ensure_collection(name, 32, sparse=True)
    await s.upsert(
        name,
        [
            Point(
                str(uuid.uuid4()), vec,
                {"key": k, "kind": "chunk", "created_at": 1_700_000_000 + i, **ATTRS[k]},
                encode_document(text),
            )
            for i, ((k, text), vec) in enumerate(zip(DOCS.items(), vectors, strict=True))
        ],
    )  # fmt: skip
    return emb


def keys(hits):
    return sorted(h.payload["key"] for h in hits)


async def test_metadata_filters_any_of_range_array_and_exclusion(store):
    s, name = store
    emb = await load_attributed(s, name)
    (q,) = await emb.embed(["anything at all"])

    async def run(**kw):
        return keys(await s.search(name, q, limit=10, **kw))

    assert await run(filters={"language": "en"}) == ["refunds", "shipping"]
    assert await run(filters={"language": ["de", "en"]}) == ["refunds", "shipping", "support"]
    assert await run(filters={"page": Range(gte=2, lte=9)}) == ["shipping", "support"]
    assert await run(filters={"page": Range(gt=1, lt=9)}) == ["shipping"]
    assert await run(filters={"created_at": Range(gte=1_700_000_001)}) == ["shipping", "support"]
    assert await run(filters={"heading_path": "Refunds"}) == ["refunds"]  # array: any element
    assert await run(filters={"heading_path": ["Shipping", "Terms"]}) == ["refunds", "shipping"]
    assert await run(exclude={"source_type": "web"}) == ["refunds", "support"]
    assert await run(exclude={"language": "de", "page": Range(lte=1)}) == ["shipping"]
    assert await run(filters={"language": "en"}, exclude={"source_type": "web"}) == ["refunds"]


async def test_sparse_search_honours_filters_and_exclusions(store):
    s, name = store
    await load_attributed(s, name)
    query = encode_query("refund shipping support parcels weekday")
    hits = await s.sparse_search(
        name, query, limit=10, filters={"language": "en"}, exclude={"source_type": "web"}
    )
    assert keys(hits) == ["refunds"]


async def test_searching_a_collection_that_does_not_exist_finds_nothing(store):
    s, _ = store
    ghost = f"atlas_test_missing_{uuid.uuid4().hex}"
    assert await s.search(ghost, [0.0] * 32, limit=3) == []
    assert await s.sparse_search(ghost, encode_query("anything"), limit=3) == []


async def test_collections_made_before_a_field_was_indexed_gain_its_index(store):
    s, name = store
    # an "old" collection: right vectors, but only the original payload indexes
    await s._call(
        "PUT",
        f"/collections/{name}",
        json={
            "vectors": {"dense": {"size": 32, "distance": "Cosine"}},
            "sparse_vectors": {"sparse": {"modifier": "idf"}},
        },
    )
    await s._call(
        "PUT",
        f"/collections/{name}/index?wait=true",
        json={"field_name": "workspace_id", "field_schema": "keyword"},
    )
    await s.ensure_collection(name, 32, sparse=True)
    info = await s._call("GET", f"/collections/{name}")
    assert set(INDEXED_FIELDS) <= set(info.json()["result"]["payload_schema"])
    schema = info.json()["result"]["payload_schema"]
    assert schema["page"]["data_type"] == "integer" and schema["language"]["data_type"] == "keyword"


async def test_nested_attribute_keys_can_be_filtered_and_updated_in_place(store):
    s, name = store
    emb = FakeEmbedder(dimension=32)
    await s.ensure_collection(name, 32, sparse=True)
    vectors = await emb.embed(list(DOCS.values()))
    attributes = {
        "refunds": {"year": 2019, "author": "Chen", "tags": ["policy"]},
        "shipping": {"year": 2024, "author": "Li", "tags": ["ops", "policy"]},
        "support": {"year": 2022, "author": "Li", "tags": ["ops"]},
    }
    await s.upsert(
        name,
        [
            Point(
                str(uuid.uuid4()), vec,
                {"key": k, "document_id": f"d-{k}", "attributes": attributes[k]},
                encode_document(text),
            )
            for (k, text), vec in zip(DOCS.items(), vectors, strict=True)
        ],
    )  # fmt: skip
    (q,) = await emb.embed(["anything at all"])

    async def run(**kw):
        return keys(await s.search(name, q, limit=10, **kw))

    assert await run(filters={"attributes.year": Range(gt=2020)}) == ["shipping", "support"]
    assert await run(filters={"attributes.author": "Li"}) == ["shipping", "support"]
    assert await run(filters={"attributes.tags": "policy"}) == ["refunds", "shipping"]
    assert await run(filters={"attributes.tags": ["ops", "nothing"]}) == ["shipping", "support"]
    assert await run(exclude={"attributes.author": "Li"}) == ["refunds"]

    # set_payload replaces one document's attributes without touching its vectors
    await s.set_payload(name, {"attributes": {"year": 2030}}, filters={"document_id": "d-refunds"})
    assert await run(filters={"attributes.year": Range(gte=2030)}) == ["refunds"]
    assert await run(filters={"attributes.author": "Chen"}) == []  # replaced as a whole
    assert await run(filters={"document_id": "d-refunds"}) == ["refunds"]  # other fields kept
    hits = await s.search(name, q, limit=1, filters={"document_id": "d-refunds"})
    assert hits[0].payload["key"] == "refunds"


async def test_stored_dense_vectors_can_be_read_back_by_id(store):
    s, name = store
    emb = FakeEmbedder(dimension=32)
    await s.ensure_collection(name, 32, sparse=True)
    (vec,) = await emb.embed(["some text"])
    point_id = str(uuid.uuid4())
    await s.upsert(name, [Point(point_id, vec, {"key": "k"}, encode_document("some text"))])
    got = await s.get_vectors(name, [point_id, str(uuid.uuid4())])
    assert set(got) == {point_id}
    assert got[point_id] == pytest.approx(vec, abs=1e-5)
    assert await s.get_vectors(name, []) == {}
    assert await s.get_vectors(f"atlas_test_missing_{uuid.uuid4().hex}", [point_id]) == {}
