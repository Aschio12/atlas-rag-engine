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
from app.providers.base import Point
from app.providers.fake import FakeEmbedder
from app.providers.qdrant import QdrantStore

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
