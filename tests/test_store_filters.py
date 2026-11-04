"""Filter semantics of the vector store contract, checked against the in-memory store.

The Qdrant adapter's translation of the same conditions is covered in test_qdrant.py, and its
behaviour on a real server in tests/integration."""

import pytest

from app.providers.base import Point, Range, SparseVector, check_filters
from app.providers.fake import InMemoryVectorStore

ROWS = [
    # id, payload
    ("a", {"language": "en", "page": 1, "source_type": "file", "heading_path": ["Intro", "Scope"]}),
    ("b", {"language": "en", "page": 5, "source_type": "web", "heading_path": ["Pricing"]}),
    ("c", {"language": "de", "page": 9, "source_type": "file", "heading_path": []}),
    ("d", {"language": None, "page": None, "source_type": "github"}),
]


@pytest.fixture
async def store():
    s = InMemoryVectorStore()
    await s.ensure_collection("c", 2, sparse=True)
    await s.upsert(
        "c",
        [Point(pid, [1.0, 0.0], payload, SparseVector([7], [1.0])) for pid, payload in ROWS],
    )
    return s


async def ids(store, **kw):
    hits = await store.search("c", [1.0, 0.0], 10, **kw)
    return sorted(h.id for h in hits)


async def test_scalar_means_equals(store):
    assert await ids(store, filters={"language": "en"}) == ["a", "b"]


async def test_a_list_means_any_of(store):
    assert await ids(store, filters={"language": ["de", "en"]}) == ["a", "b", "c"]


async def test_conditions_are_anded(store):
    assert await ids(store, filters={"language": "en", "source_type": "file"}) == ["a"]


async def test_range_bounds_are_inclusive_or_exclusive_as_named(store):
    assert await ids(store, filters={"page": Range(gte=1, lte=5)}) == ["a", "b"]
    assert await ids(store, filters={"page": Range(gt=1, lt=9)}) == ["b"]
    assert await ids(store, filters={"page": Range(gte=9)}) == ["c"]


async def test_points_without_the_field_never_match_a_range(store):
    assert "d" not in await ids(store, filters={"page": Range(gte=0)})


async def test_array_payloads_match_on_any_element(store):
    assert await ids(store, filters={"heading_path": "Scope"}) == ["a"]
    assert await ids(store, filters={"heading_path": ["Pricing", "Scope"]}) == ["a", "b"]


async def test_exclude_drops_points_matching_any_condition(store):
    assert await ids(store, exclude={"language": "en"}) == ["c", "d"]
    assert await ids(store, exclude={"language": "de", "source_type": "web"}) == ["a", "d"]
    assert await ids(store, exclude={"page": Range(gte=5)}) == ["a", "d"]


async def test_filters_and_exclusions_combine(store):
    assert await ids(store, filters={"language": "en"}, exclude={"source_type": "web"}) == ["a"]


async def test_sparse_search_honours_the_same_filters(store):
    hits = await store.sparse_search(
        "c",
        SparseVector([7], [1.0]),
        10,
        filters={"language": ["en", "de"]},
        exclude={"page": Range(gt=5)},
    )
    assert sorted(h.id for h in hits) == ["a", "b"]


async def test_delete_with_range_and_exclusion(store):
    await store.delete("c", filters={"page": Range(gte=1)}, exclude={"language": "de"})
    assert await ids(store) == ["c", "d"]


@pytest.mark.parametrize(
    "bad",
    [{"language": []}, {"language": None}, {"language": [1.5]}, {"language": {"x": 1}}],
)
def test_unusable_conditions_are_rejected(bad):
    with pytest.raises(ValueError):
        check_filters(bad)


def test_a_range_needs_a_numeric_bound():
    with pytest.raises(ValueError):
        Range()
    with pytest.raises(ValueError):
        Range(gte=True)


async def test_the_store_rejects_bad_filters_too(store):
    with pytest.raises(ValueError):
        await store.search("c", [1.0, 0.0], 3, filters={"language": []})


async def test_get_vectors_returns_the_stored_dense_vectors_and_skips_unknown_ids(store):
    got = await store.get_vectors("c", ["a", "zzz", "d"])
    assert got == {"a": [1.0, 0.0], "d": [1.0, 0.0]}
    assert await store.get_vectors("c", []) == {}
    assert await store.get_vectors("missing", ["a"]) == {}


async def test_nested_payload_keys_are_addressed_with_dots():
    s = InMemoryVectorStore()
    await s.ensure_collection("n", 2)
    await s.upsert(
        "n",
        [
            Point("a", [1.0, 0.0], {"attributes": {"year": 2024, "tags": ["rag", "eval"]}}),
            Point("b", [1.0, 0.0], {"attributes": {"year": 2019}}),
            Point("c", [1.0, 0.0], {"attributes": "not a map"}),
            Point("d", [1.0, 0.0], {}),
        ],
    )

    async def found(**kw):
        return sorted(h.id for h in await s.search("n", [1.0, 0.0], 10, **kw))

    assert await found(filters={"attributes.year": Range(gte=2020)}) == ["a"]
    assert await found(filters={"attributes.tags": "eval"}) == ["a"]
    assert await found(exclude={"attributes.year": 2019}) == ["a", "c", "d"]
    assert await found(filters={"attributes.missing": 1}) == []


async def test_set_payload_updates_matching_points_and_keeps_other_fields():
    s = InMemoryVectorStore()
    await s.ensure_collection("p", 2)
    await s.upsert(
        "p",
        [
            Point("a", [1.0, 0.0], {"document_id": "d1", "title": "old", "keep": 1}),
            Point("b", [1.0, 0.0], {"document_id": "d2", "title": "old"}),
        ],
    )
    await s.set_payload(
        "p", {"title": "new", "attributes": {"year": 1}}, filters={"document_id": "d1"}
    )
    hits = {h.id: h.payload for h in await s.search("p", [1.0, 0.0], 10)}
    assert hits["a"] == {"document_id": "d1", "title": "new", "keep": 1, "attributes": {"year": 1}}
    assert hits["b"]["title"] == "old"
    await s.set_payload("missing", {"x": 1}, filters={"document_id": "d1"})  # a no-op
    with pytest.raises(ValueError):
        await s.set_payload("p", {"x": 1}, filters={})
