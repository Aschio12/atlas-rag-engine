import pytest

from app.providers.base import ChatMessage, Point, SparseVector
from app.providers.fake import FakeEmbedder, FakeLLM, FakeReranker, InMemoryVectorStore


async def test_fake_llm_echoes_last_user_message_and_records_calls():
    llm = FakeLLM()
    msgs = [ChatMessage("system", "be brief"), ChatMessage("user", "hello there")]
    out = await llm.complete(msgs)
    assert "hello there" in out.text
    assert out.usage.prompt_tokens > 0
    assert llm.calls == [msgs]


async def test_fake_embedder_is_deterministic_and_normalised():
    e = FakeEmbedder(dimension=32)
    a1, a2 = await e.embed(["red apple", "red apple"])
    assert a1 == a2
    assert sum(x * x for x in a1) == pytest.approx(1.0)


async def test_similar_texts_score_higher_than_unrelated():
    e = FakeEmbedder()
    store = InMemoryVectorStore()
    await store.ensure_collection("c", e.dimension)
    docs = {"a": "vector database for similarity search", "b": "chocolate cake recipe"}
    vecs = await e.embed(list(docs.values()))
    await store.upsert("c", [Point(k, v, {"k": k}) for k, v in zip(docs, vecs, strict=True)])
    (q,) = await e.embed(["similarity search in a vector database"])
    hits = await store.search("c", q, limit=2)
    assert [h.id for h in hits] == ["a", "b"]
    assert hits[0].score > hits[1].score


async def test_store_filters_and_delete():
    store = InMemoryVectorStore()
    await store.ensure_collection("c", 2)
    await store.upsert(
        "c",
        [
            Point("1", [1, 0], {"ws": "x"}),
            Point("2", [1, 0], {"ws": "y"}),
        ],
    )
    assert [h.id for h in await store.search("c", [1, 0], filters={"ws": "y"})] == ["2"]
    await store.delete("c", filters={"ws": "x"})
    assert [h.id for h in await store.search("c", [1, 0])] == ["2"]
    with pytest.raises(ValueError):
        await store.delete("c")


async def test_store_rejects_dimension_mismatch():
    store = InMemoryVectorStore()
    await store.ensure_collection("c", 2)
    with pytest.raises(ValueError):
        await store.upsert("c", [Point("1", [1, 0, 0])])
    with pytest.raises(ValueError):
        await store.ensure_collection("c", 3)


async def test_fake_reranker_orders_by_overlap_and_limits():
    r = FakeReranker()
    out = await r.rerank(
        "python web framework", ["cooking pasta", "python web framework fastapi"], 1
    )
    assert [x.index for x in out] == [1]


async def test_drop_collection_removes_data():
    store = InMemoryVectorStore()
    await store.ensure_collection("c", 2)
    await store.drop_collection("c")
    assert await store.search("c", [1, 0]) == []  # like Qdrant's 404: nothing to find
    assert await store.count("c") == 0


async def test_sparse_vectors_are_scored_with_idf_so_rare_terms_dominate():
    store = InMemoryVectorStore()
    await store.ensure_collection("c", 2, sparse=True)
    common, rare = 1, 2  # term ids
    await store.upsert(
        "c",
        [
            Point("a", [1, 0], {}, SparseVector([common, rare], [1.0, 1.0])),
            Point("b", [1, 0], {}, SparseVector([common], [1.0])),
            Point("c", [1, 0], {}, SparseVector([common], [1.0])),
        ],
    )
    # "common" appears in every point (near-zero IDF); "rare" only in `a`
    hits = await store.sparse_search("c", SparseVector([common, rare], [1.0, 1.0]))
    assert hits[0].id == "a" and hits[0].score > hits[1].score
    only_rare = await store.sparse_search("c", SparseVector([rare], [1.0]))
    assert [h.id for h in only_rare] == ["a"]


async def test_sparse_search_honours_filters_and_ignores_points_without_sparse():
    store = InMemoryVectorStore()
    await store.ensure_collection("c", 2, sparse=True)
    sv = SparseVector([7], [1.0])
    await store.upsert(
        "c",
        [
            Point("1", [1, 0], {"doc": "x"}, sv),
            Point("2", [1, 0], {"doc": "y"}, sv),
            Point("3", [1, 0], {"doc": "y"}),
        ],
    )
    assert [h.id for h in await store.sparse_search("c", sv, filters={"doc": "y"})] == ["2"]
    assert await store.sparse_search("c", SparseVector([99], [1.0])) == []


async def test_sparse_points_need_a_sparse_collection_and_matching_lengths():
    store = InMemoryVectorStore()
    await store.ensure_collection("plain", 2)
    with pytest.raises(ValueError, match="without sparse"):
        await store.upsert("plain", [Point("1", [1, 0], {}, SparseVector([1], [1.0]))])
    with pytest.raises(ValueError):
        SparseVector([1, 2], [1.0])


async def test_count_and_exclude():
    store = InMemoryVectorStore()
    await store.ensure_collection("c", 2)
    await store.upsert(
        "c",
        [
            Point("1", [1, 0], {"doc": "x", "v": "old"}),
            Point("2", [1, 0], {"doc": "x", "v": "new"}),
            Point("3", [1, 0], {"doc": "y", "v": "old"}),
        ],
    )
    assert await store.count("c") == 3
    assert await store.count("c", {"doc": "x"}) == 2
    await store.delete("c", filters={"doc": "x"}, exclude={"v": "new"})
    assert await store.count("c", {"doc": "x"}) == 1  # the current version survived
    assert await store.count("c", {"doc": "y"}) == 1  # other documents are untouched
    with pytest.raises(ValueError):
        await store.delete("c", ids=["1"], filters={"doc": "x"})
