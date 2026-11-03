import pytest

from app.providers.base import ChatMessage, Point
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
