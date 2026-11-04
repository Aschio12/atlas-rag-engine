"""Context assembly through the search API: expansion, merging, budget, ordering."""

import pytest
from search_helpers import RECIPES, find, ingest, make_collection, search_url

from app.providers.errors import ProviderError

PARENT_CHILD = {
    "chunking_strategy": "parent_child",
    "chunking_config": {
        "target_tokens": 64, "overlap_tokens": 8, "max_tokens": 128,
        "parent_tokens": 256, "child_tokens": 64,
    },
}  # fmt: skip


def topic_doc(name):
    """Three sections of unique sentences that all mention 'customer care'."""
    parts = [
        f"## {name} part {i}\n\n"
        + " ".join(f"{name}{i}{j} covers the customer care topic {name}{i}{j}." for j in range(8))
        for i in range(3)
    ]
    return f"# {name}\n\n" + "\n\n".join(parts)


@pytest.fixture
async def corpus(client, app, indexed):
    """The policy document plus four distinct topic documents, in one collection."""
    for name in ("alpha", "beta", "gamma", "delta"):
        await ingest(client, app, indexed, indexed["url"], topic_doc(name), f"{name}.md")
    return indexed


def stage(out, name):
    return next(s for s in out["trace"]["stages"] if s["name"] == name)


async def parsed_text(client, space, doc_id):
    r = await client.get(f"{space['url']}/{doc_id}/parsed", headers=space["headers"])
    assert r.status_code == 200, r.text
    return r.json()["text"]


async def with_context(client, space, query="customer care handbook", **ctx):
    return await find(
        client, space, query, mode="hybrid", top_k=3, context={"enabled": True, **ctx}
    )


async def test_context_is_only_built_when_asked_for(client, indexed):
    plain = await find(client, indexed, "refunds", mode="hybrid")
    assert plain["context"] is None and "context" not in [
        s["name"] for s in plain["trace"]["stages"]
    ]
    out = await with_context(client, indexed, "refunds")
    ctx = out["context"]
    assert ctx["budget"] == 3000 and 0 < ctx["tokens"] <= 3000 and ctx["blocks"]
    assert [b["label"] for b in ctx["blocks"]] == [
        f"S{i}" for i in range(1, len(ctx["blocks"]) + 1)
    ]
    assert stage(out, "context")["details"]["tokens"] == ctx["tokens"]


async def test_every_block_is_an_exact_slice_of_the_document_with_its_chunks_listed(
    client, indexed
):
    parsed = await parsed_text(client, indexed, indexed["doc"])
    out = await with_context(client, indexed, "refunds shipping warranty", neighbors=1, pool=10)
    hit_ids = {h["chunk_id"] for h in out["hits"]}
    for block in out["context"]["blocks"]:
        assert parsed[block["char_start"] : block["char_end"]] == block["text"]
        assert set(block["retrieved_chunk_ids"]) <= set(block["chunk_ids"])
        assert block["title"] == "policy" and block["tokens"] > 0
    covered = {i for b in out["context"]["blocks"] for i in b["retrieved_chunk_ids"]}
    assert hit_ids & covered


async def test_overlapping_neighbours_are_merged_without_repeating_text(client, indexed):
    plain = await with_context(client, indexed, "Rotterdam warehouse", pool=1, mmr=False)
    wide = await with_context(
        client, indexed, "Rotterdam warehouse", pool=1, mmr=False, neighbors=1
    )
    (one,), blocks = plain["context"]["blocks"], wide["context"]["blocks"]
    assert len(blocks) == 1  # three overlapping chunks became one contiguous block
    block = blocks[0]
    assert (
        len(block["chunk_ids"]) == 3 and block["retrieved_chunk_ids"] == one["retrieved_chunk_ids"]
    )
    assert block["tokens"] > one["tokens"] and "neighbouring" in " ".join(block["notes"])
    assert block["text"].count("Orders ship from the warehouse") == 1  # overlap is not duplicated
    parsed = await parsed_text(client, indexed, indexed["doc"])
    assert parsed[block["char_start"] : block["char_end"]] == block["text"]


async def test_a_child_chunk_is_expanded_to_its_parent_section(client, app, space):
    cid, url = await make_collection(client, space, "pc", **PARENT_CHILD)
    doc = await ingest(client, app, space, url)
    sp = {**space, "url": url}
    expanded = await with_context(client, sp, "Rotterdam warehouse", pool=1, mmr=False)
    (block,) = expanded["context"]["blocks"]
    (hit,) = [h for h in expanded["hits"] if h["chunk_id"] in block["retrieved_chunk_ids"]]
    assert hit["parent_id"] in block["chunk_ids"] and hit["chunk_id"] not in block["chunk_ids"]
    assert len(block["text"]) > len(hit["text"]) and hit["text"] in block["text"]
    assert "parent section" in " ".join(block["notes"])
    parsed = await parsed_text(client, sp, doc)
    assert parsed[block["char_start"] : block["char_end"]] == block["text"]

    plain = await with_context(
        client, sp, "Rotterdam warehouse", pool=1, mmr=False, parent_expansion=False
    )
    (small,) = plain["context"]["blocks"]
    assert small["text"] == hit["text"] and small["chunk_ids"] == [hit["chunk_id"]]


async def test_the_token_budget_is_never_exceeded_and_skipped_passages_are_reported(client, corpus):
    big = await with_context(client, corpus, "customer care", pool=30, max_tokens=3000, mmr=False)
    budget = 250
    small = await with_context(
        client, corpus, "customer care", pool=30, max_tokens=budget, mmr=False
    )
    assert small["context"]["tokens"] <= budget < big["context"]["tokens"]
    assert len(small["context"]["blocks"]) < len(big["context"]["blocks"])
    details = stage(small, "context")["details"]
    assert details["skipped"] and details["budget"] == budget
    assert all(b["tokens"] <= budget for b in small["context"]["blocks"])


async def test_an_oversized_expansion_falls_back_to_the_matched_chunk(client, app, space):
    cid, url = await make_collection(client, space, "pc", **PARENT_CHILD)
    await ingest(client, app, space, url)
    out = await with_context(
        client, {**space, "url": url}, "Rotterdam warehouse", pool=1, mmr=False, max_tokens=200
    )
    (block,) = out["context"]["blocks"]
    assert block["tokens"] <= 200


async def test_edges_order_puts_the_strongest_blocks_first_and_last(client, corpus):
    out = await with_context(client, corpus, "customer care", pool=30, mmr=False, order="edges")
    scores = [b["score"] for b in out["context"]["blocks"]]
    assert len(scores) >= 4
    ranked = sorted(scores, reverse=True)
    assert scores[0] == ranked[0] and scores[-1] == ranked[1]
    assert min(scores) not in (scores[0], scores[-1])


async def test_relevance_and_document_orders(client, corpus):
    kw = {"pool": 30, "mmr": False}
    rel = await with_context(client, corpus, "customer care", order="relevance", **kw)
    scores = [b["score"] for b in rel["context"]["blocks"]]
    assert scores == sorted(scores, reverse=True)
    doc = await with_context(client, corpus, "customer care", order="document", **kw)
    seen, last_start = [], {}
    for b in doc["context"]["blocks"]:  # one document at a time, each in reading order
        if not seen or seen[-1] != b["document_id"]:
            assert b["document_id"] not in seen
            seen.append(b["document_id"])
        assert b["char_start"] >= last_start.get(b["document_id"], -1)
        last_start[b["document_id"]] = b["char_start"]
    assert len(seen) >= 4


async def test_near_duplicate_passages_across_documents_are_dropped(client, app, corpus):
    twin = topic_doc("alpha").replace("alpha00 covers", "alpha00 describes")
    await ingest(client, app, corpus, corpus["url"], twin, "alpha-copy.md")
    kw = {"pool": 40, "mmr": False}
    with_dupes = await with_context(client, corpus, "customer care", dedupe=False, **kw)
    deduped = await with_context(client, corpus, "customer care", dedupe=True, **kw)
    dropped = stage(deduped, "context")["details"]["deduplicated"]
    assert len(dropped) >= 3 and all(d["similarity"] >= 0.9 for d in dropped)
    used = {i for b in deduped["context"]["blocks"] for i in b["chunk_ids"]}
    assert not used & {d["chunk_id"] for d in dropped}  # what was dropped is not in the context
    assert deduped["context"]["tokens"] < with_dupes["context"]["tokens"]
    assert "deduplicated" not in stage(with_dupes, "context")["details"]


async def test_mmr_is_reported_and_can_be_switched_off(client, indexed):
    on = await with_context(client, indexed, "customer care", pool=10)
    mmr = stage(on, "context")["details"]["mmr"]
    assert mmr["lambda"] == 0.7 and mmr["similarity_from"] == "embeddings"
    off = await with_context(client, indexed, "customer care", pool=10, mmr=False)
    assert "mmr" not in stage(off, "context")["details"]


async def test_low_lambda_spreads_picks_across_documents(client, app, indexed):
    await ingest(client, app, indexed, indexed["url"], RECIPES, "recipes.md")
    focused = await with_context(
        client, indexed, "customer care", pool=20, mmr_lambda=1.0, max_tokens=300
    )
    diverse = await with_context(
        client, indexed, "customer care", pool=20, mmr_lambda=0.0, max_tokens=300
    )
    docs = lambda out: {b["document_id"] for b in out["context"]["blocks"]}  # noqa: E731
    assert len(docs(diverse)) >= len(docs(focused))


async def test_mmr_falls_back_to_text_when_embeddings_are_unavailable(client, app, indexed):
    async def broken(collection, ids):
        raise ProviderError("memory", "vector store is down")

    app.state.providers.vector_store.get_vectors = broken
    out = await with_context(client, indexed, "customer care", pool=10)
    mmr = stage(out, "context")["details"]["mmr"]
    assert mmr["similarity_from"] == "text" and "compared text" in mmr["note"]
    assert out["context"]["blocks"]


async def test_the_context_pool_can_exceed_top_k(client, indexed):
    out = await find(
        client, indexed, "customer care", mode="hybrid", top_k=1,
        context={"enabled": True, "pool": 10, "mmr": False, "max_tokens": 3000},
    )  # fmt: skip
    assert len(out["hits"]) == 1 and len(out["context"]["blocks"]) > 1


async def test_context_follows_reranking(client, app, indexed):
    from test_rerank_stage import Reverse, use

    use(app, Reverse())
    out = await find(
        client, indexed, "customer care", mode="hybrid", top_k=2,
        rerank={"enabled": True, "candidates": 6},
        context={"enabled": True, "pool": 6, "mmr": False, "order": "relevance"},
    )  # fmt: skip
    top_hit_ids = {h["chunk_id"] for h in out["hits"]}
    best_block = out["context"]["blocks"][0]
    assert top_hit_ids & set(best_block["retrieved_chunk_ids"])


@pytest.mark.parametrize(
    "body",
    [
        {"max_tokens": 199},
        {"max_tokens": 32_001},
        {"pool": 0},
        {"neighbors": 4},
        {"dedupe_threshold": 0.4},
        {"mmr_lambda": 1.5},
        {"order": "random"},
    ],
)
async def test_invalid_context_options_are_422(client, indexed, body):
    r = await client.post(
        search_url(indexed), json={"query": "x", "context": body}, headers=indexed["headers"]
    )
    assert r.status_code == 422
