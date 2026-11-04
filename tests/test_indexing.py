import io
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.db.models import Chunk
from app.ingest.sparse import encode_query
from app.ingest.stages import chunk_stage, embed_stage, index_stage
from app.ingest.vectors import epoch_seconds, question_point_id, vector_collection
from app.providers.errors import ProviderError
from app.providers.fake import FakeEmbedder

LEAVE = [f"Paragraph {i} explains the leave policy for team {i} in some detail." for i in range(30)]
MD = "# Handbook\n\n" + "\n\n".join(LEAVE) + "\n\n## Refunds\n\nRefunds are issued within 14 days."
SMALL = {"target_tokens": 64, "overlap_tokens": 8, "max_tokens": 128}


async def make_collection(client, space, name="c", **body):
    r = await client.post(
        f"/v1/workspaces/{space['ws']}/collections",
        json={"name": name, "chunking_config": SMALL, **body},
        headers=space["headers"],
    )
    assert r.status_code == 201, r.text
    cid = r.json()["id"]
    return cid, f"/v1/workspaces/{space['ws']}/collections/{cid}/documents"


async def ingest(client, app, space, url, text=MD, name="handbook.md"):
    files = {"file": (name, io.BytesIO(text.encode()), "text/markdown")}
    doc = (await client.post(url, files=files, headers=space["headers"])).json()
    assert await app.state.worker.run_once() is True
    final = (await client.get(f"{url}/{doc['id']}", headers=space["headers"])).json()
    return doc["id"], final


def store_of(app):
    return app.state.providers.vector_store


async def chunk_rows(app):
    async with app.state.sessionmaker() as s:
        return (await s.execute(select(Chunk).order_by(Chunk.ordinal))).scalars().all()


async def test_a_document_is_fully_indexed_with_dense_and_sparse_vectors(client, app, space):
    cid, url = await make_collection(client, space)
    doc_id, final = await ingest(client, app, space, url)
    assert final["status"] == "ready" and final["job"]["state"] == "done"
    info = final["metadata"]["indexing"]
    assert info["points"] == info["chunks"] > 5 and info["questions"] == 0
    assert info["verified"] == info["points"] and info["dimension"] == 64

    name = vector_collection(__import__("uuid").UUID(cid))
    store = store_of(app)
    assert await store.count(name) == info["points"]
    chunks = await chunk_rows(app)
    points = store._data[name]
    assert {str(c.id) for c in chunks} == set(points)  # point id == chunk id
    for c in chunks:
        p = points[str(c.id)]
        assert p.sparse is not None and p.sparse.indices
        pl = p.payload
        assert (pl["document_id"], pl["collection_id"], pl["workspace_id"]) == (
            doc_id,
            cid,
            space["ws"],
        )
        assert (pl["char_start"], pl["char_end"], pl["page"]) == (c.char_start, c.char_end, c.page)
        assert pl["kind"] == "chunk" and pl["title"] == "handbook"
        assert pl["heading_path"] == c.meta["heading_path"] and pl["language"] == "en"
        assert pl["index_version"] == info["version"]


async def test_payload_carries_the_document_attributes_that_retrieval_filters_on(
    client, app, space
):
    cid, url = await make_collection(client, space)
    doc_id, final = await ingest(client, app, space, url)
    name = vector_collection(__import__("uuid").UUID(cid))
    payloads = [p.payload for p in store_of(app)._data[name].values()]
    created = epoch_seconds(datetime.fromisoformat(final["created_at"]))
    for pl in payloads:
        assert pl["source_type"] == "file" and pl["source_id"] is None
        assert pl["file_type"] == "markdown"
        assert abs(pl["created_at"] - created) <= 1 and isinstance(pl["created_at"], int)


def test_epoch_seconds_treats_naive_database_timestamps_as_utc():
    naive = datetime(2024, 1, 1, 12, 0, 0)
    aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert epoch_seconds(naive) == epoch_seconds(aware) == 1704110400
    plus_two = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    assert epoch_seconds(plus_two) == 1704110400


async def test_dense_search_with_filters_finds_the_right_passage(client, app, space):
    cid, url = await make_collection(client, space)
    doc_id, _ = await ingest(client, app, space, url)
    name = vector_collection(__import__("uuid").UUID(cid))
    (q,) = await FakeEmbedder().embed(["Refunds are issued within 14 days"])
    hits = await store_of(app).search(name, q, limit=3, filters={"document_id": doc_id})
    assert "Refunds are issued" in next(
        c.text for c in await chunk_rows(app) if str(c.id) == hits[0].id
    )
    none = await store_of(app).search(name, q, filters={"workspace_id": "someone-else"})
    assert none == []


async def test_sparse_search_finds_exact_terms_and_uses_headings_and_keywords(client, app, space):
    cid, url = await make_collection(client, space)
    await ingest(client, app, space, url)
    name = vector_collection(__import__("uuid").UUID(cid))
    store = store_of(app)
    chunks = {str(c.id): c for c in await chunk_rows(app)}
    top = await store.sparse_search(name, encode_query("refunds 14 days"))
    assert "Refunds are issued" in chunks[top[0].id].text
    # the heading "Refunds" is indexed with the chunk, so even a heading-only query lands there
    by_heading = await store.sparse_search(name, encode_query("refunds"))
    assert "Refunds" in chunks[by_heading[0].id].meta["heading_path"]


async def test_hypothetical_questions_become_points_that_lead_back_to_their_chunk(
    client, app, space
):
    import json

    from app.providers.fake import FakeLLM

    reply = json.dumps({"questions": ["How long do I have to request a refund?"]})
    app.state.providers._llm = FakeLLM(reply=reply)
    cid, url = await make_collection(client, space, enrichment={"questions": True, "max_chunks": 3})
    _, final = await ingest(client, app, space, url)
    info = final["metadata"]["indexing"]
    assert info["questions"] == 3 and info["points"] == info["chunks"] + 3

    name = vector_collection(__import__("uuid").UUID(cid))
    store = store_of(app)
    qpoints = [p for p in store._data[name].values() if p.payload["kind"] == "question"]
    assert len(qpoints) == 3
    chunks = {str(c.id): c for c in await chunk_rows(app)}
    for p in qpoints:
        assert p.payload["chunk_id"] in chunks
        assert p.payload["question"] == "How long do I have to request a refund?"
        assert p.id == question_point_id(chunks[p.payload["chunk_id"]].id, 0)
    # a user's question matches the stored question closely
    (q,) = await FakeEmbedder().embed(["How long do I have to request a refund?"])
    best = (await store.search(name, q, limit=1))[0]
    assert best.payload["kind"] == "question"


async def test_contextual_chunks_embed_their_context_and_index_it_lexically(client, app, space):
    from app.providers.fake import FakeLLM

    seen = []

    class Spy(FakeEmbedder):
        async def embed(self, texts):
            seen.extend(texts)
            return await super().embed(texts)

    app.state.providers._llm = FakeLLM(reply="Part of the zebrafish handbook.")
    app.state.providers._embedder = Spy()
    cid, url = await make_collection(
        client, space, chunking_strategy="contextual",
        chunking_config={**SMALL, "context_max_chunks": 2},
    )  # fmt: skip
    await ingest(client, app, space, url)
    assert any(t.startswith("Part of the zebrafish handbook.\n\n") for t in seen)
    name = vector_collection(__import__("uuid").UUID(cid))
    hits = await store_of(app).sparse_search(name, encode_query("zebrafish"))
    assert hits  # the context is searchable even though the passage never says it


async def test_reindexing_replaces_old_points_but_only_after_the_new_ones_exist(client, app, space):
    import uuid

    cid, url = await make_collection(client, space)
    doc_id, first = await ingest(client, app, space, url)
    name = vector_collection(uuid.UUID(cid))
    store = store_of(app)
    old_version = first["metadata"]["indexing"]["version"]
    total = await store.count(name)

    # Re-chunk (new chunk ids) and embed again: both generations are present, nothing was lost.
    async with app.state.sessionmaker() as s:
        from app.db.models import Collection, Document
        from app.ingest.pipeline import StageContext

        doc = await s.get(Document, uuid.UUID(doc_id))
        col = await s.get(Collection, uuid.UUID(cid))
        ctx = StageContext(s, app.state.settings, app.state.blobs, app.state.providers, doc, col)
        await chunk_stage(ctx)
        await embed_stage(ctx)
        new_version = doc.meta["indexing"]["version"]
        assert new_version != old_version
        assert await store.count(name) == total * 2
        assert await store.count(name, {"index_version": old_version}) == total
        await index_stage(ctx)  # now the old generation is removed
        await s.commit()
    assert await store.count(name) == total
    assert await store.count(name, {"index_version": old_version}) == 0
    assert await store.count(name, {"index_version": new_version}) == total


async def test_embedding_runs_in_batches_with_progress(client, app, settings, space):
    batches = []

    class Spy(FakeEmbedder):
        async def embed(self, texts):
            batches.append(len(texts))
            return await super().embed(texts)

    settings.embed_batch_size = 4
    app.state.providers._embedder = Spy()
    _, url = await make_collection(client, space)
    _, final = await ingest(client, app, space, url)
    n = final["metadata"]["indexing"]["points"]
    assert batches == [4] * (n // 4) + ([n % 4] if n % 4 else [])


async def test_overlong_texts_are_truncated_and_counted(client, app, settings, space):
    settings.embed_max_chars = 256
    seen = []

    class Spy(FakeEmbedder):
        async def embed(self, texts):
            seen.extend(texts)
            return await super().embed(texts)

    app.state.providers._embedder = Spy()
    big = {"target_tokens": 400, "max_tokens": 800, "overlap_tokens": 0}
    _, url = await make_collection(client, space, chunking_config=big)
    _, final = await ingest(client, app, space, url, text="Sentence about leave. " * 200)
    assert final["metadata"]["indexing"]["truncated"] >= 1
    assert all(len(t) <= 256 for t in seen)


async def test_each_collection_embeds_with_its_own_model(client, app, space):
    models = []

    class Spy(FakeEmbedder):
        async def embed(self, texts):
            models.append(self.model)
            return await super().embed(texts)

    real = app.state.providers.embedder_for
    app.state.providers.embedder_for = lambda m: Spy(model=m or "default")
    cid, url = await make_collection(client, space, embed_model="my-special-model")
    await ingest(client, app, space, url)
    app.state.providers.embedder_for = real
    assert set(models) == {"my-special-model"}


async def test_collections_never_share_vectors(client, app, space):
    import uuid

    a, url_a = await make_collection(client, space, name="a")
    b, url_b = await make_collection(client, space, name="b")
    await ingest(client, app, space, url_a)
    await ingest(client, app, space, url_b)
    store = store_of(app)
    for cid in (a, b):
        name = vector_collection(uuid.UUID(cid))
        assert {p.payload["collection_id"] for p in store._data[name].values()} == {cid}


async def test_embedding_outage_retries_with_a_clear_message(client, app, space):
    class Down(FakeEmbedder):
        async def embed(self, texts):
            raise ProviderError("ollama", "connection refused")

    _, url = await make_collection(client, space)
    files = {"file": ("h.md", io.BytesIO(MD.encode()), "text/markdown")}
    doc = (await client.post(url, files=files, headers=space["headers"])).json()
    app.state.providers._embedder = Down()
    await app.state.worker.run_once()
    final = (await client.get(f"{url}/{doc['id']}", headers=space["headers"])).json()
    assert final["job"]["state"] == "queued" and final["status"] == "pending"
    assert final["job"]["error"] == "The 'ollama' provider failed; will retry"


async def test_a_changed_embedding_size_fails_permanently_with_advice(client, app, space):
    cid, url = await make_collection(client, space)
    await ingest(client, app, space, url)  # creates the 64-dimensional vector collection
    app.state.providers._embedder = FakeEmbedder(dimension=128)
    doc = (
        await client.post(
            url,
            files={
                "file": (
                    "other.md",
                    io.BytesIO(b"# Other\n\nA different document."),
                    "text/markdown",
                )
            },
            headers=space["headers"],
        )
    ).json()
    await app.state.worker.run_once()
    final = (await client.get(f"{url}/{doc['id']}", headers=space["headers"])).json()
    assert final["status"] == "failed"
    assert "re-index the collection" in final["job"]["error"]


async def test_an_incomplete_write_is_detected_and_retried(client, app, space, monkeypatch):
    store = store_of(app)
    real = store.count

    async def lossy(collection, filters=None):
        n = await real(collection, filters)
        return n - 1 if filters and "index_version" in filters else n

    monkeypatch.setattr(store, "count", lossy)
    _, url = await make_collection(client, space)
    _, final = await ingest(client, app, space, url)
    assert final["job"]["state"] == "queued"
    assert "expected" in final["job"]["error"] or "provider failed" in final["job"]["error"]
    assert final["status"] == "pending"


@pytest.mark.parametrize("fmt", ["handbook.md"])
async def test_indexing_summary_is_visible_on_the_document(client, app, space, fmt):
    _, url = await make_collection(client, space)
    _, final = await ingest(client, app, space, url, name=fmt)
    assert set(final["metadata"]["indexing"]) >= {
        "version",
        "points",
        "dimension",
        "model",
        "verified",
    }
