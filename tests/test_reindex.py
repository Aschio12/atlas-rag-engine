import io
import uuid

import pytest
from sqlalchemy import select

from app.db.models import Chunk
from app.ingest.vectors import vector_collection
from app.providers.errors import ProviderError
from app.providers.fake import FakeEmbedder

PARAS = [f"Paragraph {i} explains the leave policy for team {i} in some detail." for i in range(30)]
MD = "# Handbook\n\n" + "\n\n".join(PARAS)
SMALL = {"target_tokens": 64, "overlap_tokens": 8, "max_tokens": 128}


def urls(space, cid):
    base = f"/v1/workspaces/{space['ws']}/collections/{cid}"
    return base, f"{base}/documents"


async def make_collection(client, space, name="c", **body):
    r = await client.post(
        f"/v1/workspaces/{space['ws']}/collections",
        json={"name": name, "chunking_config": SMALL, **body},
        headers=space["headers"],
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def upload(client, space, docs_url, text=MD, name="a.md"):
    files = {"file": (name, io.BytesIO(text.encode()), "text/markdown")}
    r = await client.post(docs_url, files=files, headers=space["headers"])
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def run_all(app):
    while await app.state.worker.run_once():
        pass


async def status(client, space, base):
    r = await client.get(f"{base}/index", headers=space["headers"])
    assert r.status_code == 200, r.text
    return r.json()


async def chunk_ids(app):
    async with app.state.sessionmaker() as s:
        return {c.id for c in (await s.execute(select(Chunk))).scalars()}


async def test_index_status_tracks_documents_and_vectors(client, app, space):
    cid = await make_collection(client, space)
    base, docs = urls(space, cid)
    empty = await status(client, space, base)
    assert empty["documents"]["total"] == 0 and empty["in_sync"] is True
    await upload(client, space, docs)
    queued = await status(client, space, base)
    assert queued["documents"]["pending"] == 1 and queued["active_jobs"] == 1
    assert queued["in_sync"] is False
    await run_all(app)
    done = await status(client, space, base)
    assert done["documents"]["ready"] == 1 and done["active_jobs"] == 0
    assert done["vector_points"] == done["expected_points"] > 0
    assert done["searchable_chunks"] > 5 and done["in_sync"] is True
    assert done["strategy"] == "recursive"


async def test_status_notices_missing_vectors(client, app, space):
    cid = await make_collection(client, space)
    base, docs = urls(space, cid)
    await upload(client, space, docs)
    await run_all(app)
    store = app.state.providers.vector_store
    name = vector_collection(uuid.UUID(cid))
    victim = next(iter(store._data[name]))
    await store.delete(name, ids=[victim])
    st = await status(client, space, base)
    assert st["vector_points"] == st["expected_points"] - 1 and st["in_sync"] is False


async def test_reindexing_a_document_rebuilds_chunks_and_vectors_from_stored_text(
    client, app, space
):
    cid = await make_collection(client, space)
    base, docs = urls(space, cid)
    doc = await upload(client, space, docs)
    await run_all(app)
    before_ids = await chunk_ids(app)
    first = (await client.get(f"{docs}/{doc}", headers=space["headers"])).json()
    old_version = first["metadata"]["indexing"]["version"]

    r = await client.post(f"{docs}/{doc}/reindex", headers=space["headers"])
    assert r.status_code == 200
    assert r.json()["status"] == "pending" and r.json()["job"]["state"] == "queued"
    await run_all(app)

    final = (await client.get(f"{docs}/{doc}", headers=space["headers"])).json()
    assert final["status"] == "ready"
    assert final["metadata"]["indexing"]["version"] != old_version
    assert not (await chunk_ids(app)) & before_ids  # re-chunked: every chunk is new
    st = await status(client, space, base)
    assert st["in_sync"] is True  # the old generation's points are gone


async def test_reindex_reuses_the_stored_parse_unless_asked_to_reparse(client, app, space):
    cid = await make_collection(client, space)
    base, docs = urls(space, cid)
    doc = await upload(client, space, docs)
    await run_all(app)
    original = app.state.blobs.root / space["ws"] / cid / doc / "original"
    original.unlink()  # the original is gone, but the parsed text remains

    ok = await client.post(f"{docs}/{doc}/reindex", headers=space["headers"])
    await run_all(app)
    assert (await client.get(f"{docs}/{doc}", headers=space["headers"])).json()["status"] == "ready"
    assert ok.status_code == 200

    await client.post(f"{docs}/{doc}/reindex", json={"reparse": True}, headers=space["headers"])
    await run_all(app)
    failed = (await client.get(f"{docs}/{doc}", headers=space["headers"])).json()
    assert failed["status"] == "failed" and "missing from storage" in failed["job"]["error"]


async def test_a_document_that_is_processing_cannot_be_reindexed(client, space):
    cid = await make_collection(client, space)
    _, docs = urls(space, cid)
    doc = await upload(client, space, docs)  # its job is queued
    r = await client.post(f"{docs}/{doc}/reindex", headers=space["headers"])
    assert r.status_code == 409 and "still being processed" in r.json()["detail"]


async def test_changing_the_strategy_rebuilds_every_document(client, app, space):
    cid = await make_collection(client, space)
    base, docs = urls(space, cid)
    for i in range(3):
        await upload(client, space, docs, text=MD + f"\n\nUnique ending {i}.", name=f"d{i}.md")
    await run_all(app)

    cfg = {**SMALL, "parent_tokens": 256, "child_tokens": 64}
    r = await client.post(
        f"{base}/reindex",
        json={"chunking_strategy": "parent_child", "chunking_config": cfg},
        headers=space["headers"],
    )
    assert r.status_code == 200, r.text
    assert r.json()["queued"] == 3
    assert r.json()["collection"]["chunking_strategy"] == "parent_child"
    assert r.json()["collection"]["chunking_config"]["parent_tokens"] == 256
    assert (await status(client, space, base))["documents"]["pending"] == 3

    await run_all(app)
    st = await status(client, space, base)
    assert st["documents"]["ready"] == 3 and st["strategy"] == "parent_child"
    assert st["in_sync"] is True
    items = (
        await client.get(
            f"{docs}/{(await _first_doc(client, space, docs))}/chunks?limit=200",
            headers=space["headers"],
        )
    ).json()["items"]
    assert any(c["kind"] == "parent" for c in items) and any(c["kind"] == "child" for c in items)
    # parents are stored, but only children are in the vector index
    store = app.state.providers.vector_store
    kinds = {p.payload["kind"] for p in store._data[vector_collection(uuid.UUID(cid))].values()}
    assert kinds == {"chunk"}
    parent_ids = {c["id"] for c in items if c["kind"] == "parent"}
    assert not parent_ids & set(store._data[vector_collection(uuid.UUID(cid))])


async def _first_doc(client, space, docs):
    return (await client.get(docs, headers=space["headers"])).json()["items"][0]["id"]


async def test_enrichment_can_be_switched_on_for_existing_documents(client, app, space):
    import json

    from app.providers.fake import FakeLLM

    cid = await make_collection(client, space)
    base, docs = urls(space, cid)
    doc = await upload(client, space, docs)
    await run_all(app)
    app.state.providers._llm = FakeLLM(
        reply=json.dumps({"questions": ["What is the leave policy?"]})
    )
    r = await client.post(
        f"{base}/reindex",
        json={"enrichment": {"questions": True, "max_chunks": 2}},
        headers=space["headers"],
    )
    assert r.status_code == 200 and r.json()["collection"]["enrichment"]["questions"] is True
    await run_all(app)
    final = (await client.get(f"{docs}/{doc}", headers=space["headers"])).json()
    assert final["metadata"]["indexing"]["questions"] == 2
    assert (await status(client, space, base))["in_sync"] is True


async def test_changing_the_embedding_model_drops_and_rebuilds_the_index(client, app, space):
    cid = await make_collection(client, space)
    base, docs = urls(space, cid)
    await upload(client, space, docs)
    await run_all(app)
    name = vector_collection(uuid.UUID(cid))
    store = app.state.providers.vector_store
    assert store._dims[name] == 64

    used = []

    def embedder_for(model):
        used.append(model)
        return FakeEmbedder(dimension=32, model=model or "default")  # a different size

    app.state.providers.embedder_for = embedder_for
    r = await client.post(
        f"{base}/reindex", json={"embed_model": "bigger-model"}, headers=space["headers"]
    )
    assert r.status_code == 200 and r.json()["collection"]["embed_model"] == "bigger-model"
    assert name not in store._data  # old-dimension vectors are gone immediately
    await run_all(app)
    assert store._dims[name] == 32 and set(used) == {"bigger-model"}
    st = await status(client, space, base)
    assert st["in_sync"] is True and st["embed_model"] == "bigger-model"


async def test_reindex_is_refused_while_documents_are_processing(client, space):
    cid = await make_collection(client, space)
    base, docs = urls(space, cid)
    await upload(client, space, docs)  # queued, not yet processed
    r = await client.post(f"{base}/reindex", headers=space["headers"])
    assert r.status_code == 409 and "still processing" in r.json()["detail"]


async def test_an_index_outage_aborts_a_model_change_before_anything_is_modified(
    client, app, space, monkeypatch
):
    cid = await make_collection(client, space)
    base, docs = urls(space, cid)
    await upload(client, space, docs)
    await run_all(app)
    store = app.state.providers.vector_store

    async def down(name):
        raise ProviderError("qdrant", "connection refused")

    monkeypatch.setattr(store, "drop_collection", down)
    r = await client.post(
        f"{base}/reindex", json={"embed_model": "other"}, headers=space["headers"]
    )
    assert r.status_code == 502
    col = (await client.get(base, headers=space["headers"])).json()
    assert col["embed_model"] != "other"
    assert (await status(client, space, base))["documents"]["ready"] == 1


async def test_reindex_works_with_no_body_and_on_an_empty_collection(client, app, space):
    cid = await make_collection(client, space)
    base, docs = urls(space, cid)
    empty = await client.post(f"{base}/reindex", headers=space["headers"])
    assert empty.status_code == 200 and empty.json()["queued"] == 0
    await upload(client, space, docs)
    await run_all(app)
    again = await client.post(f"{base}/reindex", headers=space["headers"])
    assert again.json()["queued"] == 1
    await run_all(app)
    assert (await status(client, space, base))["in_sync"] is True


@pytest.mark.parametrize(
    "body",
    [
        {"chunking_config": {"target_tokens": 400, "overlap_tokens": 399}},
        {"chunking_strategy": "nope"},
        {"embed_model": ""},
        {"enrichment": {"max_questions": 99}},
    ],
)
async def test_reindex_validates_the_new_settings(client, space, body):
    cid = await make_collection(client, space)
    base, _ = urls(space, cid)
    r = await client.post(f"{base}/reindex", json=body, headers=space["headers"])
    assert r.status_code == 422


async def test_roles_and_tenancy(client, app, signup, space):
    from sqlalchemy import select as sel

    from app.db.models import Role, User, WorkspaceMember

    cid = await make_collection(client, space)
    base, docs = urls(space, cid)
    doc = await upload(client, space, docs)
    await run_all(app)

    eve = await signup("eve@example.com")
    for method, url in [
        ("POST", f"{base}/reindex"),
        ("POST", f"{docs}/{doc}/reindex"),
        ("GET", f"{base}/index"),
    ]:
        assert (await client.request(method, url, headers=eve)).status_code == 404

    async def join(email, role):
        h = await signup(email)
        async with app.state.sessionmaker() as s:
            uid = (await s.execute(sel(User.id).where(User.email == email))).scalar_one()
            s.add(WorkspaceMember(workspace_id=uuid.UUID(space["ws"]), user_id=uid, role=role))
            await s.commit()
        return h

    member = await join("member@example.com", Role.MEMBER)
    viewer = await join("viewer@example.com", Role.VIEWER)
    assert (await client.post(f"{base}/reindex", headers=member)).status_code == 403  # admin only
    assert (await client.post(f"{docs}/{doc}/reindex", headers=viewer)).status_code == 403
    assert (await client.get(f"{base}/index", headers=viewer)).status_code == 200
    assert (await client.post(f"{docs}/{doc}/reindex", headers=member)).status_code == 200
