import io
import uuid

import httpx
import pytest

from app.ingest import purge
from app.ingest.vectors import vector_collection
from app.providers.errors import ProviderError
from app.providers.qdrant import QdrantStore

MD = "# Handbook\n\n" + "\n\n".join(
    f"Paragraph {i} explains policy {i} in detail." for i in range(20)
)
SMALL = {"target_tokens": 64, "overlap_tokens": 8, "max_tokens": 128}


@pytest.fixture(autouse=True)
def _clean_hooks():
    purge.clear_hooks()
    yield
    purge.clear_hooks()


async def make_collection(client, space, name="c", **body):
    r = await client.post(
        f"/v1/workspaces/{space['ws']}/collections",
        json={"name": name, "chunking_config": SMALL, **body},
        headers=space["headers"],
    )
    cid = r.json()["id"]
    return uuid.UUID(cid), f"/v1/workspaces/{space['ws']}/collections/{cid}/documents"


async def ingest(client, app, space, url, text=MD, name="a.md"):
    files = {"file": (name, io.BytesIO(text.encode()), "text/markdown")}
    doc = (await client.post(url, files=files, headers=space["headers"])).json()
    await app.state.worker.run_once()
    return doc["id"]


def store_of(app):
    return app.state.providers.vector_store


async def test_deleting_a_document_removes_only_its_vectors(client, app, space):
    import json

    from app.providers.fake import FakeLLM

    app.state.providers._llm = FakeLLM(
        reply=json.dumps({"questions": ["What is policy zero about?"]})
    )
    cid, url = await make_collection(client, space, enrichment={"questions": True, "max_chunks": 2})
    keep = await ingest(client, app, space, url, text=MD, name="keep.md")
    gone = await ingest(client, app, space, url, text=MD + "\n\nExtra.", name="gone.md")
    name, store = vector_collection(cid), store_of(app)
    before = await store.count(name)
    mine = await store.count(name, {"document_id": gone})
    assert mine > 0 and await store.count(name, {"document_id": keep}) > 0

    r = await client.delete(f"{url}/{gone}", headers=space["headers"])
    assert r.status_code == 204
    assert await store.count(name, {"document_id": gone}) == 0  # chunk and question points
    assert await store.count(name) == before - mine
    assert await store.count(name, {"document_id": keep}) > 0


async def test_a_vector_store_outage_blocks_the_delete_instead_of_orphaning_vectors(
    client, app, space, monkeypatch
):
    cid, url = await make_collection(client, space)
    doc = await ingest(client, app, space, url)
    store = store_of(app)

    async def down(*a, **k):
        raise ProviderError("qdrant", "connection refused")

    monkeypatch.setattr(store, "delete", down)
    r = await client.delete(f"{url}/{doc}", headers=space["headers"])
    assert r.status_code == 502 and r.json()["provider"] == "qdrant"
    assert (await client.get(f"{url}/{doc}", headers=space["headers"])).status_code == 200
    monkeypatch.undo()
    assert (await client.delete(f"{url}/{doc}", headers=space["headers"])).status_code == 204


async def test_deleting_a_never_indexed_document_is_fine(client, space):
    _, url = await make_collection(client, space)
    files = {"file": ("p.md", io.BytesIO(b"# Pending\n\ntext"), "text/markdown")}
    doc = (await client.post(url, files=files, headers=space["headers"])).json()
    r = await client.delete(f"{url}/{doc['id']}", headers=space["headers"])
    assert r.status_code == 204  # no vector collection exists yet


async def test_deleting_a_collection_drops_its_vector_collection_only(client, app, space):
    a, url_a = await make_collection(client, space, name="a")
    b, url_b = await make_collection(client, space, name="b")
    await ingest(client, app, space, url_a)
    await ingest(client, app, space, url_b)
    store = store_of(app)
    r = await client.delete(
        f"/v1/workspaces/{space['ws']}/collections/{a}", headers=space["headers"]
    )
    assert r.status_code == 204
    assert vector_collection(a) not in store._data
    assert await store.count(vector_collection(b)) > 0


async def test_deleting_a_workspace_drops_every_vector_collection(client, app, space):
    a, url_a = await make_collection(client, space, name="a")
    b, url_b = await make_collection(client, space, name="b")
    await ingest(client, app, space, url_a)
    await ingest(client, app, space, url_b)
    r = await client.delete(f"/v1/workspaces/{space['ws']}", headers=space["headers"])
    assert r.status_code == 204
    assert not [k for k in store_of(app)._data if k.startswith("atlas_")]


async def test_cleanup_hooks_run_for_documents_and_collections(client, app, space):
    calls = []

    @purge.on_document_deleted
    async def doc_hook(collection_id, document_id):
        calls.append(("doc", collection_id, uuid.UUID(str(document_id))))

    @purge.on_collection_deleted
    async def col_hook(collection_id):
        calls.append(("col", collection_id))

    cid, url = await make_collection(client, space)
    doc = await ingest(client, app, space, url)
    await client.delete(f"{url}/{doc}", headers=space["headers"])
    await client.delete(f"/v1/workspaces/{space['ws']}/collections/{cid}", headers=space["headers"])
    assert calls == [("doc", cid, uuid.UUID(doc)), ("col", cid)]


def test_registering_a_hook_twice_runs_it_once():
    async def hook(cid, did): ...

    purge.on_document_deleted(hook)
    purge.on_document_deleted(hook)
    assert purge._document_hooks == [hook]


async def test_missing_vector_collections_count_as_empty_and_clean(client, app):
    store = store_of(app)
    assert await store.count("atlas_nope") == 0
    await store.delete("atlas_nope", filters={"document_id": "x"})  # no error


async def test_qdrant_treats_a_missing_collection_as_nothing_to_delete_or_count():
    def handler(req):
        return httpx.Response(404, json={"status": {"error": "not found"}})

    q = QdrantStore(
        "http://q:6333", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    assert await q.count("atlas_nope") == 0
    await q.delete("atlas_nope", filters={"document_id": "x"})
