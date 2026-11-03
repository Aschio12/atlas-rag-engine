import uuid

import pytest
from sqlalchemy import func, select

from app.db.models import Chunk, Collection, Document, Role, User, WorkspaceMember


async def workspace(client, headers, name="Acme"):
    return (await client.post("/v1/workspaces", json={"name": name}, headers=headers)).json()["id"]


def url(ws, cid=None):
    base = f"/v1/workspaces/{ws}/collections"
    return f"{base}/{cid}" if cid else base


async def test_create_defaults_and_fields(client, signup, settings):
    h = await signup()
    ws = await workspace(client, h)
    r = await client.post(url(ws), json={"name": "  Manuals  "}, headers=h)
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "Manuals" and body["workspace_id"] == ws
    assert body["chunking_strategy"] == "recursive"
    assert body["embed_model"] == settings.resolved_embed_model
    assert body["document_count"] == 0


async def test_create_with_options(client, signup):
    h = await signup()
    ws = await workspace(client, h)
    r = await client.post(
        url(ws),
        json={
            "name": "Papers",
            "chunking_strategy": "semantic",
            "embed_model": "my-model",
            "description": "arXiv",
        },
        headers=h,
    )
    assert r.json()["chunking_strategy"] == "semantic" and r.json()["embed_model"] == "my-model"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"name": " "},
        {"name": "x" * 121},
        {"name": "ok", "chunking_strategy": "magic"},
        {"name": "ok", "description": "d" * 2001},
    ],
)
async def test_create_validation(client, signup, body):
    h = await signup()
    ws = await workspace(client, h)
    assert (await client.post(url(ws), json=body, headers=h)).status_code == 422


async def test_duplicate_name_is_409_case_insensitive_but_allowed_across_workspaces(client, signup):
    h = await signup()
    w1, w2 = await workspace(client, h, "One"), await workspace(client, h, "Two")
    assert (await client.post(url(w1), json={"name": "Docs"}, headers=h)).status_code == 201
    assert (await client.post(url(w1), json={"name": "docs"}, headers=h)).status_code == 409
    assert (await client.post(url(w2), json={"name": "Docs"}, headers=h)).status_code == 201


async def test_list_pagination_and_document_counts(client, app, signup):
    h = await signup()
    ws = await workspace(client, h)
    ids = []
    for n in ("a", "b", "c"):
        ids.append((await client.post(url(ws), json={"name": n}, headers=h)).json()["id"])
    async with app.state.sessionmaker() as s:
        for i in range(2):
            s.add(
                Document(
                    collection_id=uuid.UUID(ids[0]),
                    title=f"d{i}",
                    source_type="text",
                    content_hash=f"h{i}",
                )
            )
        await s.commit()
    page = (await client.get(url(ws) + "?limit=2", headers=h)).json()
    assert page["total"] == 3 and len(page["items"]) == 2
    all_items = (await client.get(url(ws), headers=h)).json()["items"]
    assert {i["name"]: i["document_count"] for i in all_items} == {"a": 2, "b": 0, "c": 0}


async def test_get_and_404s(client, signup):
    h = await signup()
    ws = await workspace(client, h)
    cid = (await client.post(url(ws), json={"name": "x"}, headers=h)).json()["id"]
    assert (await client.get(url(ws, cid), headers=h)).json()["name"] == "x"
    assert (await client.get(url(ws, uuid.uuid4()), headers=h)).status_code == 404


async def test_update_name_and_description_only(client, signup):
    h = await signup()
    ws = await workspace(client, h)
    cid = (
        await client.post(url(ws), json={"name": "x", "chunking_strategy": "fixed"}, headers=h)
    ).json()["id"]
    r = await client.patch(url(ws, cid), json={"name": "y", "description": "d"}, headers=h)
    assert r.status_code == 200 and (r.json()["name"], r.json()["description"]) == ("y", "d")
    assert r.json()["chunking_strategy"] == "fixed"
    # strategy / model are immutable: unknown fields are ignored, never applied
    r = await client.patch(
        url(ws, cid), json={"description": "z", "chunking_strategy": "semantic"}, headers=h
    )
    assert r.json()["chunking_strategy"] == "fixed"
    assert (await client.patch(url(ws, cid), json={}, headers=h)).status_code == 422


async def test_rename_to_existing_name_is_409_but_same_name_is_fine(client, signup):
    h = await signup()
    ws = await workspace(client, h)
    await client.post(url(ws), json={"name": "taken"}, headers=h)
    cid = (await client.post(url(ws), json={"name": "mine"}, headers=h)).json()["id"]
    assert (await client.patch(url(ws, cid), json={"name": "Taken"}, headers=h)).status_code == 409
    assert (await client.patch(url(ws, cid), json={"name": "MINE"}, headers=h)).status_code == 200


async def test_delete_removes_documents_and_chunks_via_database_cascade(client, app, signup):
    h = await signup()
    ws = await workspace(client, h)
    cid = uuid.UUID((await client.post(url(ws), json={"name": "x"}, headers=h)).json()["id"])
    async with app.state.sessionmaker() as s:
        doc = Document(collection_id=cid, title="d", source_type="text", content_hash="h")
        s.add(doc)
        await s.flush()
        s.add(Chunk(document_id=doc.id, ordinal=0, text="t", char_start=0, char_end=1))
        await s.commit()
    assert (await client.delete(url(ws, cid), headers=h)).status_code == 204
    async with app.state.sessionmaker() as s:
        for model in (Collection, Document, Chunk):
            assert (await s.execute(select(func.count()).select_from(model))).scalar_one() == 0
    assert (await client.get(url(ws, cid), headers=h)).status_code == 404


async def test_tenant_isolation(client, signup):
    alice, bob = await signup("a@example.com"), await signup("b@example.com")
    aw, bw = await workspace(client, alice, "Alice"), await workspace(client, bob, "Bob")
    cid = (await client.post(url(aw), json={"name": "secret"}, headers=alice)).json()["id"]
    # Bob cannot see, list, create in, edit or delete anything in Alice's workspace...
    assert (await client.get(url(aw), headers=bob)).status_code == 404
    assert (await client.post(url(aw), json={"name": "x"}, headers=bob)).status_code == 404
    assert (await client.get(url(aw, cid), headers=bob)).status_code == 404
    assert (await client.patch(url(aw, cid), json={"name": "x"}, headers=bob)).status_code == 404
    assert (await client.delete(url(aw, cid), headers=bob)).status_code == 404
    # ...and cannot reach Alice's collection through his OWN workspace id either.
    assert (await client.get(url(bw, cid), headers=bob)).status_code == 404
    assert (await client.patch(url(bw, cid), json={"name": "x"}, headers=bob)).status_code == 404
    assert (await client.delete(url(bw, cid), headers=bob)).status_code == 404
    assert (await client.get(url(aw, cid), headers=alice)).status_code == 200


@pytest.mark.parametrize(
    ("role", "read", "write", "delete"),
    [(Role.VIEWER, 200, 403, 403), (Role.MEMBER, 200, 200, 403), (Role.ADMIN, 200, 200, 204)],
)
async def test_role_matrix(client, app, signup, role, read, write, delete):
    owner, other = await signup("o@example.com"), await signup("m@example.com")
    ws = await workspace(client, owner)
    cid = (await client.post(url(ws), json={"name": "x"}, headers=owner)).json()["id"]
    async with app.state.sessionmaker() as s:
        uid = (await s.execute(select(User.id).where(User.email == "m@example.com"))).scalar_one()
        s.add(WorkspaceMember(workspace_id=uuid.UUID(ws), user_id=uid, role=role))
        await s.commit()
    assert (await client.get(url(ws, cid), headers=other)).status_code == read
    assert (
        await client.patch(url(ws, cid), json={"description": "d"}, headers=other)
    ).status_code == write
    created = await client.post(url(ws), json={"name": "new"}, headers=other)
    assert created.status_code == (201 if write == 200 else 403)
    assert (await client.delete(url(ws, cid), headers=other)).status_code == delete


async def test_deleting_workspace_removes_its_collections(client, app, signup):
    h = await signup()
    ws = await workspace(client, h)
    await client.post(url(ws), json={"name": "x"}, headers=h)
    assert (await client.delete(f"/v1/workspaces/{ws}", headers=h)).status_code == 204
    async with app.state.sessionmaker() as s:
        assert (await s.execute(select(func.count()).select_from(Collection))).scalar_one() == 0
