import uuid

import pytest
from sqlalchemy import select

from app.api.workspaces import slugify
from app.db.models import Role, WorkspaceMember


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Acme Support", "acme-support"),
        ("  Crème Brûlée!! ", "creme-brulee"),
        ("---", "workspace"),
        ("日本語", "workspace"),
        ("a" * 100, "a" * 60),
    ],
)
def test_slugify(name, slug):
    assert slugify(name) == slug


async def test_create_makes_caller_the_owner(client, signup):
    h = await signup()
    r = await client.post("/v1/workspaces", json={"name": "Acme Support"}, headers=h)
    assert r.status_code == 201
    body = r.json()
    assert (body["name"], body["slug"], body["role"], body["plan"]) == (
        "Acme Support",
        "acme-support",
        "owner",
        "free",
    )


async def test_requires_authentication(client):
    assert (await client.post("/v1/workspaces", json={"name": "x"})).status_code == 401
    assert (await client.get("/v1/workspaces")).status_code == 401


async def test_generated_slug_gets_suffix_on_collision(client, signup):
    h = await signup()
    first = (await client.post("/v1/workspaces", json={"name": "Docs"}, headers=h)).json()
    second = (await client.post("/v1/workspaces", json={"name": "Docs"}, headers=h)).json()
    assert first["slug"] == "docs"
    assert second["slug"].startswith("docs-") and second["slug"] != "docs"


async def test_explicit_slug_collision_is_409(client, signup):
    h = await signup()
    await client.post("/v1/workspaces", json={"name": "A", "slug": "taken"}, headers=h)
    r = await client.post("/v1/workspaces", json={"name": "B", "slug": "taken"}, headers=h)
    assert r.status_code == 409


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"name": ""},
        {"name": "   "},
        {"name": "x" * 121},
        {"name": "ok", "slug": "Bad Slug"},
        {"name": "ok", "slug": "a"},
    ],
)
async def test_create_validation(client, signup, body):
    h = await signup()
    assert (await client.post("/v1/workspaces", json=body, headers=h)).status_code == 422


async def test_list_returns_only_my_workspaces_with_pagination(client, signup):
    a, b = await signup("a@example.com"), await signup("b@example.com")
    for n in ("one", "two", "three"):
        await client.post("/v1/workspaces", json={"name": n}, headers=a)
    await client.post("/v1/workspaces", json={"name": "bobs"}, headers=b)
    r = await client.get("/v1/workspaces?limit=2", headers=a)
    assert r.json()["total"] == 3 and len(r.json()["items"]) == 2
    page2 = await client.get("/v1/workspaces?limit=2&offset=2", headers=a)
    assert len(page2.json()["items"]) == 1
    assert {w["name"] for w in (await client.get("/v1/workspaces", headers=b)).json()["items"]} == {
        "bobs"
    }
    assert (await client.get("/v1/workspaces?limit=101", headers=a)).status_code == 422


async def test_non_member_gets_404_not_403(client, signup):
    owner, stranger = await signup("o@example.com"), await signup("s@example.com")
    ws = (await client.post("/v1/workspaces", json={"name": "Private"}, headers=owner)).json()
    for call in (
        client.get(f"/v1/workspaces/{ws['id']}", headers=stranger),
        client.patch(f"/v1/workspaces/{ws['id']}", json={"name": "x"}, headers=stranger),
        client.delete(f"/v1/workspaces/{ws['id']}", headers=stranger),
    ):
        assert (await call).status_code == 404
    unknown = await client.get(f"/v1/workspaces/{uuid.uuid4()}", headers=owner)
    assert unknown.status_code == 404
    assert (await client.get("/v1/workspaces/not-a-uuid", headers=owner)).status_code == 422


async def _add_member(app, workspace_id, user_email, role):
    from app.db.models import User

    async with app.state.sessionmaker() as s:
        uid = (await s.execute(select(User.id).where(User.email == user_email))).scalar_one()
        s.add(WorkspaceMember(workspace_id=uuid.UUID(workspace_id), user_id=uid, role=role))
        await s.commit()


@pytest.mark.parametrize(
    ("role", "can_rename", "can_delete"),
    [(Role.VIEWER, False, False), (Role.MEMBER, False, False), (Role.ADMIN, True, False)],
)
async def test_role_gates(client, app, signup, role, can_rename, can_delete):
    owner, other = await signup("o@example.com"), await signup("m@example.com")
    ws = (await client.post("/v1/workspaces", json={"name": "Team"}, headers=owner)).json()
    await _add_member(app, ws["id"], "m@example.com", role)
    got = await client.get(f"/v1/workspaces/{ws['id']}", headers=other)
    assert got.status_code == 200 and got.json()["role"] == role.value
    rename = await client.patch(f"/v1/workspaces/{ws['id']}", json={"name": "New"}, headers=other)
    assert rename.status_code == (200 if can_rename else 403)
    delete = await client.delete(f"/v1/workspaces/{ws['id']}", headers=other)
    assert delete.status_code == (204 if can_delete else 403)


async def test_owner_can_rename_and_delete_and_it_is_gone(client, signup):
    h = await signup()
    ws = (
        await client.post("/v1/workspaces", json={"name": "Old", "slug": "keep-me"}, headers=h)
    ).json()
    renamed = await client.patch(f"/v1/workspaces/{ws['id']}", json={"name": "  New  "}, headers=h)
    assert renamed.json()["name"] == "New" and renamed.json()["slug"] == "keep-me"
    assert (await client.delete(f"/v1/workspaces/{ws['id']}", headers=h)).status_code == 204
    assert (await client.get(f"/v1/workspaces/{ws['id']}", headers=h)).status_code == 404
    assert (await client.get("/v1/workspaces", headers=h)).json()["total"] == 0
