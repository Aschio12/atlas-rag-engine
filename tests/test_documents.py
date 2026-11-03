import io
import uuid

import pytest

BASE = "/v1/workspaces/{ws}/collections/{col}/documents"


@pytest.fixture
async def space(client, signup):
    """A signed-in owner with a workspace and one empty collection."""
    headers = await signup()
    ws = (await client.post("/v1/workspaces", json={"name": "Acme"}, headers=headers)).json()
    col = (
        await client.post(
            f"/v1/workspaces/{ws['id']}/collections", json={"name": "Manuals"}, headers=headers
        )
    ).json()
    return {
        "headers": headers,
        "ws": ws["id"],
        "col": col["id"],
        "url": BASE.format(ws=ws["id"], col=col["id"]),
    }


def _file(name="notes.txt", data=b"Refunds are issued within 14 days.", ctype="text/plain"):
    return {"file": (name, io.BytesIO(data), ctype)}


async def test_upload_text_file(client, space):
    r = await client.post(space["url"], files=_file(), headers=space["headers"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["title"] == "notes"
    assert body["filename"] == "notes.txt"
    assert body["kind"] == "text"
    assert body["source_type"] == "file"
    assert body["status"] == "pending"
    assert body["size_bytes"] == len(b"Refunds are issued within 14 days.")
    assert len(body["content_hash"]) == 64


async def test_upload_uses_explicit_title(client, space):
    r = await client.post(
        space["url"], files=_file(), data={"title": "Refund policy"}, headers=space["headers"]
    )
    assert r.json()["title"] == "Refund policy"


async def test_upload_sanitises_filename(client, space):
    r = await client.post(
        space["url"], files=_file("../../etc/passwd.txt"), headers=space["headers"]
    )
    assert r.json()["filename"] == "passwd.txt"


async def test_duplicate_content_is_rejected_with_the_existing_id(client, space):
    first = (await client.post(space["url"], files=_file(), headers=space["headers"])).json()
    r = await client.post(space["url"], files=_file("renamed.txt"), headers=space["headers"])
    assert r.status_code == 409
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["existing_document_id"] == first["id"]


async def test_same_content_is_allowed_in_another_collection(client, space):
    await client.post(space["url"], files=_file(), headers=space["headers"])
    other = (
        await client.post(
            f"/v1/workspaces/{space['ws']}/collections",
            json={"name": "Other"},
            headers=space["headers"],
        )
    ).json()
    r = await client.post(
        BASE.format(ws=space["ws"], col=other["id"]), files=_file(), headers=space["headers"]
    )
    assert r.status_code == 201


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("evil.pdf", b"MZ\x90\x00\x03"),
        ("photo.png", b"\x89PNG\r\n\x1a\n\x00\x00"),
        ("blank.txt", b"   \n"),
        ("program.exe", b"hello"),
    ],
)
async def test_unsupported_files_get_415(client, space, name, data):
    files = _file(name, data, "application/octet-stream")
    r = await client.post(space["url"], files=files, headers=space["headers"])
    assert r.status_code == 415, r.text
    assert r.json()["type"].endswith("unsupported_media_type")


async def test_oversize_upload_is_413_and_nothing_is_stored(client, app, space):
    app.state.settings.max_upload_mb = 1
    big = b"a" * (1024 * 1024 + 1)
    r = await client.post(space["url"], files=_file("big.txt", big), headers=space["headers"])
    assert r.status_code == 413
    listing = await client.get(space["url"], headers=space["headers"])
    assert listing.json()["total"] == 0


async def test_upload_requires_a_file(client, space):
    r = await client.post(space["url"], headers=space["headers"])
    assert r.status_code == 422


async def test_create_text_document(client, space):
    r = await client.post(
        space["url"] + "/text",
        json={"title": "  Shipping  ", "text": "We ship worldwide.", "format": "markdown"},
        headers=space["headers"],
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["title"] == "Shipping"
    assert body["source_type"] == "text"
    assert body["kind"] == "markdown"


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "", "text": "x"},
        {"title": "  ", "text": "x"},
        {"title": "t", "text": "   "},
        {"title": "t", "text": "x", "format": "docx"},
    ],
)
async def test_create_text_document_validation(client, space, payload):
    r = await client.post(space["url"] + "/text", json=payload, headers=space["headers"])
    assert r.status_code == 422


async def test_list_filters_and_paginates(client, space):
    for i in range(3):
        await client.post(
            space["url"],
            files=_file(f"d{i}.txt", f"doc number {i}".encode()),
            headers=space["headers"],
        )
    r = await client.get(space["url"], params={"limit": 2}, headers=space["headers"])
    assert r.json()["total"] == 3
    assert len(r.json()["items"]) == 2
    r = await client.get(space["url"], params={"status": "ready"}, headers=space["headers"])
    assert r.json() == {"items": [], "total": 0}
    r = await client.get(space["url"], params={"status": "bogus"}, headers=space["headers"])
    assert r.status_code == 422


async def test_get_document_and_download_original(client, space):
    created = (
        await client.post(
            space["url"],
            files=_file("Ünïcode name.txt", b"original bytes"),
            headers=space["headers"],
        )
    ).json()
    r = await client.get(f"{space['url']}/{created['id']}", headers=space["headers"])
    assert r.status_code == 200
    assert r.json()["id"] == created["id"]

    r = await client.get(f"{space['url']}/{created['id']}/original", headers=space["headers"])
    assert r.status_code == 200
    assert r.content == b"original bytes"
    assert r.headers["content-disposition"].startswith("attachment;")
    assert "UTF-8''%C3%9Cn%C3%AFcode%20name.txt" in r.headers["content-disposition"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-type"] == "application/octet-stream"


async def test_uploaded_html_is_never_served_inline(client, space):
    created = (
        await client.post(
            space["url"],
            files=_file("page.html", b"<html><script>alert(1)</script></html>", "text/html"),
            headers=space["headers"],
        )
    ).json()
    r = await client.get(f"{space['url']}/{created['id']}/original", headers=space["headers"])
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["content-disposition"].startswith("attachment")


async def test_delete_removes_row_and_files(client, app, space):
    created = (await client.post(space["url"], files=_file(), headers=space["headers"])).json()
    blobs = app.state.blobs
    key = f"{space['ws']}/{space['col']}/{created['id']}/original"
    assert await blobs.exists(key)

    r = await client.delete(f"{space['url']}/{created['id']}", headers=space["headers"])
    assert r.status_code == 204
    assert not await blobs.exists(key)
    r = await client.get(f"{space['url']}/{created['id']}", headers=space["headers"])
    assert r.status_code == 404
    # The same content can be uploaded again once the first copy is gone.
    r = await client.post(space["url"], files=_file(), headers=space["headers"])
    assert r.status_code == 201


async def test_unknown_document_and_collection_are_404(client, space):
    missing = "00000000-0000-4000-8000-000000000000"
    h = space["headers"]
    assert (await client.get(f"{space['url']}/{missing}", headers=h)).status_code == 404
    assert (await client.delete(f"{space['url']}/{missing}", headers=h)).status_code == 404
    bad = BASE.format(ws=space["ws"], col=missing)
    assert (await client.get(bad, headers=h)).status_code == 404
    assert (await client.post(bad, files=_file(), headers=h)).status_code == 404


async def test_requires_authentication(client, space):
    assert (await client.get(space["url"])).status_code == 401
    assert (await client.post(space["url"], files=_file())).status_code == 401


async def test_other_tenants_get_404_everywhere(client, signup, space):
    doc = (await client.post(space["url"], files=_file(), headers=space["headers"])).json()
    eve = await signup("eve@example.com")
    for method, url in [
        ("GET", space["url"]),
        ("GET", f"{space['url']}/{doc['id']}"),
        ("GET", f"{space['url']}/{doc['id']}/original"),
        ("DELETE", f"{space['url']}/{doc['id']}"),
    ]:
        r = await client.request(method, url, headers=eve)
        assert r.status_code == 404, (method, url)
    r = await client.post(space["url"], files=_file("x.txt", b"x"), headers=eve)
    assert r.status_code == 404


async def test_a_collection_id_from_another_workspace_does_not_leak(client, space):
    headers = space["headers"]
    ws2 = (await client.post("/v1/workspaces", json={"name": "Second"}, headers=headers)).json()
    doc = (await client.post(space["url"], files=_file(), headers=headers)).json()
    crossed = BASE.format(ws=ws2["id"], col=space["col"])
    assert (await client.get(crossed, headers=headers)).status_code == 404
    assert (await client.get(f"{crossed}/{doc['id']}", headers=headers)).status_code == 404


async def test_viewers_can_read_but_not_write(client, app, signup, space):
    from sqlalchemy import select

    from app.db.models import Role, User, WorkspaceMember

    doc = (await client.post(space["url"], files=_file(), headers=space["headers"])).json()
    viewer_headers = await signup("viewer@example.com")
    async with app.state.sessionmaker() as s:
        uid = (
            await s.execute(select(User.id).where(User.email == "viewer@example.com"))
        ).scalar_one()
        s.add(WorkspaceMember(workspace_id=uuid.UUID(space["ws"]), user_id=uid, role=Role.VIEWER))
        await s.commit()

    assert (await client.get(space["url"], headers=viewer_headers)).status_code == 200
    r = await client.get(f"{space['url']}/{doc['id']}/original", headers=viewer_headers)
    assert r.status_code == 200
    r = await client.post(space["url"], files=_file("n.txt", b"n"), headers=viewer_headers)
    assert r.status_code == 403
    r = await client.delete(f"{space['url']}/{doc['id']}", headers=viewer_headers)
    assert r.status_code == 403


async def test_documents_are_counted_on_the_collection(client, space):
    await client.post(space["url"], files=_file(), headers=space["headers"])
    r = await client.get(
        f"/v1/workspaces/{space['ws']}/collections/{space['col']}", headers=space["headers"]
    )
    assert r.json()["document_count"] == 1


async def test_losing_a_duplicate_race_cleans_up_and_returns_409(client, app, space, monkeypatch):
    """Two identical uploads pass the pre-check together; the unique constraint settles it."""
    from app.api import documents

    first = (await client.post(space["url"], files=_file(), headers=space["headers"])).json()
    real = documents._find_by_hash
    calls = {"n": 0}

    async def blind_first_time(session, collection_id, content_hash):
        calls["n"] += 1
        return None if calls["n"] == 1 else await real(session, collection_id, content_hash)

    monkeypatch.setattr(documents, "_find_by_hash", blind_first_time)
    r = await client.post(space["url"], files=_file("again.txt"), headers=space["headers"])
    assert r.status_code == 409
    assert r.json()["existing_document_id"] == first["id"]
    # The loser's original was removed: only the first document's folder remains.
    folder = app.state.blobs.root / space["ws"] / space["col"]
    assert [p.name for p in folder.iterdir()] == [first["id"]]
